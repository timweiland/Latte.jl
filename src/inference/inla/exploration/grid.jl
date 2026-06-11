export explore_hyperparameter_posterior

using StatsFuns: logsumexp

function _grid_tail_exceeded(mode_logpdf::Real, log_density::Real, max_log_drop::Real)
    isfinite(mode_logpdf) || return true
    isfinite(log_density) || return true
    return mode_logpdf - log_density > max_log_drop
end

"""
    explore_half_axis_by_steps(model, y, transform, mode_logpdf, dim, direction, evaluation_step_z, max_log_drop, interpolation_subdivisions, marginalization_method, marginalization_indices, accumulators, ws; x0)

Walk one direction along one reparameterized axis by integer step indices until
the log-density drops more than `max_log_drop` below the mode.

Pure: returns a list of `(key, GridPoint, summaries)` tuples (key = integer
coordinate tuple). `summaries` is the per-accumulator `compute_point_summary`
output (thread-safe) for integration points with finite density, else `nothing`;
the stateful `accumulate!` is deferred to a serial pass in the caller so the
half-axes can be walked concurrently without racing on accumulator state.
"""
function explore_half_axis_by_steps(
        model, y, transform::ReparameterizationTransform, mode_logpdf,
        dim::Int, direction::Int, evaluation_step_z::Float64, max_log_drop::Float64, interpolation_subdivisions::Int,
        marginalization_method, marginalization_indices, accumulators::Tuple, ws;
        x0 = nothing,
    )
    n_dim = length(transform.θ_star)
    keyed_points = []

    step_count = 1
    while true
        key_vec = zeros(Int, n_dim)
        key_vec[dim] = direction * step_count
        key = Tuple(key_vec)

        θ_test = transform(evaluation_step_z .* collect(key))  # WorkingHyperparameters

        is_integration_point = (step_count % interpolation_subdivisions == 0)

        result = evaluate_at_grid_point(
            model, y, θ_test;
            ws = ws, x0 = x0,
            compute_marginals = is_integration_point,
            marginalization_method = marginalization_method,
            marginalization_indices = marginalization_indices,
        )

        if _grid_tail_exceeded(mode_logpdf, result.log_density, max_log_drop)
            break
        end

        point = GridPoint(θ_test, result.log_density, result.marginal_result)

        # Pure: compute the (thread-safe) per-accumulator summaries now, but defer
        # the stateful `accumulate!`/key-recording to the caller's serial pass.
        summaries = (is_integration_point && !isempty(accumulators) && isfinite(result.log_density)) ?
            map(acc -> compute_point_summary(acc; result...), accumulators) : nothing

        push!(keyed_points, (key, point, summaries))
        step_count += 1
    end

    return keyed_points
end

"""
    explore_hyperparameter_posterior(strategy, model, y, θ_star, marginalization_method, marginalization_indices; kwargs...)

Explore the hyperparameter posterior around the mode `θ_star`.

Dispatches on the `strategy` argument:
- `GridExplorationStrategy`: Cartesian grid exploration
- `CCDExplorationStrategy`: Central Composite Design exploration
- `AutoExplorationStrategy`: Automatic selection (grid for D ≤ 2, CCD for D ≥ 3)

# Arguments
- `strategy::ExplorationStrategy`: Exploration strategy (controls grid layout and parameters)
- `model::LatentGaussianModel`: The INLA model
- `y`: Observed data
- `θ_star::WorkingHyperparameters`: The posterior mode in working space
- `marginalization_method`: Method for latent marginalization at each grid point
- `marginalization_indices`: Indices of latent field to marginalize

# Keyword Arguments
- `progress_callback`: Optional function for progress updates with signature `f(; kwargs...)`
- `accumulators::Tuple = ()`: Tuple of PosteriorAccumulator objects to process integration points

# Returns
- `AbstractHyperparameterExploration`: Exploration results (GridExploration or CCDExploration)
- `accumulators`: Tuple of finalized accumulators (if provided)
"""
function explore_hyperparameter_posterior(
        strategy::GridExplorationStrategy,
        model::LatentGaussianModel, y, θ_star::WorkingHyperparameters, marginalization_method, marginalization_indices;
        progress_callback = nothing,
        accumulators::Tuple = (),
        executor::ParallelExecutor = SequentialExecutor(),
        diff_strategy::DifferentiationStrategy = ADStrategy()
    )
    integration_step_z = strategy.integration_step_z
    max_log_drop = strategy.max_log_drop
    interpolation_subdivisions = strategy.interpolation_subdivisions
    # Handle progress callback
    if progress_callback === nothing
        progress_callback = (; kwargs...) -> nothing
    end

    n_dim = length(θ_star)

    # Build a pool sized for the active executor. Sequential phases below
    # check out a single workspace via `with_workspace`; the parallel
    # off-axis loop uses the pool-aware `pmap_executor` so each task gets
    # its own workspace without racing on a shared one.
    θ_star_nt = convert(NamedTuple, convert(NaturalHyperparameters, θ_star))
    pool = make_workspace_pool(model.latent_prior; size = _pool_size(executor), θ_star_nt...)

    # Step 1: Compute the transformation object. `compute_reparameterization`
    # manages its own pool-aware workspace checkouts internally (for the
    # finite-diff or AD Hessian evaluations), so we pass the pool directly.
    progress_callback(status = "Computing reparameterization", dimensions = n_dim)
    transform = compute_reparameterization(model, y, θ_star; pool = pool, executor = executor, diff_strategy = diff_strategy)

    mode_key = ntuple(_ -> 0, n_dim)
    accumulator_call_keys = NTuple{n_dim, Int}[mode_key]
    point_lookup = Dict{NTuple{n_dim, Int}, GridPoint{typeof(θ_star)}}()
    step_ranges_per_dim = Vector{UnitRange{Int}}(undef, n_dim)
    evaluation_step_z = integration_step_z / interpolation_subdivisions
    mode_log_density = 0.0
    # Warm-start seed for the axis-walk + off-axis GA solves: the latent mode
    # x*(θ*), captured from the mode-point evaluation below. Seeding every grid
    # point's GA from it halves their Newton iterations (the latent mode barely
    # moves across the grid) without changing the converged per-θ marginals.
    x0_seed = nothing

    # Step 2: mode-point evaluation (serial — captures the warm-start seed x0_seed
    # and the mode log-density that the axis walks key off). The exploration
    # phase's bar budget: axis walk → 0..30%, off-axis eval → 30..95%, assembly → 95..100%.
    mode_point = with_workspace(pool) do ws
        progress_callback(status = "Evaluating mode point", mode = θ_star)
        mode_result = evaluate_at_grid_point(
            model, y, θ_star; ws = ws, compute_marginals = true, marginalization_method, marginalization_indices
        )
        mp = GridPoint(θ_star, mode_result.log_density, mode_result.marginal_result)
        mode_log_density = mode_result.log_density
        x0_seed = mode_result.x_star
        if !isfinite(mode_log_density)
            throw(
                ArgumentError(
                    "Grid exploration cannot start because the mode point evaluated to non-finite log density ($(mode_log_density)).",
                )
            )
        end

        # Call accumulators for the mode point
        if !isempty(accumulators) && isfinite(mode_log_density)
            for acc in accumulators
                summary = compute_point_summary(acc; mode_result...)
                if summary !== nothing
                    accumulate!(acc, summary; is_mode = true)
                end
            end
        end

        point_lookup[mode_key] = mp
        mp
    end

    # Step 3: on-axis exploration. The 2·n_dim half-axes are mutually independent
    # (each walk is internally serial via the density stop test, but they don't
    # interact), so dispatch them through the executor on per-task workspaces,
    # warm-started from the mode. `explore_half_axis_by_steps` is pure; we merge
    # its points, derive the per-dim step ranges, and run the deferred accumulation
    # serially below.
    progress_callback(status = "Starting axis exploration", dimensions = n_dim, progress = 0.0)
    axis_units = [(d, dir) for d in 1:n_dim for dir in (1, -1)]
    n_axis = length(axis_units)
    axis_results = pmap_executor(
        axis_units, executor, pool;
        on_complete = function (done)
            return progress_callback(
                status = "Exploring axes", axes_explored = done, total_axes = n_axis,
                progress = 0.3 * done / n_axis,
            )
        end,
    ) do unit, ws
        explore_half_axis_by_steps(
            model, y, transform, mode_log_density, unit[1], unit[2],
            evaluation_step_z, max_log_drop, interpolation_subdivisions,
            marginalization_method, marginalization_indices, accumulators, ws; x0 = x0_seed,
        )
    end

    for halfaxis in axis_results, (key, point, _) in halfaxis
        point_lookup[key] = point
    end
    for d in 1:n_dim
        lo = 0
        hi = 0
        for halfaxis in axis_results, (key, _, _) in halfaxis
            s = key[d]
            s < lo && (lo = s)
            s > hi && (hi = s)
        end
        step_ranges_per_dim[d] = lo:hi
    end
    # Deferred accumulation in a deterministic (unit, step) order — the
    # accumulator_reorder pass later remaps these calls to grid order, so the
    # call order is free as long as keys are pushed alongside the accumulate! calls.
    for halfaxis in axis_results, (key, _, summaries) in halfaxis
        if summaries !== nothing
            for (acc, summary) in zip(accumulators, summaries)
                summary !== nothing && accumulate!(acc, summary; is_mode = false)
            end
            push!(accumulator_call_keys, key)
        end
    end

    # Step 4: Build the full grid by evaluating off-axis points
    total_grid_points = prod(length.(step_ranges_per_dim))
    progress_callback(status = "Building full grid", estimated_total_points = total_grid_points)

    raw_interpolation_points = typeof(mode_point)[]

    # Separate on-axis (already computed) from off-axis (need evaluation)
    # We must preserve grid iteration order for raw_interpolation_points
    grid_keys_in_order = collect(Iterators.product(step_ranges_per_dim...))

    # Collect off-axis work items
    off_axis_work = NamedTuple[]
    off_axis_indices = Int[]  # position in grid_keys_in_order
    for (idx, key_tuple) in enumerate(grid_keys_in_order)
        if !haskey(point_lookup, key_tuple)
            is_integration = all(iszero, collect(key_tuple) .% interpolation_subdivisions)
            θ = transform(evaluation_step_z .* collect(key_tuple))
            push!(off_axis_work, (; key_tuple, θ, is_integration))
            push!(off_axis_indices, idx)
        end
    end

    # Phase 1: Evaluate off-axis points (PARALLEL with per-task workspaces).
    # The `on_complete` callback fires per item as each evaluation finishes
    # so the user sees real progress during the long parallel step (the
    # default would block on "Building full grid" until pmap returned).
    n_off_axis = length(off_axis_work)
    off_axis_results = pmap_executor(
        off_axis_work, executor, pool;
        on_complete = n_off_axis > 0 ? function (done)
                return progress_callback(
                    status = "Evaluating off-axis points",
                    points_evaluated = done,
                    total_points = n_off_axis,
                    progress = 0.3 + 0.65 * done / n_off_axis,
                )
        end : nothing,
    ) do item, ws
        result = evaluate_at_grid_point(
            model, y, item.θ;
            ws = ws, x0 = x0_seed,
            compute_marginals = item.is_integration,
            marginalization_method = marginalization_method,
            marginalization_indices = marginalization_indices,
        )
        summaries = if isfinite(result.log_density)
            map(accumulators) do acc
                compute_point_summary(acc; result...)
            end
        else
            map(_ -> nothing, accumulators)
        end
        return (; result..., item.key_tuple, item.θ, item.is_integration, summaries)
    end

    # Build lookup of off-axis results by grid index
    off_axis_result_map = Dict{Int, eltype(off_axis_results)}()
    for (i, grid_idx) in enumerate(off_axis_indices)
        off_axis_result_map[grid_idx] = off_axis_results[i]
    end

    # Phase 2: Assemble grid in iteration order + accumulate (SEQUENTIAL)
    for (idx, key_tuple) in enumerate(grid_keys_in_order)
        if haskey(point_lookup, key_tuple)
            push!(raw_interpolation_points, point_lookup[key_tuple])
        elseif haskey(off_axis_result_map, idx)
            r = off_axis_result_map[idx]
            if isfinite(r.log_density) && mode_log_density - r.log_density <= max_log_drop
                push!(raw_interpolation_points, GridPoint(r.θ, r.log_density, r.marginal_result))

                if r.is_integration && !isempty(accumulators)
                    for (acc, summary) in zip(accumulators, r.summaries)
                        if summary !== nothing
                            accumulate!(acc, summary; is_mode = false)
                        end
                    end
                    push!(accumulator_call_keys, r.key_tuple)
                end
            end

            progress_callback(
                status = "Evaluating grid points",
                points_evaluated = idx,
                total_points = total_grid_points,
                current_θ = r.θ,
                log_density = r.log_density
            )
        end
    end

    # Step 5: Compute permutation from grid order → accumulator call order.
    # Accumulators were called eagerly in discovery order (mode, neg axis, pos axis, off-axis),
    # but integration_indices follows grid order (Iterators.product iteration).
    # The permutation lets get_integration_weights reorder weights to match call order.
    accumulator_reorder = Int[]
    if !isempty(accumulators)
        # Build grid-order position index via a single pass over the grid
        call_key_set = Set(accumulator_call_keys)
        grid_position = Dict{NTuple{n_dim, Int}, Int}()
        grid_idx = 0
        for key_tuple in Iterators.product(step_ranges_per_dim...)
            if key_tuple in call_key_set
                grid_idx += 1
                grid_position[key_tuple] = grid_idx
            end
        end
        # accumulator_reorder[k] = grid position of the k-th accumulator call
        # get_integration_weights uses: weights_grid[accumulator_reorder] → call order
        accumulator_reorder = [grid_position[k] for k in accumulator_call_keys]
    end

    # Step 6: Compute the normalization constant using the Jacobian
    progress_callback(status = "Computing normalization", total_explored_points = length(raw_interpolation_points))
    integration_indices = findall(p -> p.marginal_result !== nothing, raw_interpolation_points)
    unnormalized_integration_logpdfs = [p.log_density for p in raw_interpolation_points[integration_indices]]

    log_z_cell_volume = n_dim * log(integration_step_z)
    log_normalization_constant = logsumexp(unnormalized_integration_logpdfs) + logdet_jacobian(transform) + log_z_cell_volume

    # Step 6: Create the final, clean, normalized GridPoint objects for the user
    progress_callback(status = "Finalizing exploration", integration_points = length(integration_indices))

    # Warn if exploration found very few points
    if length(raw_interpolation_points) < 3
        @warn "Exploration found only $(length(raw_interpolation_points)) grid points. Consider relaxing exploration parameters: increase max_log_drop (current: $max_log_drop) or decrease integration_step_z (current: $integration_step_z) for better coverage."
    end

    final_grid_points = typeof(mode_point)[]
    for p in raw_interpolation_points
        normalized_log_density = p.log_density - log_normalization_constant
        push!(final_grid_points, GridPoint(p.θ, normalized_log_density, p.marginal_result))
    end

    # Create exploration object
    exploration = GridExploration(
        final_grid_points,
        integration_indices,
        transform,
        log_normalization_constant;
        accumulator_reorder = isempty(accumulator_reorder) ? collect(1:length(integration_indices)) : accumulator_reorder
    )

    # Finalize accumulators
    if !isempty(accumulators)
        progress_callback(status = "Finalizing accumulators", n_accumulators = length(accumulators))
        for acc in accumulators
            finalize!(acc, exploration)
        end
    end

    progress_callback(status = "Exploration complete", final_points = length(final_grid_points))

    return exploration, accumulators
end
