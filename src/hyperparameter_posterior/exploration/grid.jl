export explore_hyperparameter_posterior

using StatsFuns: logsumexp

"""
    explore_half_axis_by_steps(model, y, transform, mode_logpdf, dim, direction, evaluation_step_z, max_log_drop, interpolation_subdivisions, marginalization_method, marginalization_indices, accumulators)

Explores one direction along a dimension using integer step indices.
Returns a list of (key, GridPoint) tuples where key is an integer coordinate tuple.
"""
function explore_half_axis_by_steps(
        model, y, transform::ReparameterizationTransform, mode_logpdf,
        dim::Int, direction::Int, evaluation_step_z::Float64, max_log_drop::Float64, interpolation_subdivisions::Int,
        marginalization_method, marginalization_indices, accumulators::Tuple,
        accumulator_call_keys::Vector
    )
    n_dim = length(transform.θ_star)
    keyed_points = []

    step_count = 1
    while true
        key_vec = zeros(Int, n_dim)
        key_vec[dim] = direction * step_count
        key = Tuple(key_vec)

        z_vec = evaluation_step_z .* collect(key)
        θ_test = transform(z_vec)  # Returns WorkingHyperparameters directly

        is_integration_point = (step_count % interpolation_subdivisions == 0)

        result = evaluate_at_grid_point(
            model, y, θ_test;
            compute_marginals = is_integration_point,
            marginalization_method = marginalization_method,
            marginalization_indices = marginalization_indices
        )

        if mode_logpdf - result.log_density > max_log_drop
            break
        end

        point = GridPoint(θ_test, result.log_density, result.marginal_result)

        # Call accumulators eagerly and record the grid key for ordering
        if is_integration_point && !isempty(accumulators) && result.log_density > -Inf
            for acc in accumulators
                summary = compute_point_summary(acc; result...)
                if summary !== nothing
                    accumulate!(acc, summary; is_mode = false)
                end
            end
            push!(accumulator_call_keys, key)
        end

        push!(keyed_points, (key, point))
        step_count += 1
    end

    return keyed_points
end

"""
    explore_dimension_and_build_lookup(model, y, transform, mode_logpdf, dim, evaluation_step_z, max_log_drop, interpolation_subdivisions, marginalization_method, marginalization_indices, accumulators)

Explore a single dimension and build lookup table for integer-based grid coordinates.
Calls explore_half_axis_by_steps for both directions and collects results.

Returns (point_lookup, step_range) where:
- point_lookup: Dictionary mapping integer tuple keys to GridPoint objects
- step_range: Range of integer steps found for this dimension
"""
function explore_dimension_and_build_lookup(
        model, y, transform::ReparameterizationTransform, mode_logpdf,
        dim::Int, evaluation_step_z::Float64, max_log_drop::Float64, interpolation_subdivisions::Int,
        marginalization_method, marginalization_indices, accumulators::Tuple,
        accumulator_call_keys::Vector
    )
    # Call our helper function for both directions
    pos_points = explore_half_axis_by_steps(
        model, y, transform, mode_logpdf,
        dim, 1, evaluation_step_z, max_log_drop, interpolation_subdivisions,
        marginalization_method, marginalization_indices, accumulators,
        accumulator_call_keys
    )
    neg_points = explore_half_axis_by_steps(
        model, y, transform, mode_logpdf,
        dim, -1, evaluation_step_z, max_log_drop, interpolation_subdivisions,
        marginalization_method, marginalization_indices, accumulators,
        accumulator_call_keys
    )

    # Return a dictionary for fast, safe lookups
    point_lookup = Dict(vcat(pos_points, neg_points))

    # And the range of integer steps it found for this dimension
    step_indices = [p[1][dim] for p in vcat(pos_points, neg_points)]
    push!(step_indices, 0) # Include the mode
    step_range = minimum(step_indices):maximum(step_indices)

    return point_lookup, step_range
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
- `model::INLAModel`: The INLA model
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
        model::INLAModel, y, θ_star::WorkingHyperparameters, marginalization_method, marginalization_indices;
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

    # Step 1: Compute the transformation object
    progress_callback(status = "Computing reparameterization", dimensions = n_dim)
    transform = compute_reparameterization(model, y, θ_star; executor = executor, diff_strategy = diff_strategy)

    # Step 2: Evaluate the mode point once, authoritatively
    progress_callback(status = "Evaluating mode point", mode = θ_star)
    mode_result = evaluate_at_grid_point(
        model, y, θ_star; compute_marginals = true, marginalization_method, marginalization_indices
    )
    mode_point = GridPoint(θ_star, mode_result.log_density, mode_result.marginal_result)
    mode_log_density = mode_result.log_density

    # Track the order in which accumulators are called (grid coordinate keys).
    # This lets us compute the permutation from call order → grid order in finalize!.
    mode_key = ntuple(_ -> 0, n_dim)
    accumulator_call_keys = NTuple{n_dim, Int}[mode_key]

    # Call accumulators for the mode point (two-phase: compute summary, then accumulate)
    if !isempty(accumulators) && mode_log_density > -Inf
        for acc in accumulators
            summary = compute_point_summary(acc; mode_result...)
            if summary !== nothing
                accumulate!(acc, summary; is_mode = true)
            end
        end
    end

    # Step 3: Explore axes and build the lookup table of raw (unnormalized) points
    progress_callback(status = "Starting axis exploration", dimensions = n_dim)
    point_lookup = Dict{NTuple{n_dim, Int}, typeof(mode_point)}()
    point_lookup[mode_key] = mode_point
    step_ranges_per_dim = Vector{UnitRange{Int}}(undef, n_dim)
    evaluation_step_z = integration_step_z / interpolation_subdivisions

    for d in 1:n_dim
        progress_callback(status = "Exploring axis", current_dimension = d, total_dimensions = n_dim)
        axis_points, axis_range = explore_dimension_and_build_lookup(
            model, y, transform, mode_log_density, d, evaluation_step_z, max_log_drop,
            interpolation_subdivisions, marginalization_method, marginalization_indices, accumulators,
            accumulator_call_keys
        )
        merge!(point_lookup, axis_points) # Add all on-axis points to the master table
        step_ranges_per_dim[d] = axis_range
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

    # Phase 1: Evaluate off-axis points (PARALLEL)
    off_axis_results = pmap_executor(off_axis_work, executor) do item
        result = evaluate_at_grid_point(
            model, y, item.θ;
            compute_marginals = item.is_integration,
            marginalization_method = marginalization_method,
            marginalization_indices = marginalization_indices
        )
        summaries = if result.log_density > -Inf
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
            if mode_log_density - r.log_density <= max_log_drop
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
