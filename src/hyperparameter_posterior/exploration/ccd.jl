"""
    ccd_integration_weights(n_points::Int, d::Int, f0::Float64) -> (w_sphere, w_center)

Compute CCD integration weights following Rue et al. (2009), Section 6.5.

Returns `(w_sphere, w_center)` where `w_sphere` is the weight for each non-center
design point and `w_center` is the weight for the center point (mode).

The weights are derived by requiring the quadrature to be exact for
∫1·π(z)dz = 1 and ∫zᵀz·π(z)dz = d when π(z) = N(0,I).
"""
function ccd_integration_weights(n_points::Int, d::Int, f0::Float64)
    f = f0 * sqrt(Float64(d))
    w_sphere = 1.0 / ((n_points - 1) * (1.0 + exp(-0.5 * f^2) * (f^2 / d - 1.0)))
    w_center = 1.0 - (n_points - 1) * w_sphere
    return w_sphere, w_center
end

"""
    generate_factorial_points(d::Int) -> Vector{Vector{Float64}}

Generate factorial design points (all combinations of ±1) for `d` dimensions.
For d ≤ 4, returns full factorial (2^d points).
For d > 4, returns a fractional factorial design that is symmetric (if z is included, -z is too).
"""
function generate_factorial_points(d::Int)
    if d <= 4
        # Full factorial: all 2^d combinations of ±1
        n = 2^d
        points = Vector{Vector{Float64}}(undef, n)
        for i in 0:(n - 1)
            point = Vector{Float64}(undef, d)
            for j in 1:d
                point[j] = (i >> (j - 1)) & 1 == 0 ? -1.0 : 1.0
            end
            points[i + 1] = point
        end
        return points
    else
        # Fractional factorial: generate base points from (d-2) dimensions,
        # then include each point and its negation to ensure symmetry.
        # Use a half-fraction: first d-2 columns full factorial,
        # (d-1)-th column = product of first d-2 columns,
        # d-th column = product of first two columns (independent generator).
        n_base = d - 2
        n_half = 2^n_base
        points = Vector{Vector{Float64}}(undef, 2 * n_half)
        for i in 0:(n_half - 1)
            point = Vector{Float64}(undef, d)
            for j in 1:n_base
                point[j] = (i >> (j - 1)) & 1 == 0 ? -1.0 : 1.0
            end
            # Column d-1: product of first n_base columns
            point[d - 1] = prod(@view point[1:n_base])
            # Column d: product of first two columns
            point[d] = point[1] * point[2]
            points[2i + 1] = point
            points[2i + 2] = -point
        end
        return unique(points)
    end
end

"""
    generate_ccd_points(d::Int; f0::Float64 = 1.1) -> Vector{Vector{Float64}}

Generate Central Composite Design points in standardized z-space for `d` dimensions,
following R-INLA's convention (Rue et al. 2009, Section 6.5).

All non-center design points are placed on a sphere of radius `f0 * √d`:
1. Center point at origin (1 point)
2. Axial points at ±f₀√d along each axis (2d points)
3. Factorial points normalized to radius f₀√d (2^d or fractional for d > 4)

The scaling factor `f0` must be > 1 (default 1.1, matching R-INLA).
"""
function generate_ccd_points(d::Int; f0::Float64 = 1.1)
    @assert f0 > 1.0 "f0 must be > 1.0, got $f0"
    radius = f0 * sqrt(Float64(d))

    factorial_pts = generate_factorial_points(d)
    n_total = 1 + 2d + length(factorial_pts)
    points = Vector{Vector{Float64}}()
    sizehint!(points, n_total)

    # Center point
    push!(points, zeros(d))

    # Axial points: ±radius along each coordinate axis
    for i in 1:d
        e = zeros(d)
        e[i] = radius
        push!(points, e)
        push!(points, -e)
    end

    # Factorial points: ±1 in all dimensions, then scale to radius
    # Raw factorial points have norm √d, so scale by f0 to get radius f0√d
    # For d=1, factorial points may duplicate axial points, so deduplicate
    for p in factorial_pts
        scaled = p .* f0
        if scaled ∉ points
            push!(points, scaled)
        end
    end

    return points
end

"""
    explore_hyperparameter_posterior(::CCDExplorationStrategy, model, y, θ_star, ...)

CCD-based exploration of the hyperparameter posterior (Rue et al. 2009, Section 6.5).

Uses O(2d² + 1) design points instead of a full Cartesian grid (O(m^d)),
enabling models with 3+ hyperparameters. Integration weights are computed
analytically following R-INLA's convention, which is exact when the posterior
in z-space is standard Gaussian.
"""
function explore_hyperparameter_posterior(
        strategy::CCDExplorationStrategy,
        model::LatentGaussianModel, y, θ_star::WorkingHyperparameters,
        marginalization_method, marginalization_indices;
        progress_callback = nothing,
        accumulators::Tuple = (),
        executor::ParallelExecutor = SequentialExecutor(),
        diff_strategy::DifferentiationStrategy = ADStrategy()
    )
    f0 = strategy.f0
    if progress_callback === nothing
        progress_callback = (; kwargs...) -> nothing
    end

    d = length(θ_star)

    # Build a pool sized for the active executor. CCD's per-design-point
    # evaluation runs through the pool-aware pmap_executor so each threaded
    # task gets its own workspace without racing.
    θ_star_nt = convert(NamedTuple, convert(NaturalHyperparameters, θ_star))
    pool = make_workspace_pool(model.latent_prior; size = _pool_size(executor), θ_star_nt...)

    # Step 1: Compute reparameterization (same as grid approach)
    progress_callback(status = "Computing reparameterization", dimensions = d)
    transform = compute_reparameterization(model, y, θ_star; pool = pool, executor = executor, diff_strategy = diff_strategy)

    # Step 2: Generate CCD points in z-space with f0 scaling
    z_points = generate_ccd_points(d; f0 = f0)
    n_design = length(z_points)
    progress_callback(status = "CCD design", n_points = n_design, dimensions = d)

    # Step 3: Compute analytical CCD integration weights (Rue et al. 2009)
    w_sphere, w_center = ccd_integration_weights(n_design, d, f0)

    # Step 4: Evaluate all CCD points (PARALLEL with per-task workspaces)
    # Build work items with pre-computed θ values
    work_items = [(z = z, θ = transform(z), is_center = all(iszero, z)) for z in z_points]

    eval_results = pmap_executor(work_items, executor, pool) do item, ws
        result = evaluate_at_grid_point(
            model, y, item.θ;
            ws = ws,
            compute_marginals = true,
            marginalization_method = marginalization_method,
            marginalization_indices = marginalization_indices,
        )
        # Compute accumulator summaries while ga/obs_lik are still alive
        # Skip for rejected points (log_density == -Inf, ga/obs_lik are nothing)
        summaries = if result.log_density > -Inf
            map(accumulators) do acc
                compute_point_summary(acc; result...)
            end
        else
            map(_ -> nothing, accumulators)
        end
        return (; result..., θ = item.θ, z = item.z, is_center = item.is_center, summaries)
    end

    progress_callback(status = "CCD evaluation complete", n_evaluated = length(eval_results))

    # Step 5: Build grid points + accumulate (SEQUENTIAL)
    GP = GridPoint{typeof(θ_star)}
    grid_points = Vector{GP}()
    sizehint!(grid_points, n_design)

    mode_raw_logp = NaN
    axial_raw_logp_plus = fill(NaN, d)
    axial_raw_logp_minus = fill(NaN, d)

    for r in eval_results
        if r.log_density > -Inf
            # Capture raw log-density at mode and axial points for CCD interpolant
            if r.is_center
                mode_raw_logp = r.log_density
            elseif count(!iszero, r.z) == 1
                dim_idx = findfirst(!iszero, r.z)
                if r.z[dim_idx] > 0
                    axial_raw_logp_plus[dim_idx] = r.log_density
                else
                    axial_raw_logp_minus[dim_idx] = r.log_density
                end
            end

            # Store log(Δ_k * π̃(θ_k|y)) as the log_density for this point.
            Δ_k = r.is_center ? w_center : w_sphere
            weighted_log_density = log(Δ_k) + r.log_density
            push!(grid_points, GridPoint(r.θ, weighted_log_density, r.marginal_result))

            # Accumulate from pre-computed summaries
            if !isempty(accumulators)
                for (acc, summary) in zip(accumulators, r.summaries)
                    if summary !== nothing
                        accumulate!(acc, summary; is_mode = r.is_center)
                    end
                end
            end
        end
    end

    # Step 6: Normalize
    progress_callback(status = "Computing normalization", n_valid = length(grid_points))

    log_densities = [p.log_density for p in grid_points]
    log_normalization_constant = logsumexp(log_densities) + logdet_jacobian(transform)

    # Create normalized grid points
    normalized_points = [
        GridPoint(p.θ, p.log_density - log_normalization_constant, p.marginal_result)
            for p in grid_points
    ]

    # All valid points are integration points; no reordering needed
    integration_indices = collect(1:length(normalized_points))
    accumulator_reorder = collect(1:length(normalized_points))

    exploration = CCDExploration(
        normalized_points,
        integration_indices,
        transform,
        log_normalization_constant,
        mode_raw_logp,
        axial_raw_logp_plus,
        axial_raw_logp_minus,
        f0;
        accumulator_reorder = accumulator_reorder
    )

    # Step 7: Finalize accumulators
    if !isempty(accumulators)
        progress_callback(status = "Finalizing accumulators", n_accumulators = length(accumulators))
        for acc in accumulators
            finalize!(acc, exploration)
        end
    end

    progress_callback(status = "CCD exploration complete", final_points = length(normalized_points))

    return exploration, accumulators
end
