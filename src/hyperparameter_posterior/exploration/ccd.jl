export explore_hyperparameter_posterior_ccd

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
    explore_hyperparameter_posterior_ccd(
        model::INLAModel, y, θ_star::WorkingHyperparameters,
        marginalization_method, marginalization_indices;
        f0=1.1, progress_callback=nothing, accumulators::Tuple=()
    )

CCD-based exploration of the hyperparameter posterior (Rue et al. 2009, Section 6.5).

Uses O(2d² + 1) design points instead of a full Cartesian grid (O(m^d)),
enabling models with 3+ hyperparameters. Integration weights are computed
analytically following R-INLA's convention, which is exact when the posterior
in z-space is standard Gaussian.

The `f0` parameter (default 1.1, matching R-INLA) controls how far design points
are placed from the mode: all non-center points lie on a sphere of radius `f0 * √d`.

Returns `(HyperparameterExploration, accumulators)`, the same output type as
`explore_hyperparameter_posterior`, so all downstream code works unchanged.
"""
function explore_hyperparameter_posterior_ccd(
        model::INLAModel, y, θ_star::WorkingHyperparameters,
        marginalization_method, marginalization_indices;
        f0::Float64 = 1.1,
        progress_callback = nothing,
        accumulators::Tuple = ()
    )
    if progress_callback === nothing
        progress_callback = (; kwargs...) -> nothing
    end

    d = length(θ_star)

    # Step 1: Compute reparameterization (same as grid approach)
    progress_callback(status = "Computing reparameterization", dimensions = d)
    transform = compute_reparameterization(model, y, θ_star)

    # Step 2: Generate CCD points in z-space with f0 scaling
    z_points = generate_ccd_points(d; f0 = f0)
    n_design = length(z_points)
    progress_callback(status = "CCD design", n_points = n_design, dimensions = d)

    # Step 3: Compute analytical CCD integration weights (Rue et al. 2009)
    w_sphere, w_center = ccd_integration_weights(n_design, d, f0)

    # Step 4: Evaluate all CCD points
    grid_points = Vector{GridPoint}()
    sizehint!(grid_points, n_design)

    for (point_idx, z) in enumerate(z_points)
        θ = transform(z)
        center = all(iszero, z)

        progress_callback(
            status = "Evaluating CCD points",
            points_evaluated = point_idx,
            total_points = n_design,
            current_θ = θ
        )

        result = evaluate_at_grid_point(
            model, y, θ;
            compute_marginals = true,
            marginalization_method = marginalization_method,
            marginalization_indices = marginalization_indices
        )

        if result.log_density > -Inf
            # Store log(Δ_k * π̃(θ_k|y)) as the log_density for this point.
            # The quadrature weight Δ_k differs for center vs sphere points.
            Δ_k = center ? w_center : w_sphere
            weighted_log_density = log(Δ_k) + result.log_density
            push!(grid_points, GridPoint(θ, weighted_log_density, result.marginal_result))

            # Call accumulators eagerly
            if !isempty(accumulators)
                for acc in accumulators
                    accumulate!(
                        acc;
                        result...,
                        θ = θ,
                        y = y,
                        is_mode = center
                    )
                end
            end
        end
    end

    # Step 5: Normalize
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

    exploration = HyperparameterExploration(
        normalized_points,
        integration_indices,
        transform,
        log_normalization_constant;
        accumulator_reorder = accumulator_reorder
    )

    # Step 6: Finalize accumulators
    if !isempty(accumulators)
        progress_callback(status = "Finalizing accumulators", n_accumulators = length(accumulators))
        for acc in accumulators
            finalize!(acc, exploration)
        end
    end

    progress_callback(status = "CCD exploration complete", final_points = length(normalized_points))

    return exploration, accumulators
end
