using LinearAlgebra
using FiniteDiff

export explore_hyperparameter_posterior

"""
    compute_reparameterization(model::INLAModel, y, θ_star)

Compute the reparameterization θ(z) = θ* + V√Λ z around the mode.

Returns (H, V, Λ, mode_logpdf) where θ(z) = θ_star + V * sqrt.(abs.(Λ)) .* z
"""
function compute_reparameterization(model::INLAModel, y, θ_star)
    mode_logpdf = hyperparameter_logpdf(model, θ_star, y)

    # Compute Hessian using FiniteDiff.jl
    H = FiniteDiff.finite_difference_hessian(θ -> hyperparameter_logpdf(model, θ, y), θ_star)
    H = -H  # Make negative definite (negative Hessian of log-density)

    # Eigendecomposition for reparameterization
    eigen_result = eigen(H)
    Λ = eigen_result.values
    V = eigen_result.vectors

    Λ_inv = 1 ./ Λ
    Λ_inv_sqrt = Diagonal(sqrt.(Λ_inv))

    return H, V, Λ_inv_sqrt, mode_logpdf
end

"""
    explore_direction(model::INLAModel, y, θ_star, V, Λ_inv_sqrt, mode_logpdf, dim, δ_π, step_size, direction)

Explore one direction along a dimension until log-density drops by δ_π.

Returns (θ_points, logpdf_values) for this direction (not including mode).
"""
function explore_direction(model::INLAModel, y, θ_star, V, Λ_inv_sqrt, mode_logpdf, dim::Int, δ_π, step_size, direction::Int)
    n_dim = length(θ_star)
    θ_points = Vector{Vector{Float64}}()
    logpdf_values = Float64[]

    z = direction * step_size
    while true
        z_vec = zeros(n_dim)
        z_vec[dim] = z
        θ_test = θ_star + V * Λ_inv_sqrt * z_vec

        logpdf_test = hyperparameter_logpdf(model, θ_test, y)
        if mode_logpdf - logpdf_test > δ_π
            break
        end

        push!(θ_points, θ_test)
        push!(logpdf_values, logpdf_test)
        z += direction * step_size
    end

    return θ_points, logpdf_values
end

"""
    explore_dimension(model::INLAModel, y, θ_star, V, Λ_inv_sqrt, mode_logpdf, dim, δ_π, step_size, interpolation_factor)

Explore one dimension until log-density drops by δ_π.

Returns (θ_points, logpdf_values, is_integration_point) for this dimension.
"""
function explore_dimension(model::INLAModel, y, θ_star, V, Λ_inv_sqrt, mode_logpdf, dim::Int, δ_π, step_size, interpolation_factor::Int)
    # Explore both directions
    pos_θ, pos_logpdf = explore_direction(model, y, θ_star, V, Λ_inv_sqrt, mode_logpdf, dim, δ_π, step_size, 1)
    neg_θ, neg_logpdf = explore_direction(model, y, θ_star, V, Λ_inv_sqrt, mode_logpdf, dim, δ_π, step_size, -1)

    # Combine with mode in the middle
    θ_points = vcat(reverse(neg_θ), [θ_star], pos_θ)
    logpdf_values = vcat(reverse(neg_logpdf), [nothing], pos_logpdf)

    # Create integration markers
    n_neg = length(neg_θ)
    n_total = length(θ_points)
    mode_index = n_neg + 1  # Index of the mode (middle point)

    is_integration_point = Vector{Bool}(undef, n_total)

    for i in 1:n_total
        # Distance from mode (step count)
        distance_from_mode = abs(i - mode_index)

        # Integration point if at mode or every interpolation_factor-th step away
        is_integration_point[i] = (distance_from_mode % interpolation_factor == 0)
    end

    return θ_points, logpdf_values, is_integration_point
end

"""
    build_multidimensional_grid(model, y, θ_points_per_dim, logpdf_per_dim, integration_markers_per_dim, n_dim, mode_logpdf, δ_π)

Build the multidimensional grid from per-dimension exploration results.

Returns (interpolation_θs, interpolation_logpdfs, integration_indices).
"""
function build_multidimensional_grid(model, y, θ_points_per_dim, logpdf_per_dim, integration_markers_per_dim, n_dim, mode_logpdf, δ_π)
    multi_θ_points = Iterators.product(θ_points_per_dim...)
    multi_logpdfs = Iterators.product(logpdf_per_dim...)
    multi_integration_markers = Iterators.product(integration_markers_per_dim...)

    interpolation_θs = Vector{Vector{Float64}}()
    interpolation_logpdfs = Vector{Float64}()
    integration_indices = Vector{Int}()

    for (θ_point, logpdf_vals, integration_tuple) in zip(multi_θ_points, multi_logpdfs, multi_integration_markers)
        θ_point = [θ_point[i][i] for i in eachindex(θ_point)]
        is_integration = all(integration_tuple)

        logpdf_val = Inf
        num_on_zero = 0
        for cur_val in logpdf_vals
            if cur_val === nothing
                num_on_zero += 1
            else
                logpdf_val = cur_val
            end
        end
        if num_on_zero !== (n_dim - 1)
            # This point is NOT on an axis, so we need to compute its logpdf
            logpdf_val = hyperparameter_logpdf(model, θ_point, y)
            if mode_logpdf - logpdf_val > δ_π
                # Don't add this point
                continue
            end
        elseif num_on_zero == n_dim
            is_integration = true
            logpdf_val = mode_logpdf
        end

        push!(interpolation_θs, θ_point)
        push!(interpolation_logpdfs, logpdf_val)

        if is_integration
            push!(integration_indices, length(interpolation_θs))  # Current index
        end
    end

    return interpolation_θs, interpolation_logpdfs, integration_indices
end

"""
    explore_hyperparameter_posterior(model::INLAModel, y, θ_star, mode_points, mode_logdensities;
                                    δ_π=2.5, interpolation_fraction=3.0)

Explore the hyperparameter posterior around the mode to create integration and interpolation grids.

# Arguments
- `model`: INLA model specification  
- `y`: Observed data
- `θ_star`: Posterior mode
- `mode_points`: Points from mode-finding (can be nothing)
- `mode_logdensities`: Log-densities from mode-finding (can be nothing)
- `δ_π`: Step size for integration grid (log-density tolerance)
- `interpolation_fraction`: Makes interpolation grid denser (> 1)

# Returns
- `HyperparameterExploration`: Complete exploration results
"""
function explore_hyperparameter_posterior(
        model::INLAModel, y, θ_star, mode_points, mode_logdensities;
        δ_d = 1.0, δ_π = 2.5, interpolation_factor::Int = 2
    )

    n_dim = length(θ_star)

    # Step 1: Compute reparameterization
    H, V, Λ_inv_sqrt, mode_logpdf = compute_reparameterization(model, y, θ_star)

    # Step 2: Compute step size once
    step_size = δ_d / interpolation_factor

    # Step 3: Explore each dimension
    θ_points_per_dim = Vector{Vector{Vector{Float64}}}()
    logpdf_per_dim = Vector{Vector{Union{Nothing, Float64}}}()
    integration_markers_per_dim = Vector{Vector{Bool}}()

    for d in 1:n_dim
        θ_points, logpdf_values, is_integration = explore_dimension(model, y, θ_star, V, Λ_inv_sqrt, mode_logpdf, d, δ_π, step_size, interpolation_factor)
        push!(θ_points_per_dim, θ_points)
        push!(logpdf_per_dim, logpdf_values)
        push!(integration_markers_per_dim, is_integration)
    end

    # Step 4: Build up multi-dimensional grid
    interpolation_θs, interpolation_logpdfs, integration_indices = build_multidimensional_grid(
        model, y, θ_points_per_dim, logpdf_per_dim, integration_markers_per_dim, n_dim, mode_logpdf, δ_π
    )

    # Step 5: Mode points already included in dimensional exploration grid

    # Step 6: Normalize the log densities using integration points
    # Compute step size in θ space for each dimension using δ_d (integration step size)
    θ_step_sizes = Float64[]
    for dim in 1:n_dim
        z_vec = zeros(n_dim)
        z_vec[dim] = δ_d
        θ_step_size = norm(V * Λ_inv_sqrt * z_vec)
        push!(θ_step_sizes, θ_step_size)
    end

    # Area weight is product of step sizes across all dimensions
    area_weight = prod(θ_step_sizes)

    # Extract integration points and their log densities
    integration_logpdfs = [interpolation_logpdfs[i] for i in integration_indices]

    # Compute normalization constant using logsumexp for numerical stability
    function logsumexp(x)
        max_x = maximum(x)
        return max_x + log(sum(exp.(x .- max_x)))
    end

    log_normalization = logsumexp(integration_logpdfs) + log(area_weight)

    # Normalize all log densities
    normalized_logpdfs = interpolation_logpdfs .- log_normalization

    # Store transformation info
    transformation = (V = V, Λ_inv_sqrt = Λ_inv_sqrt, H = H, mode_logpdf = mode_logpdf, log_normalization = log_normalization)

    # Compute integration bounds for marginalization
    n_dims = length(θ_star)
    integration_bounds = Matrix{Float64}(undef, n_dims, 2)

    for dim in 1:n_dims
        dim_values = [θ[dim] for θ in interpolation_θs]
        integration_bounds[dim, 1] = minimum(dim_values)  # Lower bound
        integration_bounds[dim, 2] = maximum(dim_values)  # Upper bound
    end

    return HyperparameterExploration(
        θ_star, interpolation_θs, integration_indices,
        normalized_logpdfs, transformation, integration_bounds
    )
end
