export GridPoint, HyperparameterExploration, integration_bounds

"""
Represents a single point on the exploration grid with its computed results.
"""
struct GridPoint
    θ::Vector{Float64}
    log_density::Float64
    marginal_result::Union{Nothing, MarginalResult}
end

"""
Represents the complete result of the hyperparameter posterior exploration.
This is the primary object returned to the user.
"""
struct HyperparameterExploration
    grid_points::Vector{GridPoint}
    integration_indices::Vector{Int}
    transform::ReparameterizationTransform
    log_normalization_constant::Float64
end

"""
    integration_bounds(exploration::HyperparameterExploration)

Compute the integration bounds from the exploration results.
Returns a matrix where bounds[i, 1] is the minimum and bounds[i, 2] is the maximum
for the i-th hyperparameter dimension.

# Arguments
- `exploration::HyperparameterExploration`: The exploration results

# Returns
- `Matrix{Float64}`: (n_dims × 2) matrix with min/max bounds for each dimension
"""
function integration_bounds(exploration::HyperparameterExploration)
    # Get integration points only
    integration_points = exploration.grid_points[exploration.integration_indices]

    if isempty(integration_points)
        error("No integration points found in exploration")
    end

    # Extract parameter values
    θ_values = [point.θ for point in integration_points]
    n_dims = length(θ_values[1])

    # Compute bounds for each dimension
    bounds = Matrix{Float64}(undef, n_dims, 2)

    for dim in 1:n_dims
        dim_values = [θ[dim] for θ in θ_values]
        bounds[dim, 1] = minimum(dim_values)  # min
        bounds[dim, 2] = maximum(dim_values)  # max
    end

    return bounds
end
