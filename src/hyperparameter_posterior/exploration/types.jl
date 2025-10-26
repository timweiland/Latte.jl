using Printf

export GridPoint, HyperparameterExploration

"""
Represents a single point on the exploration grid with its computed results.

# Type Parameters
- `W <: WorkingHyperparameters`: Type of working hyperparameters

# Fields
- `θ::W`: Hyperparameters in working space
- `log_density::Float64`: Log density at this point
- `marginal_result::Union{Nothing, MarginalResult}`: Optional marginal results
"""
struct GridPoint{W <: WorkingHyperparameters}
    θ::W
    log_density::Float64
    marginal_result::Union{Nothing, MarginalResult}
end

"""
Represents the complete result of the hyperparameter posterior exploration.
This is the primary object returned to the user.

# Type Parameters
- `GP <: GridPoint`: Type of grid points
"""
struct HyperparameterExploration{GP <: GridPoint}
    grid_points::Vector{GP}
    integration_indices::Vector{Int}
    transform::ReparameterizationTransform
    log_normalization_constant::Float64
    integration_bounds::Matrix{Float64}

    # Constructor that computes integration bounds
    function HyperparameterExploration(grid_points::Vector{GP}, integration_indices, transform, log_normalization_constant) where {GP <: GridPoint}
        # Compute integration bounds once during construction
        integration_points = grid_points[integration_indices]

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

        return new{GP}(grid_points, integration_indices, transform, log_normalization_constant, bounds)
    end
end

# Custom show methods for better user experience

function Base.show(io::IO, gp::GridPoint)
    print(io, "GridPoint(θ=")
    θ_vec = gp.θ.θ  # Extract underlying vector from WorkingHyperparameters
    if length(θ_vec) <= 3
        print(io, "[", join([@sprintf("%.4f", x) for x in θ_vec], ", "), "]")
    else
        print(io, "[", @sprintf("%.4f", θ_vec[1]), ", ", @sprintf("%.4f", θ_vec[2]), ", ..., ", @sprintf("%.4f", θ_vec[end]), "]")
    end
    print(io, ", log_density=", @sprintf("%.4f", gp.log_density))
    if gp.marginal_result !== nothing
        print(io, ", marginals=", length(gp.marginal_result.marginals), " variables")
    else
        print(io, ", marginals=none")
    end
    return print(io, ")")
end

function Base.show(io::IO, exploration::HyperparameterExploration)
    println(io, "HyperparameterExploration:")
    println(io, "  Grid points: ", length(exploration.grid_points))
    println(io, "  Integration points: ", length(exploration.integration_indices))

    # Get parameter dimension
    if !isempty(exploration.grid_points)
        n_dim = length(exploration.grid_points[1].θ)
        println(io, "  Parameter dimensions: ", n_dim)

        # Show parameter ranges for integration points
        if !isempty(exploration.integration_indices)
            integration_points = exploration.grid_points[exploration.integration_indices]

            for dim in 1:n_dim
                dim_values = [point.θ[dim] for point in integration_points]
                min_val = minimum(dim_values)
                max_val = maximum(dim_values)
                println(io, "    Dimension ", dim, ": [", @sprintf("%.4f", min_val), ", ", @sprintf("%.4f", max_val), "]")
            end
        end
    end

    # Show log-density range
    log_densities = [point.log_density for point in exploration.grid_points]
    if !isempty(log_densities)
        println(io, "  Log density range: [", @sprintf("%.4f", minimum(log_densities)), ", ", @sprintf("%.4f", maximum(log_densities)), "]")
    end

    return print(io, "  Log normalization constant: ", @sprintf("%.4f", exploration.log_normalization_constant))
end
