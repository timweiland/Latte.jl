using DataInterpolations
using ScatteredInterpolation
using Printf

export build_posterior_interpolant, HyperparameterPosteriorApproximation

"""
    HyperparameterPosteriorApproximation{I}

Interpolated approximation to the hyperparameter posterior.

# Fields
- `exploration::HyperparameterExploration`: The underlying exploration data
- `interpolant::I`: Interpolation object for log π(θ | y)
"""
struct HyperparameterPosteriorApproximation{I}
    exploration::HyperparameterExploration
    interpolant::I
end

"""
    (approx::HyperparameterPosteriorApproximation)(θ)

Evaluate the hyperparameter posterior approximation at point θ.
Handles both DataInterpolations.jl (CubicSpline) and ScatteredInterpolation.jl (RBF) interpolants.
"""
function (approx::HyperparameterPosteriorApproximation)(θ)
    if isa(approx.interpolant, CubicSpline)
        # DataInterpolations.jl interface: interpolant(x)
        return approx.interpolant(θ isa Vector ? θ[1] : θ)
    else
        # ScatteredInterpolation.jl interface: evaluate(interpolant, point)
        θ_vec = θ isa Vector ? θ : [θ]
        result = evaluate(approx.interpolant, θ_vec)
        return result[1]  # ScatteredInterpolation returns a vector
    end
end

"""
    build_posterior_interpolant(exploration::HyperparameterExploration)

Build an interpolant for the hyperparameter posterior log-density.

# Arguments
- `exploration`: Results from explore_hyperparameter_posterior

# Returns  
- `HyperparameterPosteriorApproximation`: Interpolated posterior approximation
"""
function build_posterior_interpolant(exploration::HyperparameterExploration)
    n_dim = length(exploration.transform.θ_star)

    if n_dim == 1
        # For 1D case, use spline interpolation
        θ_values = [p.θ[1] for p in exploration.grid_points]
        log_densities = [p.log_density for p in exploration.grid_points]
        perm = sortperm(θ_values)
        sorted_θ = θ_values[perm]
        sorted_logpdf = log_densities[perm]

        interpolant = CubicSpline(sorted_logpdf, sorted_θ)
    else
        # Multidimensional case - use thin-plate spline interpolation
        # Convert points to matrix format (n_dim × n_points)
        points_matrix = reduce(hcat, [p.θ for p in exploration.grid_points])
        log_densities = [p.log_density for p in exploration.grid_points]

        # Use thin-plate splines (best performance in practice)
        rbf = ThinPlate()
        interpolant = interpolate(rbf, points_matrix, log_densities)
    end

    return HyperparameterPosteriorApproximation(exploration, interpolant)
end

# Custom show method for better user experience
function Base.show(io::IO, approx::HyperparameterPosteriorApproximation)
    n_dim = length(approx.exploration.grid_points[1].θ)
    n_points = length(approx.exploration.grid_points)
    n_integration = length(approx.exploration.integration_indices)

    println(io, "HyperparameterPosteriorApproximation:")
    println(io, "  Interpolation method: ", typeof(approx.interpolant).name.name)
    println(io, "  Parameter dimensions: ", n_dim)
    println(io, "  Interpolation points: ", n_points)
    println(io, "  Integration points: ", n_integration)

    # Show parameter bounds
    return if !isempty(approx.exploration.integration_indices)
        integration_points = approx.exploration.grid_points[approx.exploration.integration_indices]
        θ_values = [point.θ for point in integration_points]

        println(io, "  Parameter bounds:")
        for dim in 1:n_dim
            dim_values = [θ[dim] for θ in θ_values]
            min_val = minimum(dim_values)
            max_val = maximum(dim_values)
            println(io, "    Dimension ", dim, ": [", @sprintf("%.4f", min_val), ", ", @sprintf("%.4f", max_val), "]")
        end
    end
end
