using DataInterpolations
using ScatteredInterpolation

export build_posterior_interpolant

"""
    build_posterior_interpolant(exploration::HyperparameterExploration)

Build an interpolant for the hyperparameter posterior log-density.

# Arguments
- `exploration`: Results from explore_hyperparameter_posterior

# Returns  
- `HyperparameterPosteriorApproximation`: Interpolated posterior approximation
"""
function build_posterior_interpolant(exploration::HyperparameterExploration)

    n_dim = length(exploration.mode)

    if n_dim == 1
        # For 1D case, use spline interpolation
        θ_values = [p[1] for p in exploration.interpolation_points]
        perm = sortperm(θ_values)
        sorted_θ = θ_values[perm]
        sorted_logpdf = exploration.log_densities[perm]

        interpolant = CubicSpline(sorted_logpdf, sorted_θ)
    else
        # Multidimensional case - use thin-plate spline interpolation
        # Convert points to matrix format (n_dim × n_points)
        points_matrix = reduce(hcat, exploration.interpolation_points)

        # Use thin-plate splines (best performance in practice)
        rbf = ThinPlate()
        interpolant = interpolate(rbf, points_matrix, exploration.log_densities)
    end

    return HyperparameterPosteriorApproximation(exploration, interpolant)
end
