using DataInterpolations
using ScatteredInterpolation
using Printf

export build_posterior_interpolant, HyperparameterPosteriorApproximation

"""
    HyperparameterPosteriorApproximation{I}

Interpolated approximation to the hyperparameter posterior.

# Fields
- `exploration::AbstractHyperparameterExploration`: The underlying exploration data
- `interpolant::I`: Interpolation object for log π(θ | y)
"""
struct HyperparameterPosteriorApproximation{I}
    exploration::AbstractHyperparameterExploration
    interpolant::I
end

"""
    (approx::HyperparameterPosteriorApproximation)(θ)

Evaluate the hyperparameter posterior approximation at point θ.

# Arguments
- `θ`: Can be `WorkingHyperparameters`, `NaturalHyperparameters`, or a vector (assumed to be in natural space)

# Details
The interpolant is built in working space. If θ is in natural space (NaturalHyperparameters or plain vector),
it is converted to working space before evaluation. The returned log density accounts for the Jacobian
when converting from natural to working space.
"""
function (approx::HyperparameterPosteriorApproximation)(θ::WorkingHyperparameters)
    # Already in working space, evaluate directly
    if isa(approx.interpolant, Union{CubicSpline, LinearInterpolation, ConstantInterpolation, QuadraticInterpolation})
        # DataInterpolations.jl interface: interpolant(x)
        return approx.interpolant(θ.θ[1])
    else
        # ScatteredInterpolation.jl interface: evaluate(interpolant, point)
        result = evaluate(approx.interpolant, θ.θ)
        return result[1]  # ScatteredInterpolation returns a vector
    end
end

function (approx::HyperparameterPosteriorApproximation)(θ::NaturalHyperparameters)
    # Convert to working space and evaluate
    θ_working = convert(WorkingHyperparameters, θ)
    log_p_working = approx(θ_working)
    # Add Jacobian correction to get natural-space density
    return log_p_working + logdetjac(θ)
end

function (approx::HyperparameterPosteriorApproximation)(θ_vec::AbstractVector)
    # Assume vector is in natural space - need spec to convert
    spec = approx.exploration.transform.θ_star.spec
    θ_natural = NaturalHyperparameters(θ_vec, spec)
    return approx(θ_natural)
end

"""
    build_posterior_interpolant(exploration::AbstractHyperparameterExploration; progress_callback=nothing)

Build an interpolant for the hyperparameter posterior log-density.

# Arguments
- `exploration`: Results from explore_hyperparameter_posterior
- `progress_callback`: Optional function for progress updates with signature `f(; kwargs...)`

# Returns  
- `HyperparameterPosteriorApproximation`: Interpolated posterior approximation
"""
function build_posterior_interpolant(exploration::AbstractHyperparameterExploration; progress_callback = nothing)
    n_dim = length(exploration.transform.θ_star)
    n_points = length(exploration.grid_points)

    # Handle progress callback
    if progress_callback === nothing
        progress_callback = (; kwargs...) -> nothing
    end

    if n_dim == 1
        # For 1D case, use spline interpolation
        progress_callback(status = "Building 1D spline interpolant", dimensions = n_dim, points = n_points)
        θ_values = [p.θ[1] for p in exploration.grid_points]
        log_densities = [p.log_density for p in exploration.grid_points]
        perm = sortperm(θ_values)
        sorted_θ = θ_values[perm]
        sorted_logpdf = log_densities[perm]

        # Use appropriate interpolation based on number of points
        if length(sorted_θ) == 1
            interpolant = ConstantInterpolation(sorted_logpdf, sorted_θ)
            progress_callback(status = "Constant interpolant complete", method = "ConstantInterpolation")
        elseif length(sorted_θ) == 2
            interpolant = LinearInterpolation(sorted_logpdf, sorted_θ)
            progress_callback(status = "Linear interpolant complete", method = "LinearInterpolation")
        else
            interpolant = QuadraticInterpolation(sorted_logpdf, sorted_θ; extrapolation = ExtrapolationType.Linear)
            progress_callback(status = "Spline interpolant complete", method = "CubicSpline")
        end
    else
        # Multidimensional case - use thin-plate spline interpolation
        progress_callback(status = "Building RBF interpolant", dimensions = n_dim, points = n_points)
        # Convert points to matrix format (n_dim × n_points)
        points_matrix = reduce(hcat, [p.θ for p in exploration.grid_points])
        log_densities = [p.log_density for p in exploration.grid_points]

        # Use thin-plate splines (best performance in practice)
        rbf = ThinPlate()
        interpolant = interpolate(rbf, points_matrix, log_densities)
        progress_callback(status = "RBF interpolant complete", method = "ThinPlate")
    end

    progress_callback(status = "Interpolation complete", final_interpolant = typeof(interpolant))
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
