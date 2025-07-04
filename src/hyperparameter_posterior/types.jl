"""
Type definitions for hyperparameter posterior exploration and approximation.
"""

using Distributions
using DataInterpolations
using ScatteredInterpolation

export HyperparameterExploration, HyperparameterPosteriorApproximation

"""
    HyperparameterExploration{T}

Results from exploring the hyperparameter posterior around the mode.

# Fields
- `mode::Vector{T}`: The posterior mode θ*
- `interpolation_points::Vector{Vector{T}}`: All evaluation points 
- `integration_indices::Vector{Int}`: Indices into interpolation_points for integration
- `log_densities::Vector{T}`: Log π(θ | y) evaluated at interpolation_points
- `transformation::NamedTuple`: Reparameterization info (V, Λ, H, mode_logpdf)
- `integration_bounds::Matrix{T}`: Integration bounds for marginalization (n_dims × 2)
"""
struct HyperparameterExploration{T}
    mode::Vector{T}
    interpolation_points::Vector{Vector{T}}
    integration_indices::Vector{Int}
    log_densities::Vector{T}
    transformation::NamedTuple
    integration_bounds::Matrix{T}
end

"""
    HyperparameterPosteriorApproximation{T, I}

Interpolated approximation to the hyperparameter posterior.

# Fields
- `exploration::HyperparameterExploration{T}`: The underlying exploration data
- `interpolant::I`: Interpolation object for log π(θ | y)
"""
struct HyperparameterPosteriorApproximation{T, I}
    exploration::HyperparameterExploration{T}
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

# Method extension for computing mode of Product distributions
"""
    mode(d::Product)

Compute the mode of a Product distribution by computing the mode of each marginal distribution.

The mode of a product of independent distributions is the vector of modes of the marginal distributions.
"""
function Distributions.mode(d::Product)
    return [mode(marginal) for marginal in d.v]
end
