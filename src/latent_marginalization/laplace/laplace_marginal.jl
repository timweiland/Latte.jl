using GaussianMarkovRandomFields

export LaplaceMarginal

"""
    LaplaceMarginal <: MarginalApproximation

Laplace marginalization: uses spline-corrected Gaussian approximation.
Computes π̃_LA(x_i | θ, y) ≈ π̃_G(x_i | θ, y) * exp(spline(x_i)).

# Fields
- `normalize_exactly::Bool`: If true, use numerical integration for exact normalization; if false, use Gauss-Hermite approximation (faster)
"""
struct LaplaceMarginal <: MarginalApproximation
    normalize_exactly::Bool
end

# Convenience constructor with default
LaplaceMarginal() = LaplaceMarginal(false)

"""
    _marginalize_impl(ga, obs_lik, log_prior_θ, method::LaplaceMarginal, indices, prior_gmrf)

Implementation for Laplace marginalization using spline correction.
"""
function _marginalize_impl(
        ga, obs_lik, log_prior_θ::Real,
        method::LaplaceMarginal, indices::Vector{Int}, prior_gmrf
    )
    # Validate that prior_gmrf is provided for Laplace marginalization
    if isnothing(prior_gmrf)
        throw(ArgumentError("prior_gmrf is required for Laplace marginalization"))
    end

    # Compute μ and σ once outside the loop
    # Note: GMRF's std() returns SparseVector, convert to Vector for cache compatibility
    μ = mean(ga)
    σ = Vector(std(ga))

    # Reuse the GMRF's existing LinearSolve cache (contains precomputed factorization)
    lsc = GaussianMarkovRandomFields.linsolve_cache(ga)

    marginals = SplineAugmentedGaussian{Float64}[]

    for i in indices
        # Create cache for this variable, reusing the GMRF's precomputed factorization
        cache = LaplaceApproximationCache(ga, obs_lik, i, μ, σ, prior_gmrf, lsc)

        # Fit spline correction using the normalization method specified in the LaplaceMarginal
        spline, log_norm_const, nodes, corrections = fit_density_correction_spline(
            cache, log_prior_θ; normalize_exactly = method.normalize_exactly
        )

        # Create base Gaussian using precomputed values
        base_gaussian = Normal(μ[i], σ[i])

        # Create spline-augmented distribution
        marginal = SplineAugmentedGaussian(base_gaussian, spline, log_norm_const)
        push!(marginals, marginal)
    end

    return marginals
end
