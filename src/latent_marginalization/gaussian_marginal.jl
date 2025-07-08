using Distributions
using GaussianMarkovRandomFields

export GaussianMarginal

"""
    GaussianMarginal <: MarginalApproximation

Gaussian marginalization: directly marginalize the Gaussian approximation π̃_G.
This is the fastest method but ignores non-Gaussian structure.
"""
struct GaussianMarginal <: MarginalApproximation end

"""
    _marginalize_impl(ga, obs_model, θ, y, log_prior_θ, ::GaussianMarginal, indices, prior_gmrf)

Implementation for Gaussian marginalization.
"""
function _marginalize_impl(
        ga, obs_model, θ, y, log_prior_θ::Real,
        ::GaussianMarginal, indices::AbstractVector{<:Integer}, prior_gmrf
    )
    # prior_gmrf is ignored for Gaussian marginalization
    μ = mean(ga)
    σ = std(ga)  # Marginal variances

    marginals = Normal{Float64}[]
    for i in indices
        push!(marginals, Normal(μ[i], σ[i]))
    end

    return marginals
end
