# `_pointwise_cdf(η, lik)` — per-observation CDF values `F(y_i | η_i)`.
# For continuous obs, the value of the cumulative distribution at `y_i`;
# for discrete obs, the midpoint PIT `F(y_i) - 0.5 · f(y_i)` (used by
# CPO's PIT diagnostic when the obs model exposes a CDF).
#
# Dispatch lives in this file so it can be extended for new likelihoods
# without touching the accumulator implementations.

using Distributions
using GaussianMarkovRandomFields:
    NormalLikelihood, PoissonLikelihood, BernoulliLikelihood,
    BinomialLikelihood, CompositeLikelihood, IdentityLink, apply_invlink

function _pointwise_cdf end

# Composite: each component slices into the same `η` vector via its
# `indices` field (set by `linear_predictor_marginals` on composites).
function _pointwise_cdf(η, lik::CompositeLikelihood)
    return reduce(vcat, _pointwise_cdf(η, comp) for comp in lik.components)
end

function _pointwise_cdf(η, obs_lik::NormalLikelihood{IdentityLink})
    y = obs_lik.y
    σ = obs_lik.σ
    indices = obs_lik.indices
    return [
        cdf(Normal(η[indices === nothing ? i : indices[i]], σ), y[i])
            for i in eachindex(y)
    ]
end

function _pointwise_cdf(η, obs_lik::NormalLikelihood)
    y = obs_lik.y
    σ = obs_lik.σ
    indices = obs_lik.indices
    return [
        cdf(Normal(apply_invlink(obs_lik.link, η[indices === nothing ? i : indices[i]]), σ), y[i])
            for i in eachindex(y)
    ]
end

function _pointwise_cdf(η, obs_lik::PoissonLikelihood)
    y = obs_lik.y
    indices = obs_lik.indices
    result = Vector{Float64}(undef, length(y))
    for i in eachindex(y)
        idx = indices === nothing ? i : indices[i]
        λ = apply_invlink(obs_lik.link, η[idx])
        if obs_lik.logexposure !== nothing
            λ *= exp(obs_lik.logexposure[i])
        end
        d = Poisson(max(λ, 1.0e-20))
        result[i] = cdf(d, y[i]) - 0.5 * pdf(d, y[i])
    end
    return result
end

function _pointwise_cdf(η, obs_lik::BernoulliLikelihood)
    y = obs_lik.y
    indices = obs_lik.indices
    result = Vector{Float64}(undef, length(y))
    for i in eachindex(y)
        idx = indices === nothing ? i : indices[i]
        p = apply_invlink(obs_lik.link, η[idx])
        d = Bernoulli(clamp(p, 1.0e-10, 1 - 1.0e-10))
        result[i] = cdf(d, y[i]) - 0.5 * pdf(d, y[i])
    end
    return result
end

function _pointwise_cdf(η, obs_lik::BinomialLikelihood)
    y = obs_lik.y
    indices = obs_lik.indices
    result = Vector{Float64}(undef, length(y))
    for i in eachindex(y)
        idx = indices === nothing ? i : indices[i]
        p = apply_invlink(obs_lik.link, η[idx])
        d = Binomial(obs_lik.n[i], clamp(p, 1.0e-10, 1 - 1.0e-10))
        result[i] = cdf(d, y[i]) - 0.5 * pdf(d, y[i])
    end
    return result
end
