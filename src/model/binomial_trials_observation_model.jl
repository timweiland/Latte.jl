# Binomial per-site trial-count wrapper.
#
# `ExponentialFamily(Binomial, LogitLink)` needs `BinomialObservations`
# (successes + trials) at materialisation time; a raw `Vector{Int}` of
# successes is insufficient. This wrapper sits between the user's plain
# success-count vector and the ExpFam obs model, carrying the per-site
# trial counts (detected from the DPPL model at adapter time) so that
# `_normalize_observations` can assemble the right `BinomialObservations`.

using GaussianMarkovRandomFields:
    ObservationModel, BinomialObservations,
    loglik, loggrad, loghessian, hyperparameters, latent_dimension,
    pointwise_loglik, conditional_distribution

import GaussianMarkovRandomFields:
    loglik, loggrad, loghessian, hyperparameters, latent_dimension,
    pointwise_loglik, conditional_distribution

"""
    BinomialTrialsObservationModel(base::ObservationModel, trials::AbstractVector{<:Integer})

Observation-model wrapper attaching per-site Binomial trial counts to
`base` (typically `ExponentialFamily(Binomial, LogitLink)`). The only
difference from the base is how raw `y` is normalised: instead of the
base's default (pass-through), the wrapper produces a
`BinomialObservations(y, trials)`. Everything else — call, likelihood
methods, conditional distribution — delegates through.
"""
struct BinomialTrialsObservationModel{
        M <: ObservationModel, T <: AbstractVector{<:Integer},
    } <: ObservationModel
    base::M
    trials::T
end

# ─── Model → likelihood materialisation ─────────────────────────────────
(m::BinomialTrialsObservationModel)(y; kwargs...) = m.base(y; kwargs...)

hyperparameters(m::BinomialTrialsObservationModel) = hyperparameters(m.base)
latent_dimension(m::BinomialTrialsObservationModel) = latent_dimension(m.base)

conditional_distribution(m::BinomialTrialsObservationModel, x; kwargs...) =
    conditional_distribution(m.base, x; kwargs...)

# ─── Prediction-via-missing: restrict base + trials together ────────────
function _restrict_obs_model_to_indices(m::BinomialTrialsObservationModel, indices)
    return BinomialTrialsObservationModel(
        _restrict_obs_model_to_indices(m.base, indices),
        m.trials[indices],
    )
end

# ─── The whole point of the wrapper: assemble BinomialObservations ──────
function _normalize_observations(y::AbstractVector{<:Integer}, m::BinomialTrialsObservationModel)
    return BinomialObservations(collect(Int, y), m.trials)
end

# If user already hands in a BinomialObservations, just pass it through.
function _normalize_observations(y::BinomialObservations, ::BinomialTrialsObservationModel)
    return y
end
