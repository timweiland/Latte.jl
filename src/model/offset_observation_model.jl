# Observation-level offset wrapper.
#
# Adds a per-site constant `offset[i]` to the linear predictor before
# materialising the wrapped likelihood. Lives at the obs-model layer so
# it works uniformly across augmented and non-augmented LGMs: the
# gradient / Hessian of the likelihood in `x` is translation-invariant,
# so only the primal loglik needs to see the shifted predictor.
#
# Intended for use by the DPPL adapter's fast-path when the detected
# linear predictor has a non-zero constant term (e.g. Poisson log-
# exposure, Bernoulli logit-shift, Normal mean offset).

using GaussianMarkovRandomFields:
    ObservationModel, ObservationLikelihood,
    loglik, loggrad, loghessian, hyperparameters, latent_dimension,
    pointwise_loglik, conditional_distribution

import GaussianMarkovRandomFields:
    loglik, loggrad, loghessian, hyperparameters, latent_dimension,
    pointwise_loglik, conditional_distribution

export OffsetObservationModel

"""
    OffsetObservationModel(base::ObservationModel, offset::AbstractVector)

Wraps `base` to add a constant per-site offset to the linear predictor
before evaluating the likelihood:

    log p(y | x) := log p(y | x + offset, base)

The offset must match `base`'s expected number of observation sites.
Gradient and Hessian in `x` are unchanged (translation invariance), so
downstream Laplace / Newton machinery sees exactly the same curvature
as the unwrapped base likelihood.

Typical construction is via the DPPL adapter's fast path; manual use is
fine too:

```julia
base = ExponentialFamily(Poisson, LogLink())
obs  = OffsetObservationModel(base, log.(exposure_vec))
lgm  = LatentGaussianModel(spec, FunctionLatentModel(...), obs)
```
"""
struct OffsetObservationModel{M <: ObservationModel, O <: AbstractVector} <: ObservationModel
    base::M
    offset::O
end

"""
    OffsetObservationLikelihood(base_lik::ObservationLikelihood, offset::AbstractVector)

Materialised form of `OffsetObservationModel`. Shifts `x` by `offset`
and delegates to the wrapped `base_lik` for all likelihood calls.
"""
struct OffsetObservationLikelihood{L <: ObservationLikelihood, O <: AbstractVector} <:
    ObservationLikelihood
    base_lik::L
    offset::O
end

# ─── Model → likelihood materialisation ─────────────────────────────────
(m::OffsetObservationModel)(y; kwargs...) =
    OffsetObservationLikelihood(m.base(y; kwargs...), m.offset)

# ─── Hyperparameter introspection ───────────────────────────────────────
hyperparameters(m::OffsetObservationModel) = hyperparameters(m.base)
latent_dimension(m::OffsetObservationModel) = latent_dimension(m.base)

# ─── Likelihood interface: shift x on the base-lik's indexed subset ────
# The base likelihood reads only specific components of `x` (e.g.
# `lik.indices` on exponential-family likelihoods) — typically the
# linear-predictor (η) components in an augmented LGM. We add offset to
# those components only and pass the modified `x` through; the base lik
# handles the rest of its usual indexing unchanged.
function _shift_by_offset(x, l::OffsetObservationLikelihood)
    base = l.base_lik
    if hasproperty(base, :indices) && getproperty(base, :indices) !== nothing
        idx = base.indices
        # Preserve element type so Duals propagate through outer AD.
        T = promote_type(eltype(x), eltype(l.offset))
        out = Vector{T}(x)
        @views out[idx] .+= l.offset
        return out
    else
        # base reads all of x — broadcast, relying on matching dimensions
        return x .+ l.offset
    end
end

loglik(x, l::OffsetObservationLikelihood) = loglik(_shift_by_offset(x, l), l.base_lik)
loggrad(x, l::OffsetObservationLikelihood) = loggrad(_shift_by_offset(x, l), l.base_lik)
loghessian(x, l::OffsetObservationLikelihood) = loghessian(_shift_by_offset(x, l), l.base_lik)
pointwise_loglik(x, l::OffsetObservationLikelihood) = pointwise_loglik(_shift_by_offset(x, l), l.base_lik)

# CPO accumulator: Latte-side dispatch on materialised likelihoods
# (defined in src/posterior/accumulators/cpo.jl). Delegate the same way.
function _pointwise_cdf(x, l::OffsetObservationLikelihood)
    return _pointwise_cdf(_shift_by_offset(x, l), l.base_lik)
end

# ─── Conditional distribution (for posterior-predictive y sampling) ────
# Handed `x` of whatever shape the base obs model expects. We don't have
# an indices field on the model itself (only on the materialised lik),
# so we only cover the "x reads all of x" shape here; augmented sampling
# goes through the LTM / η slice anyway.
conditional_distribution(m::OffsetObservationModel, x; kwargs...) =
    conditional_distribution(m.base, x .+ m.offset; kwargs...)

# ─── Plumbing for the prediction-via-missing subset path ───────────────
# `_restrict_obs_model_to_indices` is dispatched by the LGM constructor
# when it augments and by `_prepare_for_prediction` when `y` has missings.
# We restrict the wrapped base AND the offset vector together.
function _restrict_obs_model_to_indices(m::OffsetObservationModel, indices)
    return OffsetObservationModel(
        _restrict_obs_model_to_indices(m.base, indices),
        m.offset[indices],
    )
end

# `_normalize_observations(y, obs_model)` normalises raw user-y (e.g.
# Vector{Int} → PoissonObservations). Delegate through the wrapper.
function _normalize_observations(y::AbstractVector, m::OffsetObservationModel)
    return _normalize_observations(y, m.base)
end
