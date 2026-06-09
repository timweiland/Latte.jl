# Shared machinery for the WAIC / CPO accumulators. The core idea: every
# pointwise predictive integral can be reduced to a per-observation set
# of `(log_lik, log_weight)` pairs. The "samples" come from either:
#
#   - Gauss-Hermite quadrature over each obs's scalar linear predictor
#     (when the obs likelihood supports `linear_predictor_marginals`),
#   - Monte Carlo sampling from the Gaussian approximation (otherwise),
#   - Per-component concatenation of the two for composite likelihoods.
#
# Both accumulators consume the same `Vector{PointwiseLogLikSamples}`;
# WAIC does `logsumexp(log_w + log_lik)` / weighted mean, CPO does the
# same on negated log-lik with PSIS smoothing.

using Random: AbstractRNG, default_rng
using StatsFuns: logsumexp
using FastGaussQuadrature: gausshermite
using GaussianMarkovRandomFields

# ─────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────
"""
    ObsIntegralData

Per-observation data carried from the gather step to the accumulator
aggregation. Two concrete subtypes:

- `PointwiseLogLikSamples`: log-likelihood values + log-weights from
  either Gauss-Hermite quadrature (analytic path) or Monte Carlo
  sampling (fallback). Accumulator aggregations integrate against this
  empirical-style representation.
- `NormalIdentityClosedForm`: η-marginal `(μ_η, v_η)` plus the
  observation `y` and noise scale `σ`. Used when the stripped η-likelihood
  is Normal-IdentityLink, where both WAIC and CPO have exact closed-form
  pointwise integrals that GH-15 would only approximate (CPO's
  `exp((y-η)²/(2σ²))` integrand is exponentially-growing in η and
  Gauss-Hermite under-resolves the tail; the closed form also encodes
  the `v ≥ σ²` divergence as `Inf` rather than a large finite number).
"""
abstract type ObsIntegralData end

struct PointwiseLogLikSamples <: ObsIntegralData
    log_lik::Vector{Float64}
    log_weight::Vector{Float64}
end

struct NormalIdentityClosedForm <: ObsIntegralData
    μ_η::Float64
    v_η::Float64
    y::Float64
    σ::Float64
end

# ─────────────────────────────────────────────────────────────────────────
# Capability trait + dispatch
# ─────────────────────────────────────────────────────────────────────────
"""
    _supports_lpm(obs_lik) -> Bool

True iff `linear_predictor_marginals(ga, obs_lik)` is defined.
Composites recurse: `true` only when every component supports it. When
the trait returns `false` for a composite, the per-component gather
falls back to MC for unsupported components and keeps analytic GH for
the rest — supported channels never incur MC noise.
"""
_supports_lpm(::Any) = false
_supports_lpm(::GaussianMarkovRandomFields.ExponentialFamilyLikelihood) = true
_supports_lpm(lik::GaussianMarkovRandomFields.LinearlyTransformedLikelihood) =
    _supports_lpm(lik.base_likelihood)
_supports_lpm(lik::GaussianMarkovRandomFields.CompositeLikelihood) =
    all(_supports_lpm, lik.components)

"""
    _validate_fallback(fallback::Symbol)

Throw `ArgumentError` unless `fallback` is one of `:sample`, `:error`.
"""
function _validate_fallback(fallback::Symbol)
    fallback in (:sample, :error) ||
        throw(ArgumentError("fallback must be :sample or :error"))
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────
# Gather: one entry point that returns per-obs samples
# ─────────────────────────────────────────────────────────────────────────
"""
    _gather_pointwise_samples(ga, obs_lik; n_nodes, n_samples, fallback) ->
        (samples::Vector{PointwiseLogLikSamples}, eta_lik)

For each observation, produce `(log_lik, log_weight)` summarising its
posterior-predictive integral under the Gaussian approximation `ga`.

`eta_lik` is the observation likelihood expressed in η-coordinates (for
analytic paths) or the original `obs_lik` (for MC paths). Consumers
that need to evaluate per-obs CDFs (PIT) use it directly.

For mixed composites, returns a flat per-obs vector concatenated across
components (component order matches the composite's layout).
"""
function _gather_pointwise_samples(
        ga, obs_lik;
        n_nodes::Int, n_samples::Int, fallback::Symbol, latent_mean_override = nothing,
    )
    if _supports_lpm(obs_lik)
        return _gh_pointwise(ga, obs_lik, n_nodes; latent_mean_override = latent_mean_override)
    end
    if obs_lik isa GaussianMarkovRandomFields.CompositeLikelihood
        return _mixed_pointwise(ga, obs_lik, n_nodes, n_samples, fallback)
    end
    fallback === :error && throw(_unsupported_error(obs_lik))
    _warn_mc_fallback(obs_lik)
    return _mc_pointwise(ga, obs_lik, n_samples)
end

# Analytic path. When the stripped η-likelihood is Normal-IdentityLink,
# emit per-obs `NormalIdentityClosedForm` records (accumulators use the
# closed-form integrals). Otherwise fall back to Gauss-Hermite samples.
# Shift a predictor mean μ_η = A·μ0 to A·μ* for a VBC latent-mean override (the
# offset is already in μ_η; the variance is untouched — VBC corrects only the
# mean). No-op when `override` is nothing (non-VBC accumulator runs).
_apply_vbc_predictor_shift(μ_η, ga, obs_lik, override) =
    override === nothing ? μ_η : μ_η .+ obs_lik.design_matrix * (override .- mean(ga))

function _gh_pointwise(ga, obs_lik, n_nodes::Int; latent_mean_override = nothing)
    μ_η, v_η, eta_lik = GaussianMarkovRandomFields.linear_predictor_marginals(ga, obs_lik)
    μ_η = _apply_vbc_predictor_shift(μ_η, ga, obs_lik, latent_mean_override)
    return _from_eta_marginals(μ_η, v_η, eta_lik, n_nodes), eta_lik
end

function _from_eta_marginals(μ_η, v_η, eta_lik, n_nodes::Int)
    if eta_lik isa GaussianMarkovRandomFields.NormalLikelihood{GaussianMarkovRandomFields.IdentityLink}
        return _normal_identity_closed_form(μ_η, v_η, eta_lik)
    end
    if eta_lik isa GaussianMarkovRandomFields.CompositeLikelihood
        return _composite_eta_marginals(μ_η, v_η, eta_lik, n_nodes)
    end
    return _gh_samples_from_marginals(μ_η, v_η, eta_lik, n_nodes)
end

function _normal_identity_closed_form(μ_η, v_η, eta_lik)
    σ = eta_lik.σ
    y = eta_lik.y
    indices = eta_lik.indices
    # `μ_η` / `v_η` are laid out per the eta_lik's `indices` slice; map
    # each obs index back to its (μ_η_i, v_η_i, y_i) triple.
    n_obs = length(y)
    return [
        NormalIdentityClosedForm(
                μ_η[indices === nothing ? i : indices[i]],
                v_η[indices === nothing ? i : indices[i]],
                y[i], σ,
            ) for i in 1:n_obs
    ]
end

# Composite η-likelihood: each component carries its own indices into
# the concatenated (μ_η, v_η) vector. Recurse per component, concatenate
# per-obs records (mixing closed-form and GH-samples as appropriate).
function _composite_eta_marginals(
        μ_η, v_η, eta_lik::GaussianMarkovRandomFields.CompositeLikelihood, n_nodes::Int,
    )
    parts = ObsIntegralData[]
    for comp in eta_lik.components
        comp_records = if comp isa GaussianMarkovRandomFields.NormalLikelihood{GaussianMarkovRandomFields.IdentityLink}
            _normal_identity_closed_form(μ_η, v_η, comp)
        else
            # Slice μ_η, v_η for this component, then build GH samples.
            idx = comp.indices === nothing ? (1:length(μ_η)) : comp.indices
            _gh_samples_from_marginals(μ_η[idx], v_η[idx], comp, n_nodes)
        end
        append!(parts, comp_records)
    end
    return parts
end

function _gh_samples_from_marginals(μ_η, v_η, eta_lik, n_nodes::Int)
    nodes, weights = gausshermite(n_nodes)
    log_w = log.(weights ./ sqrt(π))
    σ_η = sqrt.(v_η)
    n_obs = length(μ_η)
    ll_matrix = Matrix{Float64}(undef, n_nodes, n_obs)
    for j in 1:n_nodes
        η_j = μ_η .+ sqrt(2) .* σ_η .* nodes[j]
        ll_matrix[j, :] .= GaussianMarkovRandomFields.pointwise_loglik(η_j, eta_lik)
    end
    return [PointwiseLogLikSamples(ll_matrix[:, i], log_w) for i in 1:n_obs]
end

# Monte Carlo: sample `x` from `ga`, evaluate `pointwise_loglik(x_s, obs_lik)`
# per sample. All MC samples carry uniform log-weight `-log(n_samples)`.
function _mc_pointwise(ga, obs_lik, n_samples::Int)
    samples_x = [rand(default_rng(), ga) for _ in 1:n_samples]
    first_ll = GaussianMarkovRandomFields.pointwise_loglik(samples_x[1], obs_lik)
    n_obs = length(first_ll)
    ll_matrix = Matrix{Float64}(undef, n_samples, n_obs)
    ll_matrix[1, :] .= first_ll
    for s in 2:n_samples
        ll_matrix[s, :] .= GaussianMarkovRandomFields.pointwise_loglik(samples_x[s], obs_lik)
    end
    log_w = fill(-log(n_samples), n_samples)
    samples_v = [
        PointwiseLogLikSamples(ll_matrix[:, i], log_w) for i in 1:n_obs
    ]
    return samples_v, obs_lik
end

# Mixed composite: per-component dispatch. Supported components run the
# analytic GH path; unsupported components share one batch of MC samples
# from `ga`. Concatenate per-obs results across components.
function _mixed_pointwise(
        ga, lik::GaussianMarkovRandomFields.CompositeLikelihood,
        n_nodes::Int, n_samples::Int, fallback::Symbol,
    )
    if fallback === :error
        unsupported = [nameof(typeof(c)) for c in lik.components if !_supports_lpm(c)]
        isempty(unsupported) ||
            throw(_composite_unsupported_error(lik))
    end
    # Pre-draw shared samples if any unsupported component exists.
    any_unsupported = any(!_supports_lpm(c) for c in lik.components)
    if any_unsupported
        _warn_mc_fallback(lik)
    end
    shared_samples = any_unsupported ?
        [rand(default_rng(), ga) for _ in 1:n_samples] : nothing
    log_w_mc = any_unsupported ? fill(-log(n_samples), n_samples) : nothing

    # Abstract element type — supported Normal-Identity components emit
    # `NormalIdentityClosedForm` records while other supported / MC paths
    # emit `PointwiseLogLikSamples`. The narrow `PointwiseLogLikSamples[]`
    # would silently `convert`-fail at append time.
    all_samples = ObsIntegralData[]
    eta_liks = Any[]   # for PIT: one eta-likelihood per component
    for comp in lik.components
        comp_samples, comp_eta_lik = if _supports_lpm(comp)
            _gh_pointwise(ga, comp, n_nodes)
        else
            comp_first = GaussianMarkovRandomFields.pointwise_loglik(shared_samples[1], comp)
            n_comp_obs = length(comp_first)
            ll_m = Matrix{Float64}(undef, n_samples, n_comp_obs)
            ll_m[1, :] .= comp_first
            for s in 2:n_samples
                ll_m[s, :] .= GaussianMarkovRandomFields.pointwise_loglik(shared_samples[s], comp)
            end
            (
                [PointwiseLogLikSamples(ll_m[:, i], log_w_mc) for i in 1:n_comp_obs],
                comp,
            )
        end
        append!(all_samples, comp_samples)
        push!(eta_liks, comp_eta_lik)
    end
    # Reconstruct a CompositeLikelihood of the per-component eta-likelihoods
    # if every component contributed one; consumers that need PIT can
    # dispatch on it (the analytic path produces real η-likelihoods, the
    # MC path returns the original component which may or may not have a
    # CDF method).
    return all_samples, GaussianMarkovRandomFields.CompositeLikelihood(Tuple(eta_liks))
end

# ─────────────────────────────────────────────────────────────────────────
# Error / warning helpers
# ─────────────────────────────────────────────────────────────────────────
function _unsupported_error(obs_lik)
    return ArgumentError(
        "Accumulator: no `linear_predictor_marginals` method for " *
            "$(nameof(typeof(obs_lik))) and `fallback = :error`. " *
            "Pass `fallback = :sample` on the strategy to use the " *
            "sample-based fallback, or define " *
            "`linear_predictor_marginals(ga, ::$(typeof(obs_lik)))`."
    )
end

function _composite_unsupported_error(lik)
    unsupported = [nameof(typeof(c)) for c in lik.components if !_supports_lpm(c)]
    return ArgumentError(
        "Accumulator: composite has unsupported components $(unsupported) " *
            "and `fallback = :error`. Pass `fallback = :sample`."
    )
end

function _warn_mc_fallback(obs_lik)
    @warn "Accumulator: using Monte Carlo for $(nameof(typeof(obs_lik))) " *
        "(no `linear_predictor_marginals` method). Results carry MC error; " *
        "tune `n_samples` or set `fallback = :error` to require analytic " *
        "support." maxlog = 1
    return nothing
end
