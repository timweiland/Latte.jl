export CPOAccumulator, CPOStrategy, CPOPointSummary

using StatsFuns: logsumexp

"""
    CPOStrategy(; n_nodes=15, compute_pit=true, fallback=:sample, n_samples=512)

Conditional Predictive Ordinates (CPO) + Probability Integral Transform (PIT),
the standard INLA leave-one-out diagnostics (Rue, Martino, Chopin 2009).
For each observation:

    1/CPO_i = E_post[ 1/p(y_i | x) ]
    PIT_i   = E_post[ F(y_i | x) ]

Expectations are taken under the Gaussian approximation, integrating
over each obs's scalar linear predictor η_i. The unified gather
(`_gather_pointwise_samples`) routes between analytic Gauss-Hermite
(when the obs likelihood supports `linear_predictor_marginals`) and
Monte Carlo (otherwise); composite likelihoods mix per-component.

CPO uses Pareto-Smoothed Importance Sampling on `log_weight − log_lik`
to stabilise the inverse-likelihood expectation when samples are MC.
The per-obs PSIS shape parameter `k̂` is folded into the `failure`
field; `k̂ > 0.7` flags the LOO estimate as unreliable (Vehtari et al.
2017/2024).

PIT requires a per-likelihood CDF (`_pointwise_cdf`); when unavailable
PIT is silently disabled and `compute_pit` is preserved as the user's
original request.

# Fields (after `finalize!`)
- `CPO::Vector{Float64}`, `log_CPO`, `LPML::Float64 = sum(log_CPO)`
- `PIT::Vector{Float64}` (when CDF available)
- `failure::Vector{Float64}` per-obs failure score (combines outer
  integration-weight ESS and PSIS k̂)
- `n_failures::Int`
"""
struct CPOStrategy <: PosteriorStrategy
    n_nodes::Int
    compute_pit::Bool
    fallback::Symbol
    n_samples::Int
    function CPOStrategy(;
            n_nodes::Int = 15, compute_pit::Bool = true,
            fallback::Symbol = :sample, n_samples::Int = 512,
        )
        _validate_fallback(fallback)
        return new(n_nodes, compute_pit, fallback, n_samples)
    end
end

mutable struct CPOAccumulator <: PosteriorAccumulator
    cfg::CPOStrategy
    pit_active::Bool                                       # tracks runtime PIT availability

    log_inv_lik_expectations::Vector{Vector{Float64}}      # log E[1/p(y_i|x)] per θ_k
    pit_expectations::Vector{Vector{Float64}}              # E[F(y_i|x)] per θ_k
    pareto_k::Vector{Vector{Float64}}                      # per-obs PSIS k̂ per θ_k (NaN on GH)

    CPO::Vector{Float64}
    log_CPO::Vector{Float64}
    LPML::Float64
    PIT::Vector{Float64}
    failure::Vector{Float64}
    n_failures::Int

    CPOAccumulator(cfg::CPOStrategy) = new(
        cfg, cfg.compute_pit,
        Vector{Float64}[], Vector{Float64}[], Vector{Float64}[],
        Float64[], Float64[], 0.0, Float64[], Float64[], 0,
    )
end

CPOAccumulator(; kwargs...) = CPOAccumulator(CPOStrategy(; kwargs...))

function Base.getproperty(acc::CPOAccumulator, name::Symbol)
    if name in fieldnames(CPOStrategy)
        return getfield(getfield(acc, :cfg), name)
    end
    return getfield(acc, name)
end

materialize(s::CPOStrategy) = CPOAccumulator(s)

# Per-obs CPO aggregation. Dispatches on the `ObsIntegralData` subtype.
#
# `PointwiseLogLikSamples`: log E[1/p] is an importance-sampling
# integral with log-weights `log_weight − log_lik`. PSIS smooths the
# upper tail via a generalised Pareto fit; the shape parameter k̂
# flags the obs when the tail is too heavy to trust.
function _cpo_aggregate(s::PointwiseLogLikSamples)
    inv_log_w = s.log_weight .- s.log_lik
    smoothed, k_hat = _psis_smooth_log_weights(inv_log_w)
    log_inv_lik = logsumexp(smoothed)
    return log_inv_lik, k_hat
end

# Normal-IdentityLink: closed form for `log E[1/p(y|η)]` where η ~ N(μ_η, v_η).
# Diverges when v_η ≥ σ² — encoded as Inf, which the failure-score logic
# in `finalize!` flags correctly.
function _cpo_aggregate(s::NormalIdentityClosedForm)
    ratio = s.v_η / s.σ^2
    if ratio >= 1.0
        return Inf, NaN
    end
    log_inv_lik = 0.5 * log(2π * s.σ^2) -
        0.5 * log(1 - ratio) +
        (s.y - s.μ_η)^2 / (2 * s.σ^2 * (1 - ratio))
    return log_inv_lik, NaN
end

struct CPOPointSummary
    log_inv_lik_exp::Vector{Float64}
    pit_exp::Vector{Float64}
    pareto_k::Vector{Float64}
end

function compute_point_summary(acc::CPOAccumulator; ga, obs_lik, x_star_vbc = nothing, kwargs...)
    samples, eta_lik = _gather_pointwise_samples(
        ga, obs_lik;
        n_nodes = acc.cfg.n_nodes,
        n_samples = acc.cfg.n_samples,
        fallback = acc.cfg.fallback,
        latent_mean_override = x_star_vbc,
    )

    # CPO: per-obs PSIS-smoothed inverse expectation.
    cpo_pairs = _cpo_aggregate.(samples)
    log_inv_lik_exp = first.(cpo_pairs)
    pareto_k = last.(cpo_pairs)

    # PIT: only when (a) user requested it AND (b) the gather's eta_lik
    # exposes a CDF method. The current PIT pathway needs η-values
    # threaded through samples; for the simplified pipeline we defer
    # PIT to the analytic path only and disable it for any obs whose
    # gather went through MC (those don't have a clean per-obs CDF
    # without storing η alongside log_lik).
    pit_exp = if acc.pit_active && _all_analytic_pit_supported(obs_lik, eta_lik)
        _analytic_pit(ga, obs_lik, acc.cfg.n_nodes; latent_mean_override = x_star_vbc)
    else
        acc.pit_active = false
        Float64[]
    end

    return CPOPointSummary(log_inv_lik_exp, pit_exp, pareto_k)
end

# Check whether every component (recursively) has an `_pointwise_cdf`
# method on its stripped η-likelihood. If yes, PIT can be computed
# analytically via `cdf(family(η), y)` integrated against `N(μ_η, v_η)`.
_all_analytic_pit_supported(obs_lik, eta_lik) =
    _supports_lpm(obs_lik) && _eta_lik_has_cdf(eta_lik)
_eta_lik_has_cdf(lik) =
    hasmethod(_pointwise_cdf, Tuple{Vector{Float64}, typeof(lik)})
_eta_lik_has_cdf(lik::GaussianMarkovRandomFields.CompositeLikelihood) =
    all(_eta_lik_has_cdf, lik.components)

# Analytic PIT: `E_η[F(y_i | η)]` via 1-D Gauss-Hermite over each obs's
# η-marginal. Mirrors the gather's GH layout.
function _analytic_pit(ga, obs_lik, n_nodes::Int; latent_mean_override = nothing)
    μ_η, v_η, eta_lik = GaussianMarkovRandomFields.linear_predictor_marginals(ga, obs_lik)
    μ_η = _apply_vbc_predictor_shift(μ_η, ga, obs_lik, latent_mean_override)
    nodes, weights = gausshermite(n_nodes)
    scaled_w = weights ./ sqrt(π)
    σ_η = sqrt.(v_η)
    n_obs = length(μ_η)
    pit = zeros(n_obs)
    for j in 1:n_nodes
        η_j = μ_η .+ sqrt(2) .* σ_η .* nodes[j]
        pit .+= scaled_w[j] .* _pointwise_cdf(η_j, eta_lik)
    end
    return pit
end

function accumulate!(acc::CPOAccumulator, summary::CPOPointSummary; kwargs...)
    push!(acc.log_inv_lik_expectations, summary.log_inv_lik_exp)
    push!(acc.pareto_k, summary.pareto_k)
    acc.pit_active && push!(acc.pit_expectations, summary.pit_exp)
    return nothing
end

function accumulate!(acc::CPOAccumulator; ga, obs_lik, kwargs...)
    accumulate!(acc, compute_point_summary(acc; ga = ga, obs_lik = obs_lik))
    return nothing
end

# Functional entry point — returns `(log_inv_lik, pit, pareto_k)`. Takes
# a `CPOStrategy` for configuration; callers that don't already have one
# can build it inline: `_cpo_pointwise_integrals(ga, obs_lik, CPOStrategy(; …))`.
function _cpo_pointwise_integrals(ga, obs_lik, cfg::CPOStrategy = CPOStrategy())
    samples, eta_lik = _gather_pointwise_samples(
        ga, obs_lik;
        n_nodes = cfg.n_nodes, n_samples = cfg.n_samples, fallback = cfg.fallback,
    )
    cpo_pairs = _cpo_aggregate.(samples)
    log_inv_lik = first.(cpo_pairs)
    pareto_k = last.(cpo_pairs)
    pit = if cfg.compute_pit && _all_analytic_pit_supported(obs_lik, eta_lik)
        _analytic_pit(ga, obs_lik, cfg.n_nodes)
    else
        Float64[]
    end
    return log_inv_lik, pit, pareto_k
end

function finalize!(acc::CPOAccumulator, exploration::AbstractHyperparameterExploration)
    weights = get_integration_weights(exploration)
    log_weights = log.(weights)
    n_obs = length(acc.log_inv_lik_expectations[1])
    n_points = length(weights)

    acc.CPO = Vector{Float64}(undef, n_obs)
    acc.log_CPO = Vector{Float64}(undef, n_obs)
    acc.failure = Vector{Float64}(undef, n_obs)
    acc.n_failures = 0

    log_terms = Vector{Float64}(undef, n_points)
    for i in 1:n_obs
        log_inv_cpo, all_finite = _per_obs_log_inv_cpo!(log_terms, acc, log_weights, i)
        acc.log_CPO[i] = -log_inv_cpo
        acc.CPO[i] = exp(-log_inv_cpo)

        outer = _outer_failure_score(log_terms, log_inv_cpo, all_finite, n_points)
        pareto = _pareto_failure_score(acc.pareto_k, i)
        acc.failure[i] = max(outer, pareto)
        if !isfinite(acc.CPO[i]) || acc.CPO[i] <= 0 || acc.failure[i] > 0
            acc.n_failures += 1
        end
    end

    acc.LPML = sum(lc for lc in acc.log_CPO if isfinite(lc))

    if acc.pit_active
        acc.PIT = Vector{Float64}(undef, n_obs)
        for i in 1:n_obs
            acc.PIT[i] = sum(weights[k] * acc.pit_expectations[k][i] for k in 1:n_points)
        end
    end
    return nothing
end

# Per-obs weighted log-sum-exp across integration points plus an
# all-finite flag for the outer-ESS branch downstream. Mutates `log_terms`
# in place.
function _per_obs_log_inv_cpo!(log_terms, acc::CPOAccumulator, log_weights, i)
    n_points = length(log_weights)
    all_finite = true
    for k in 1:n_points
        lh = acc.log_inv_lik_expectations[k][i]
        log_terms[k] = log_weights[k] + lh
        all_finite = all_finite && isfinite(lh)
    end
    return logsumexp(log_terms), all_finite
end

# ESS of the `w_k · h_i(θ_k)` contributions across integration points.
# Returns the failure score `max(0, 1 - ESS)` — 0 when contributions are
# evenly spread, → 1 when one integration point dominates.
function _outer_failure_score(log_terms, log_inv_cpo, all_finite, n_points)
    if !all_finite || n_points <= 1
        return isfinite(log_inv_cpo) ? 0.0 : Inf
    end
    log_sum_sq = logsumexp(2.0 .* log_terms)
    outer_ess = exp(2.0 * log_inv_cpo - log_sum_sq)
    return max(0.0, 1.0 - outer_ess)
end

# Largest finite PSIS k̂ across integration points for obs i, mapped to
# a failure score (0 below 0.7, growing past). NaN entries from analytic
# paths don't carry IS noise — they're skipped.
function _pareto_failure_score(pareto_k, i)
    max_k = -Inf
    for k in eachindex(pareto_k)
        kv = pareto_k[k][i]
        isfinite(kv) && kv > max_k && (max_k = kv)
    end
    return isfinite(max_k) ? max(0.0, max_k - 0.7) : 0.0
end

function Base.show(io::IO, ::MIME"text/plain", acc::CPOAccumulator)
    println(io, "Conditional Predictive Ordinates (CPO):")
    println(io, "  LPML: ", round(acc.LPML, digits = 2))
    n = length(acc.CPO)
    if n > 0
        println(io, "  Mean CPO: ", round(mean(acc.CPO), digits = 4))
        println(io, "  Min CPO: ", round(minimum(acc.CPO), digits = 4))
    end
    if acc.n_failures > 0
        println(io, "  Unreliable observations: ", acc.n_failures, " / ", n)
        if !isempty(acc.failure)
            println(io, "  Max failure score: ", round(maximum(acc.failure), digits = 2))
        end
    end
    if acc.pit_active && !isempty(acc.PIT)
        println(io, "  PIT computed: ", length(acc.PIT), " values")
        return println(io, "  PIT mean: ", round(mean(acc.PIT), digits = 4), " (ideal: 0.5)")
    end
end
