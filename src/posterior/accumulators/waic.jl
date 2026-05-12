export WAICAccumulator, WAICStrategy, WAICPointSummary

using StatsFuns: logsumexp

"""
    WAICStrategy(; n_nodes=15, fallback=:sample, n_samples=512)

Watanabe-Akaike Information Criterion. Computes WAIC = -2·(lppd - p_WAIC)
where lppd = ∑ᵢ log E_post[p(yᵢ|x)] and p_WAIC = 2·∑ᵢ (lppd_i − E_post[log p(yᵢ|x)]).

The expectations are taken under the Gaussian approximation to the
latent posterior, integrating over each observation's *scalar linear
predictor* η_i. For obs likelihoods supporting `linear_predictor_marginals`
(direct exponential-family obs, `LinearlyTransformedObservationModel`,
composites of these) this uses analytic η-marginals + 1-D Gauss-Hermite
quadrature. For obs without that support (e.g. `AutoDiffLikelihood`)
the fallback samples `x` from the Gaussian approximation.

`fallback`:
- `:sample` (default): MC fallback on unsupported obs.
- `:error`: raise on unsupported obs.

Lower WAIC = better predictive accuracy.

# References
Watanabe (2010); Gelman, Hwang & Vehtari (2014).
"""
struct WAICStrategy <: PosteriorStrategy
    n_nodes::Int
    fallback::Symbol
    n_samples::Int
    function WAICStrategy(;
            n_nodes::Int = 15, fallback::Symbol = :sample, n_samples::Int = 512,
        )
        _validate_fallback(fallback)
        return new(n_nodes, fallback, n_samples)
    end
end

mutable struct WAICAccumulator <: PosteriorAccumulator
    cfg::WAICStrategy

    integrated_lls::Vector{Vector{Float64}}
    expected_log_lls::Vector{Vector{Float64}}

    lppd::Float64
    p_WAIC::Float64
    WAIC::Float64

    WAICAccumulator(cfg::WAICStrategy) = new(
        cfg, Vector{Float64}[], Vector{Float64}[], 0.0, 0.0, 0.0,
    )
end

WAICAccumulator(; kwargs...) = WAICAccumulator(WAICStrategy(; kwargs...))

function Base.getproperty(acc::WAICAccumulator, name::Symbol)
    if name in fieldnames(WAICStrategy)
        return getfield(getfield(acc, :cfg), name)
    end
    return getfield(acc, name)
end

materialize(s::WAICStrategy) = WAICAccumulator(s)

# Per-obs aggregation. Dispatches on `ObsIntegralData` subtype:
#   - `PointwiseLogLikSamples`: GH / MC samples → numerical integration.
#   - `NormalIdentityClosedForm`: closed-form expressions (Normal × Normal).
function _waic_aggregate(s::PointwiseLogLikSamples)
    integrated_ll = logsumexp(s.log_weight .+ s.log_lik)
    expected_log_ll = sum(exp.(s.log_weight) .* s.log_lik)
    return integrated_ll, expected_log_ll
end

# Normal-IdentityLink: y ~ N(η, σ), η ~ N(μ_η, v_η).
# integrated_ll = log N(y; μ_η, √(σ² + v_η))
# expected_log_ll = E_η[log N(y; η, σ)] = -½ log(2π) - log σ - ((y-μ_η)² + v_η)/(2σ²)
function _waic_aggregate(s::NormalIdentityClosedForm)
    total_var = s.σ^2 + s.v_η
    integrated_ll = -0.5 * log(2π * total_var) - (s.y - s.μ_η)^2 / (2 * total_var)
    expected_log_ll = -0.5 * log(2π) - log(s.σ) -
        ((s.y - s.μ_η)^2 + s.v_η) / (2 * s.σ^2)
    return integrated_ll, expected_log_ll
end

struct WAICPointSummary
    integrated_ll::Vector{Float64}
    expected_log_ll::Vector{Float64}
end

function compute_point_summary(acc::WAICAccumulator; ga, obs_lik, kwargs...)
    samples, _ = _gather_pointwise_samples(
        ga, obs_lik;
        n_nodes = acc.cfg.n_nodes,
        n_samples = acc.cfg.n_samples,
        fallback = acc.cfg.fallback,
    )
    pairs = _waic_aggregate.(samples)
    return WAICPointSummary(first.(pairs), last.(pairs))
end

function accumulate!(acc::WAICAccumulator, summary::WAICPointSummary; kwargs...)
    push!(acc.integrated_lls, summary.integrated_ll)
    push!(acc.expected_log_lls, summary.expected_log_ll)
    return nothing
end

function accumulate!(acc::WAICAccumulator; ga, obs_lik, kwargs...)
    accumulate!(acc, compute_point_summary(acc; ga = ga, obs_lik = obs_lik))
    return nothing
end

# Functional entry point — used by tests that want `(integrated_ll,
# expected_log_ll)` directly. Takes a `WAICStrategy` for configuration.
function _waic_pointwise_integrals(ga, obs_lik, cfg::WAICStrategy = WAICStrategy())
    samples, _ = _gather_pointwise_samples(
        ga, obs_lik;
        n_nodes = cfg.n_nodes, n_samples = cfg.n_samples, fallback = cfg.fallback,
    )
    pairs = _waic_aggregate.(samples)
    return first.(pairs), last.(pairs)
end

function finalize!(acc::WAICAccumulator, exploration::AbstractHyperparameterExploration)
    weights = get_integration_weights(exploration)
    log_weights = log.(weights)
    n_points = length(weights)
    n_obs = length(acc.integrated_lls[1])

    lppd_total = 0.0
    mean_ell_total = 0.0
    for i in 1:n_obs
        log_terms = [log_weights[k] + acc.integrated_lls[k][i] for k in 1:n_points]
        lppd_total += logsumexp(log_terms)
        mean_ell_total += sum(weights[k] * acc.expected_log_lls[k][i] for k in 1:n_points)
    end

    acc.lppd = lppd_total
    acc.p_WAIC = 2.0 * (lppd_total - mean_ell_total)
    acc.WAIC = -2 * (lppd_total - acc.p_WAIC)
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", acc::WAICAccumulator)
    println(io, "Watanabe-Akaike Information Criterion (WAIC):")
    println(io, "  WAIC: ", round(acc.WAIC, digits = 2))
    println(io, "  Effective parameters (p_WAIC): ", round(acc.p_WAIC, digits = 2))
    return println(io, "  Log pointwise predictive density (lppd): ", round(acc.lppd, digits = 2))
end
