export CPOAccumulator, CPOStrategy, CPOPointSummary

using FastGaussQuadrature: gausshermite
using StatsFuns: logsumexp, logaddexp

"""
    CPOAccumulator(; n_nodes=15, compute_pit=true)

Compute Conditional Predictive Ordinates (CPO) and Probability Integral Transform (PIT).

CPO provides leave-one-out cross-validation diagnostics without refitting the model,
using the harmonic mean identity (Rue, Martino, Chopin 2009):

    1/CPO_i = E_{π(θ|y)} [ E_{π(x_i|y,θ)} [ 1/p(y_i|x_i) ] ]

PIT provides calibration diagnostics:

    PIT_i = E_{π(θ|y)} [ E_{π(x_i|y,θ)} [ F(y_i|x_i) ] ]

where F is the CDF. For discrete observations, the midpoint PIT is used.

# Keyword Arguments
- `n_nodes::Int = 15`: Number of Gauss-Hermite quadrature nodes
- `compute_pit::Bool = true`: Whether to compute PIT values

# Fields (after finalize!)
- `CPO::Vector{Float64}`: CPO_i for each observation
- `log_CPO::Vector{Float64}`: log(CPO_i) for each observation
- `LPML::Float64`: Log pseudo-marginal likelihood = Σ log(CPO_i)
- `PIT::Vector{Float64}`: PIT_i for each observation (if compute_pit)
- `failure::Vector{Float64}`: Per-observation failure score (0 = reliable, >0 = suspect)
- `n_failures::Int`: Number of observations with CPO computation failures

# Failure Detection

CPO reliability is assessed at two levels:

**Inner (GH quadrature)**: For each (observation, θ_k) pair, computes the effective sample
size (ESS) of the GH node contributions. ESS ≈ 1 means one node dominates the sum —
the GH estimate is unreliable. ESS ≈ n_nodes means all nodes contribute equally.

**Outer (θ integration)**: Computes the weighted effective sample size (ESS) of the
`w_k · h_i(θ_k)` contributions. Low ESS means one integration point dominates the CPO sum.

The `failure` score combines both: `max(inner_score, outer_score)` where
`inner_score = max(0, 1 - min_inner_ESS)` and `outer_score = max(0, 1 - outer_ESS)`.
Observations with `failure > 0` should be interpreted with caution.

# References
Rue, Martino & Chopin (2009). "Approximate Bayesian inference for latent
Gaussian models by using integrated nested Laplace approximations."
"""
mutable struct CPOAccumulator <: PosteriorAccumulator
    # Configuration
    n_nodes::Int
    compute_pit::Bool

    # Accumulated data (one vector per integration point)
    log_inv_lik_expectations::Vector{Vector{Float64}}  # log E[1/p(y_i|x_i)] per θ_k
    pit_expectations::Vector{Vector{Float64}}           # E[F(y_i|x_i)] per θ_k
    inner_ess::Vector{Vector{Float64}}                   # effective sample size of GH nodes per θ_k

    # Results (computed in finalize!)
    CPO::Vector{Float64}
    log_CPO::Vector{Float64}
    LPML::Float64
    PIT::Vector{Float64}
    failure::Vector{Float64}
    n_failures::Int

    CPOAccumulator(; n_nodes::Int = 15, compute_pit::Bool = true) = new(
        n_nodes, compute_pit,
        Vector{Float64}[], Vector{Float64}[], Vector{Float64}[],
        Float64[], Float64[], 0.0, Float64[], Float64[], 0
    )
end

"""
    CPOStrategy(; n_nodes=15, compute_pit=true)

Immutable config requesting CPO / PIT computation during `inla()`. Materialises
into a fresh `CPOAccumulator(; n_nodes, compute_pit)` per run.
"""
struct CPOStrategy <: PosteriorStrategy
    n_nodes::Int
    compute_pit::Bool
    CPOStrategy(; n_nodes::Int = 15, compute_pit::Bool = true) = new(n_nodes, compute_pit)
end

materialize(s::CPOStrategy) = CPOAccumulator(; n_nodes = s.n_nodes, compute_pit = s.compute_pit)

# ── Pointwise CDF helpers ─────────────────────────────────────────────

"""
    _pointwise_cdf(x, obs_lik)

Compute per-observation CDF values F(y_i | x_i) for each observation.
For discrete distributions, returns the midpoint PIT: F(y_i) - 0.5·f(y_i).
"""
function _pointwise_cdf end

# LinearlyTransformedLikelihood: compute η = A·x, delegate to base lik.
# Needed for CPO on non-augmented LTM-obs LGMs; the augmented path
# stores an ExpFam likelihood directly and hits the family-specific
# methods below.
function _pointwise_cdf(x, ltlik::GaussianMarkovRandomFields.LinearlyTransformedLikelihood)
    η = ltlik.design_matrix * x
    return _pointwise_cdf(η, ltlik.base_likelihood)
end

function _pointwise_cdf(x, obs_lik::NormalLikelihood{IdentityLink})
    y = obs_lik.y
    σ = obs_lik.σ
    indices = obs_lik.indices
    n_obs = length(y)
    result = Vector{Float64}(undef, n_obs)
    for i in 1:n_obs
        idx = indices === nothing ? i : indices[i]
        result[i] = cdf(Normal(x[idx], σ), y[i])
    end
    return result
end

function _pointwise_cdf(x, obs_lik::NormalLikelihood)
    y = obs_lik.y
    σ = obs_lik.σ
    indices = obs_lik.indices
    n_obs = length(y)
    result = Vector{Float64}(undef, n_obs)
    for i in 1:n_obs
        idx = indices === nothing ? i : indices[i]
        μ = apply_invlink(obs_lik.link, x[idx])
        result[i] = cdf(Normal(μ, σ), y[i])
    end
    return result
end

# Discrete distributions use midpoint PIT: cdf(d, y) - 0.5·pdf(d, y)
function _pointwise_cdf(x, obs_lik::PoissonLikelihood)
    y = obs_lik.y
    indices = obs_lik.indices
    n_obs = length(y)
    result = Vector{Float64}(undef, n_obs)
    for i in 1:n_obs
        idx = indices === nothing ? i : indices[i]
        λ = apply_invlink(obs_lik.link, x[idx])
        if obs_lik.logexposure !== nothing
            λ *= exp(obs_lik.logexposure[i])
        end
        d = Poisson(max(λ, 1.0e-20))
        result[i] = cdf(d, y[i]) - 0.5 * pdf(d, y[i])
    end
    return result
end

function _pointwise_cdf(x, obs_lik::BernoulliLikelihood)
    y = obs_lik.y
    indices = obs_lik.indices
    n_obs = length(y)
    result = Vector{Float64}(undef, n_obs)
    for i in 1:n_obs
        idx = indices === nothing ? i : indices[i]
        p = apply_invlink(obs_lik.link, x[idx])
        d = Bernoulli(clamp(p, 1.0e-10, 1 - 1.0e-10))
        result[i] = cdf(d, y[i]) - 0.5 * pdf(d, y[i])
    end
    return result
end

function _pointwise_cdf(x, obs_lik::BinomialLikelihood)
    y = obs_lik.y
    indices = obs_lik.indices
    n_obs = length(y)
    result = Vector{Float64}(undef, n_obs)
    for i in 1:n_obs
        idx = indices === nothing ? i : indices[i]
        p = apply_invlink(obs_lik.link, x[idx])
        d = Binomial(obs_lik.n[i], clamp(p, 1.0e-10, 1 - 1.0e-10))
        result[i] = cdf(d, y[i]) - 0.5 * pdf(d, y[i])
    end
    return result
end

# ── CPO/PIT integral computation ──────────────────────────────────────

"""
    _cpo_pit_integrals(ga, obs_lik; n_nodes=15, compute_pit=true)

Compute per-observation CPO and PIT integrals by integrating over the
latent field marginals from the Gaussian approximation.

Returns `(log_inv_lik_exp, pit_exp, inner_ess)` where:
- `log_inv_lik_exp[i]` = log E_{N(μ_i,v_i)}[1/p(y_i|x_i)]  (for CPO)
- `pit_exp[i]` = E_{N(μ_i,v_i)}[F(y_i|x_i)]  (for PIT)
- `inner_ess[i]` = effective sample size of GH node contributions (for failure detection)

For Normal+IdentityLink, CPO and PIT are analytic and `inner_ess` = n_nodes (perfect).
Otherwise, Gauss-Hermite quadrature is used.
"""
function _cpo_pit_integrals end

# Analytic for Normal + IdentityLink
function _cpo_pit_integrals(
        ga, obs_lik::NormalLikelihood{IdentityLink};
        n_nodes::Int = 15, compute_pit::Bool = true
    )
    μ = mean(ga)
    v = var(ga)
    σ_obs = obs_lik.σ
    y = obs_lik.y
    indices = obs_lik.indices

    n_obs = length(y)
    log_inv_lik_exp = Vector{Float64}(undef, n_obs)
    pit_exp = compute_pit ? Vector{Float64}(undef, n_obs) : Float64[]

    for i in 1:n_obs
        idx = indices === nothing ? i : indices[i]
        ratio = v[idx] / σ_obs^2

        if ratio >= 1.0
            # CPO failure: integral diverges when posterior variance >= obs variance
            log_inv_lik_exp[i] = Inf
        else
            # log E[1/p(y|x)] where p(y|x) = N(y; x, σ), x ~ N(μ, v)
            # = 0.5·log(2π·σ²) - 0.5·log(1-v/σ²) + (y-μ)²/(2σ²·(1-v/σ²))
            log_inv_lik_exp[i] = 0.5 * log(2π * σ_obs^2) -
                0.5 * log(1 - ratio) +
                (y[i] - μ[idx])^2 / (2 * σ_obs^2 * (1 - ratio))
        end

        if compute_pit
            # PIT = Φ((y - μ) / √(σ² + v))
            pit_exp[i] = cdf(Normal(μ[idx], sqrt(σ_obs^2 + v[idx])), y[i])
        end
    end

    # Analytic path has no inner quadrature — perfect ESS
    inner_ess = fill(Float64(n_nodes), n_obs)

    return log_inv_lik_exp, pit_exp, inner_ess
end

# Generic: Gauss-Hermite quadrature
function _cpo_pit_integrals(
        ga, obs_lik;
        n_nodes::Int = 15, compute_pit::Bool = true
    )
    μ = mean(ga)
    σ = std(ga)
    nodes, weights = gausshermite(n_nodes)
    scaled_weights = weights ./ sqrt(π)
    log_scaled_weights = log.(scaled_weights)

    # Evaluate at all quadrature nodes
    ll_at_nodes = Vector{Vector{Float64}}(undef, n_nodes)
    cdf_at_nodes = compute_pit ? Vector{Vector{Float64}}(undef, n_nodes) : nothing
    for j in 1:n_nodes
        x_j = μ .+ sqrt(2) .* σ .* nodes[j]
        ll_at_nodes[j] = pointwise_loglik(x_j, obs_lik)
        if compute_pit
            cdf_at_nodes[j] = _pointwise_cdf(x_j, obs_lik)
        end
    end

    n_obs = length(ll_at_nodes[1])
    log_inv_lik_exp = Vector{Float64}(undef, n_obs)
    pit_exp = compute_pit ? Vector{Float64}(undef, n_obs) : Float64[]
    inner_ess = Vector{Float64}(undef, n_obs)

    # Pre-allocate buffers to avoid per-observation allocations
    log_terms = Vector{Float64}(undef, n_nodes)
    log_terms_sq = Vector{Float64}(undef, n_nodes)

    for i in 1:n_obs
        # log E[1/p(y_i|x_i)] = logsumexp_j(log(w_j/√π) + (-ll_j[i]))
        for j in 1:n_nodes
            log_terms[j] = log_scaled_weights[j] - ll_at_nodes[j][i]
        end
        log_inv_lik_exp[i] = logsumexp(log_terms)

        # Inner reliability: effective sample size of GH contributions
        # ESS = (Σ w_j·f_j)² / (Σ w_j²·f_j²) where f_j = exp(-ll_j[i])
        # In log space: log(ESS) = 2·logsumexp(log_terms) - logsumexp(2·log_terms)
        for j in 1:n_nodes
            log_terms_sq[j] = 2.0 * log_terms[j]
        end
        inner_ess[i] = exp(2.0 * log_inv_lik_exp[i] - logsumexp(log_terms_sq))

        if compute_pit
            # E[F(y_i|x_i)] = Σ_j (w_j/√π) · F_j[i]
            pit_exp[i] = sum(scaled_weights[j] * cdf_at_nodes[j][i] for j in 1:n_nodes)
        end
    end

    return log_inv_lik_exp, pit_exp, inner_ess
end

# ── Accumulator interface ─────────────────────────────────────────────

"""Pre-computed summary data for one grid point (CPO)."""
struct CPOPointSummary
    log_inv_lik_exp::Vector{Float64}
    pit_exp::Vector{Float64}
    inner_ess::Vector{Float64}
end

function compute_point_summary(acc::CPOAccumulator; ga, obs_lik, kwargs...)
    _maybe_disable_pit!(acc, obs_lik)
    log_h, pit, ess = _cpo_pit_integrals(
        ga, obs_lik; n_nodes = acc.n_nodes, compute_pit = acc.compute_pit
    )
    return CPOPointSummary(log_h, pit, ess)
end

# PIT (`E[F(y_i|x_i)]`) needs `_pointwise_cdf`, which is only defined for the
# specific likelihood families this file dispatches on. For black-box AD-routed
# likelihoods (e.g. an `AutoDiffLikelihood` from a DPPL model that doesn't fit
# the fast path), no `_pointwise_cdf` method exists. Auto-fall-back to CPO-only
# with a one-time warning so users get the rest of the diagnostics rather than
# a hard crash.
function _maybe_disable_pit!(acc::CPOAccumulator, obs_lik)
    if acc.compute_pit && !hasmethod(_pointwise_cdf, Tuple{Vector{Float64}, typeof(obs_lik)})
        @warn "CPO: PIT unavailable for $(nameof(typeof(obs_lik))); computing CPO only. Pass `CPOStrategy(compute_pit=false)` to silence." maxlog = 1
        acc.compute_pit = false
    end
    return nothing
end

function accumulate!(acc::CPOAccumulator, summary::CPOPointSummary; kwargs...)
    push!(acc.log_inv_lik_expectations, summary.log_inv_lik_exp)
    push!(acc.inner_ess, summary.inner_ess)
    if acc.compute_pit
        push!(acc.pit_expectations, summary.pit_exp)
    end
    return nothing
end

function accumulate!(
        acc::CPOAccumulator;
        ga,
        obs_lik,
        kwargs...
    )
    _maybe_disable_pit!(acc, obs_lik)
    log_h, pit, ess = _cpo_pit_integrals(
        ga, obs_lik; n_nodes = acc.n_nodes, compute_pit = acc.compute_pit
    )
    push!(acc.log_inv_lik_expectations, log_h)
    push!(acc.inner_ess, ess)
    if acc.compute_pit
        push!(acc.pit_expectations, pit)
    end
    return nothing
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

    # Pre-allocate buffers for per-observation loops
    log_terms = Vector{Float64}(undef, n_points)

    for i in 1:n_obs
        # 1/CPO_i = Σ_k w_k · h_i(θ_k)
        # In log space: log(1/CPO_i) = logsumexp_k(log(w_k) + log_h_i(θ_k))
        all_finite = true
        for k in 1:n_points
            log_h_k = acc.log_inv_lik_expectations[k][i]
            log_terms[k] = log_weights[k] + log_h_k
            all_finite = all_finite && isfinite(log_h_k)
        end
        log_inv_cpo = logsumexp(log_terms)

        acc.log_CPO[i] = -log_inv_cpo
        acc.CPO[i] = exp(-log_inv_cpo)

        # Inner failure: minimum ESS across all θ_k
        # ESS measures how many GH nodes effectively contribute.
        # When ESS ≈ 1, the estimate is dominated by a single node.
        min_inner_ess = minimum(acc.inner_ess[k][i] for k in 1:n_points)
        inner_score = max(0.0, 1.0 - min_inner_ess)

        # Outer failure: ESS of w_k·h_i(θ_k) contributions across integration points.
        # Measures whether one θ_k dominates the CPO sum (analogous to inner ESS).
        # ESS = (Σ w_k·h_k)² / Σ (w_k·h_k)², in [1, n_points]. Computed in log space.
        if all_finite && n_points > 1
            # log_inv_cpo = log(Σ w_k h_k) — already computed
            # log_terms[k] = log(w_k h_k), so logsumexp(2·log_terms) = log(Σ w_k² h_k²)
            log_sum_sq = logsumexp(2.0 .* log_terms)
            outer_ess = exp(2.0 * log_inv_cpo - log_sum_sq)
            outer_score = max(0.0, 1.0 - outer_ess)
        else
            outer_score = isfinite(log_inv_cpo) ? 0.0 : Inf
        end

        acc.failure[i] = max(inner_score, outer_score)

        if !isfinite(acc.CPO[i]) || acc.CPO[i] <= 0 || acc.failure[i] > 0
            acc.n_failures += 1
        end
    end

    acc.LPML = sum(lc for lc in acc.log_CPO if isfinite(lc))

    if acc.compute_pit
        acc.PIT = Vector{Float64}(undef, n_obs)
        for i in 1:n_obs
            acc.PIT[i] = sum(weights[k] * acc.pit_expectations[k][i] for k in 1:n_points)
        end
    end

    return nothing
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
            max_fail = maximum(acc.failure)
            println(io, "  Max failure score: ", round(max_fail, digits = 2))
        end
    end
    if acc.compute_pit && !isempty(acc.PIT)
        println(io, "  PIT computed: ", length(acc.PIT), " values")
        return println(io, "  PIT mean: ", round(mean(acc.PIT), digits = 4), " (ideal: 0.5)")
    end
end
