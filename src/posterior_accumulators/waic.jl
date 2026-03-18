export WAICAccumulator, WAICPointSummary

using FastGaussQuadrature: gausshermite
using StatsFuns: logsumexp

"""
    WAICAccumulator(; n_nodes=15)

Compute Watanabe-Akaike Information Criterion (WAIC) using the p_WAIC1 formula.

WAIC is a fully Bayesian model comparison metric that uses the entire
posterior distribution, making it strictly preferred over DIC:
- WAIC = -2 * (lppd - p_WAIC)
- lppd = ∑ᵢ log(E_post[p(yᵢ|xᵢ)]) (log pointwise predictive density)
- p_WAIC = 2 * ∑ᵢ (lppd_i - E_post[log p(yᵢ|xᵢ)]) (effective number of parameters)

The expectations integrate over BOTH the latent field (via Gauss-Hermite
quadrature or analytically) AND hyperparameters (via grid integration weights).

Lower WAIC indicates better out-of-sample predictive performance.

# Keyword Arguments
- `n_nodes::Int = 15`: Number of Gauss-Hermite quadrature nodes for integrating
  over latent marginals. Ignored for Normal+IdentityLink (analytic).

# Fields (after finalize!)
- `WAIC::Float64`: Watanabe-Akaike Information Criterion
- `p_WAIC::Float64`: Effective number of parameters (p_WAIC1)
- `lppd::Float64`: Log pointwise predictive density

# References
Watanabe (2010). "Asymptotic equivalence of Bayes cross validation and widely
applicable information criterion in singular learning theory."
Gelman, Hwang & Vehtari (2014). "Understanding predictive information criteria
for Bayesian models."
"""
mutable struct WAICAccumulator <: PosteriorAccumulator
    # Configuration
    n_nodes::Int

    # Accumulated data (unweighted) — one vector per integration point
    # integrated_lls[k][i] = log ∫ p(yᵢ|xᵢ) π(xᵢ|y,θₖ) dxᵢ  (for lppd)
    integrated_lls::Vector{Vector{Float64}}
    # expected_log_lls[k][i] = ∫ log p(yᵢ|xᵢ) π(xᵢ|y,θₖ) dxᵢ  (for p_WAIC)
    expected_log_lls::Vector{Vector{Float64}}

    # Results (computed in finalize!)
    lppd::Float64
    p_WAIC::Float64
    WAIC::Float64

    WAICAccumulator(; n_nodes::Int = 15) = new(
        n_nodes, Vector{Float64}[], Vector{Float64}[], 0.0, 0.0, 0.0
    )
end

"""
    _integrated_pointwise_loglik(ga, obs_lik; n_nodes=15)

Compute per-observation predictive quantities by integrating over the
latent field marginals from the Gaussian approximation.

Returns `(integrated_ll, expected_log_ll)` where:
- `integrated_ll[i]` = log ∫ p(yᵢ|xᵢ) π(xᵢ|y,θ) dxᵢ  (for lppd)
- `expected_log_ll[i]` = ∫ log p(yᵢ|xᵢ) π(xᵢ|y,θ) dxᵢ  (for p_WAIC1)

For Normal+IdentityLink observations, both are analytic. For all other
observation models, Gauss-Hermite quadrature is used.
"""
function _integrated_pointwise_loglik end

# Analytic for Normal + IdentityLink:
# integrated_ll:   log N(yᵢ; μᵢ, σ²_obs + vᵢ)
# expected_log_ll: E_x[log N(yᵢ; xᵢ, σ_obs)] = -½log(2π) - log(σ) - [(yᵢ-μᵢ)² + vᵢ]/(2σ²)
function _integrated_pointwise_loglik(
        ga, obs_lik::NormalLikelihood{IdentityLink}; n_nodes::Int = 15
    )
    μ = mean(ga)
    v = var(ga)
    σ_obs = obs_lik.σ
    y = obs_lik.y
    indices = obs_lik.indices

    n_obs = length(y)
    integrated_ll = Vector{Float64}(undef, n_obs)
    expected_log_ll = Vector{Float64}(undef, n_obs)
    for i in 1:n_obs
        idx = indices === nothing ? i : indices[i]
        total_var = σ_obs^2 + v[idx]
        integrated_ll[i] = logpdf(Normal(μ[idx], sqrt(total_var)), y[i])
        # E_x[log N(y; x, σ)] = -½log(2π) - log(σ) - E[(y-x)²]/(2σ²)
        # where E[(y-x)²] = (y-μ)² + v under x ~ N(μ, v)
        expected_log_ll[i] = -0.5 * log(2π) - log(σ_obs) -
            ((y[i] - μ[idx])^2 + v[idx]) / (2 * σ_obs^2)
    end
    return integrated_ll, expected_log_ll
end

# Generic: Gauss-Hermite quadrature using pointwise_loglik
function _integrated_pointwise_loglik(ga, obs_lik; n_nodes::Int = 15)
    μ = mean(ga)
    σ = std(ga)
    nodes, weights = gausshermite(n_nodes)
    scaled_weights = weights ./ sqrt(π)  # w_j / √π
    log_scaled_weights = log.(scaled_weights)

    # Evaluate pointwise_loglik at each quadrature node
    # x_j = μ + √2 · σ · ξ_j (element-wise perturbation)
    ll_at_nodes = Vector{Vector{Float64}}(undef, n_nodes)
    for j in 1:n_nodes
        x_j = μ .+ sqrt(2) .* σ .* nodes[j]
        ll_at_nodes[j] = pointwise_loglik(x_j, obs_lik)
    end

    n_obs = length(ll_at_nodes[1])
    integrated_ll = Vector{Float64}(undef, n_obs)
    expected_log_ll = Vector{Float64}(undef, n_obs)

    # For each observation: compute both quantities from the same nodes
    for i in 1:n_obs
        # integrated_ll: log ∫ p(y_i|x_i) π(x_i) dx_i = logsumexp(log(w_j/√π) + ll_j[i])
        log_terms = [log_scaled_weights[j] + ll_at_nodes[j][i] for j in 1:n_nodes]
        integrated_ll[i] = logsumexp(log_terms)

        # expected_log_ll: ∫ log p(y_i|x_i) π(x_i) dx_i = Σ_j (w_j/√π) * ll_j[i]
        expected_log_ll[i] = sum(scaled_weights[j] * ll_at_nodes[j][i] for j in 1:n_nodes)
    end

    return integrated_ll, expected_log_ll
end

"""Pre-computed summary data for one grid point (WAIC)."""
struct WAICPointSummary
    integrated_ll::Vector{Float64}
    expected_log_ll::Vector{Float64}
end

function compute_point_summary(acc::WAICAccumulator; ga, obs_lik, kwargs...)
    ill, ell = _integrated_pointwise_loglik(ga, obs_lik; n_nodes = acc.n_nodes)
    return WAICPointSummary(ill, ell)
end

function accumulate!(acc::WAICAccumulator, summary::WAICPointSummary; kwargs...)
    push!(acc.integrated_lls, summary.integrated_ll)
    push!(acc.expected_log_lls, summary.expected_log_ll)
    return nothing
end

function accumulate!(
        acc::WAICAccumulator;
        ga,
        obs_lik,
        kwargs...
    )
    integrated_ll, expected_log_ll = _integrated_pointwise_loglik(
        ga, obs_lik; n_nodes = acc.n_nodes
    )
    push!(acc.integrated_lls, integrated_ll)
    push!(acc.expected_log_lls, expected_log_ll)
    return nothing
end

function finalize!(acc::WAICAccumulator, exploration::AbstractHyperparameterExploration)
    weights = get_integration_weights(exploration)
    n_points = length(weights)
    n_obs = length(acc.integrated_lls[1])
    log_weights = log.(weights)

    # Compute lppd and p_WAIC per observation
    lppd_total = 0.0
    mean_ell_total = 0.0  # E_post[log p(y_i|x_i)]

    for i in 1:n_obs
        # lppd_i = log(∑_k w_k * exp(integrated_ll_k[i]))
        log_terms = [log_weights[k] + acc.integrated_lls[k][i] for k in 1:n_points]
        lppd_total += logsumexp(log_terms)

        # E_post[log p(y_i|x_i)] = ∑_k w_k * E_{x|θ_k}[log p(y_i|x_i)]
        mean_ell_total += sum(weights[k] * acc.expected_log_lls[k][i] for k in 1:n_points)
    end

    acc.lppd = lppd_total
    # p_WAIC1 = 2 * (lppd - E_post[log p(y|x)])
    # This captures BOTH latent and hyperparameter uncertainty via law of total variance
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
