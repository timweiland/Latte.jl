using Distributions
using Distributions: loggamma

"Compute log Σ_{n=1}^{n_max} term(n) with log-sum-exp stabilisation."
function _tweedie_log_W(y, μ, φ, p; n_max::Int = 150)
    α = (2 - p) / (p - 1)
    log_y = log(y)
    log_λ = (2 - p) * log(μ) - log(φ * (2 - p))
    log_β = log(φ * (p - 1)) + (p - 1) * log(μ)
    log_term(n) = n * log_λ - loggamma(n + 1) +
        n * α * (log_y - log_β) - loggamma(n * α)
    m = log_term(1)
    @inbounds for n in 2:n_max
        t = log_term(n)
        if t > m
            m = t
        end
    end
    s = zero(m)
    @inbounds for n in 1:n_max
        s += exp(log_term(n) - m)
    end
    return m + log(s)
end

"Tweedie compound Poisson-Gamma log-density at `y` with mean μ, dispersion φ, power p ∈ (1,2)."
function tweedie_logpdf(y, μ, φ, p)
    log_λ = (2 - p) * log(μ) - log(φ * (2 - p))
    if y == 0
        return -exp(log_λ)
    else
        log_β = log(φ * (p - 1)) + (p - 1) * log(μ)
        return -exp(log_λ) - y / exp(log_β) - log(y) +
            _tweedie_log_W(y, μ, φ, p)
    end
end

struct Tweedie{T <: Real} <: ContinuousUnivariateDistribution
    μ::T
    φ::T
    p::T
end
# Promoting constructor lets users mix Float and AD Dual arguments.
function Tweedie(μ::Real, φ::Real, p::Real)
    μp, φp, pp = promote(μ, φ, p)
    return Tweedie{typeof(μp)}(μp, φp, pp)
end
Distributions.logpdf(d::Tweedie, y::Real) = tweedie_logpdf(y, d.μ, d.φ, d.p)
Distributions.minimum(::Tweedie) = 0.0
Distributions.maximum(::Tweedie) = Inf
Distributions.insupport(::Tweedie, y::Real) = y >= 0

let μ = 5.0, φ = 1.0, p = 1.99, y = 4.0
    α = (2 - p) / (p - 1)
    β = φ * (p - 1) * μ^(p - 1)
    (tweedie_logpdf(y, μ, φ, p), logpdf(Gamma(α, β), y))
end

using LinearAlgebra
using Random
using DataFrames

function rand_tweedie(rng, μ, φ, p)
    λ = μ^(2 - p) / (φ * (2 - p))
    α = (2 - p) / (p - 1)
    β_scale = φ * (p - 1) * μ^(p - 1)
    n = rand(rng, Poisson(λ))
    n == 0 && return 0.0
    return rand(rng, Gamma(n * α, β_scale))
end

Random.seed!(42)
n = 200
true_β = [2.0, -0.4]
X = hcat(ones(n), randn(n))
true_μ = exp.(X * true_β)
true_φ = 1.5
true_p = 1.6
y = [rand_tweedie(Random.GLOBAL_RNG, μ_i, true_φ, true_p) for μ_i in true_μ]

df = DataFrame(x = X[:, 2], claim = y, has_claim = y .> 0)

using AlgebraOfGraphics, CairoMakie

fig = Figure(size = (820, 360))
ax1 = Axis(
    fig[1, 1], title = "Claim distribution",
    xlabel = "Claim amount", ylabel = "Density",
)
hist!(ax1, y; bins = 40, color = (:steelblue, 0.6), strokewidth = 0)
ax2 = Axis(
    fig[1, 2], title = "Claim vs covariate",
    xlabel = "Driver risk score (x)", ylabel = "Claim amount",
)
scatter!(ax2, df.x, df.claim; markersize = 7, color = (:black, 0.5))
fig

using Latte
using DynamicPPL: @model

@model function tweedie_glm(y, X, p_fixed)
    log_φ ~ Normal(0.0, 2.0)
    φ = exp(log_φ)
    β ~ MvNormal(zeros(size(X, 2)), 100.0 * I(size(X, 2)))
    for i in eachindex(y)
        μ_i = exp(dot(X[i, :], β))
        y[i] ~ Tweedie(μ_i, φ, p_fixed)
    end
end

lgm = latte_from_dppl(
    tweedie_glm(y, X, true_p);
    random = :β, likelihood_hessian_pattern = :dense,
)

result = inla(
    lgm, y;
    latent_marginalization_method = SimplifiedLaplace(),
    progress = false,
)

β_summary = summary_df(result.latent_marginals)

hp_summary = summary_df(result.hyperparameter_marginals)

truth = DataFrame(
    parameter = ["β₁ (intercept)", "β₂ (slope)", "log_φ"],
    truth = [true_β[1], true_β[2], log(true_φ)],
    posterior_mean = [β_summary.mean[1], β_summary.mean[2], hp_summary.mean[1]],
    q2_5 = [β_summary.q2_5[1], β_summary.q2_5[2], hp_summary.q2_5[1]],
    q97_5 = [β_summary.q97_5[1], β_summary.q97_5[2], hp_summary.q97_5[1]],
)

fig2 = Figure(size = (1100, 320))
for (j, (name, marginal, true_val)) in enumerate(
        [
            ("β₁ (intercept)", result.latent_marginals[1], true_β[1]),
            ("β₂ (slope)", result.latent_marginals[2], true_β[2]),
            ("log_φ", result.hyperparameter_marginals.log_φ, log(true_φ)),
        ]
    )
    ax = Axis(fig2[1, j]; title = name, xlabel = "value", ylabel = "density")
    xs = range(quantile(marginal, 0.001), quantile(marginal, 0.999); length = 200)
    lines!(ax, xs, pdf.(marginal, xs); color = :steelblue, linewidth = 2)
    vlines!(ax, [true_val]; color = :crimson, linestyle = :dash, linewidth = 2)
end
fig2

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
