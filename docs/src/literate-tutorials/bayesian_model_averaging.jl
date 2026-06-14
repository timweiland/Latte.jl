# # Bayesian Model Averaging
#
# When comparing multiple candidate models, a natural question arises: why pick just one?
# **Bayesian Model Averaging (BMA)** combines posterior marginals from multiple models,
# weighted by how well each model explains the data. This accounts for model uncertainty
# in your final inference.
#
# The idea is simple. Given $K$ models $M_1, \ldots, M_K$, the averaged posterior for
# any quantity of interest is:
#
# ```math
# p(\Delta \mid y) = \sum_{k=1}^{K} p(\Delta \mid y, M_k) \cdot p(M_k \mid y)
# ```
#
# where the model weights are $p(M_k \mid y) \propto p(y \mid M_k) \cdot p(M_k)$,
# and $p(y \mid M_k)$ is the marginal likelihood that INLA already computes.
#
# ## Setup: Poisson regression
#
# Let's simulate some count data from a Poisson regression with two covariates:
using Random
Random.seed!(42)

using Distributions
n = 40
x1 = randn(n)
x2 = randn(n)
β_true = [1.0, 0.6, -0.3]
η_true = β_true[1] .+ β_true[2] .* x1 .+ β_true[3] .* x2
y = rand.(Poisson.(exp.(η_true)))

# ## Fitting competing models
#
# We'll fit three Poisson regressions that differ in which covariates they include.
# Each is a plain DynamicPPL `@model`; the coefficients share an IID Gaussian
# prior with an `Exponential(1)` prior on the precision. The adapter turns each
# one into a `LatentGaussianModel` and hands it to `inla`.
using Latte
using LinearAlgebra

@latte function m1_dppl(y, x1)
    τ ~ Exponential(1.0)
    β ~ MvNormal(zeros(2), (1 / τ) * I(2))
    for i in eachindex(y)
        y[i] ~ Poisson(exp(β[1] + β[2] * x1[i]))
    end
end

@latte function m2_dppl(y, x2)
    τ ~ Exponential(1.0)
    β ~ MvNormal(zeros(2), (1 / τ) * I(2))
    for i in eachindex(y)
        y[i] ~ Poisson(exp(β[1] + β[2] * x2[i]))
    end
end

@latte function m3_dppl(y, x1, x2)
    τ ~ Exponential(1.0)
    β ~ MvNormal(zeros(3), (1 / τ) * I(3))
    for i in eachindex(y)
        y[i] ~ Poisson(exp(β[1] + β[2] * x1[i] + β[3] * x2[i]))
    end
end

# **Model 1**: Intercept + x1 only
lgm1 = m1_dppl(y, x1)
r1 = inla(lgm1, y; progress = false)

# **Model 2**: Intercept + x2 only
lgm2 = m2_dppl(y, x2)
r2 = inla(lgm2, y; progress = false)

# **Model 3**: Intercept + x1 + x2 (the true model)
lgm3 = m3_dppl(y, x1, x2)
r3 = inla(lgm3, y; progress = false)

# ## Comparing marginal likelihoods
#
# Before averaging, let's see how the models compare:
log_mlls = [
    r1.exploration.log_normalization_constant,
    r2.exploration.log_normalization_constant,
    r3.exploration.log_normalization_constant,
]
println("Log marginal likelihoods:")
for (i, ll) in enumerate(log_mlls)
    println("  Model $i: $(round(ll, digits = 2))")
end

# ## Model averaging
#
# We can only average models with the same latent dimension, so we average
# Model 1 and Model 2 (both have 2 base latent variables) to see how BMA
# handles the covariate selection question:
bma_12 = model_average([r1, r2])

println("BMA weights (Model 1 vs Model 2):")
println("  Model 1 (x1): $(round(bma_12.model_weights[1], digits = 4))")
println("  Model 2 (x2): $(round(bma_12.model_weights[2], digits = 4))")

# Since the true data-generating process uses x1 with a larger coefficient (0.6)
# than x2 (-0.3), we'd expect Model 1 to get more weight.
#
# ## Visualizing the averaged marginals
#
# Let's compare the intercept marginals from each model and the BMA result.
# We'll write a small helper to plot a distribution's PDF:
using CairoMakie

function plot_dist!(ax, d; n_points = 200, kwargs...)
    μ, σ = mean(d), std(d)
    xs = range(μ - 4σ, μ + 4σ, length = n_points)
    ys = pdf.(d, xs)
    return lines!(ax, xs, ys; kwargs...)
end

fig = Figure(size = (800, 400))

## Intercept comparison
ax1 = Axis(fig[1, 1]; title = "Intercept (β₁)", xlabel = "value", ylabel = "density")
plot_dist!(ax1, base_latent_marginals(r1)[1]; color = :blue, label = "Model 1 (x1)")
plot_dist!(ax1, base_latent_marginals(r2)[1]; color = :red, label = "Model 2 (x2)")
plot_dist!(ax1, bma_12.latent_marginals[1]; color = :black, linewidth = 3, label = "BMA")
vlines!(ax1, β_true[1]; color = :green, linestyle = :dash, label = "truth")
axislegend(ax1; position = :lt, framevisible = false)

## Slope comparison (β₂ from each model — different covariates!)
ax2 = Axis(fig[1, 2]; title = "Slope (β₂)", xlabel = "value", ylabel = "density")
plot_dist!(ax2, base_latent_marginals(r1)[2]; color = :blue, label = "Model 1 (coeff of x1)")
plot_dist!(ax2, base_latent_marginals(r2)[2]; color = :red, label = "Model 2 (coeff of x2)")
plot_dist!(ax2, bma_12.latent_marginals[2]; color = :black, linewidth = 3, label = "BMA")
axislegend(ax2; position = :lt, framevisible = false)

fig

# The BMA marginal (black) sits between the individual models, weighted toward the
# model with higher marginal likelihood. The credible intervals from BMA are typically
# wider than those from any single model, reflecting the additional model uncertainty.
#
# ## Using prior model weights
#
# By default, `model_average` uses equal prior weights. If you have domain knowledge
# suggesting one model is more plausible a priori, you can specify prior weights:
bma_prior = model_average([r1, r2]; prior_weights = [0.8, 0.2])
println("BMA with prior preference for Model 1:")
println("  Model 1: $(round(bma_prior.model_weights[1], digits = 4))")
println("  Model 2: $(round(bma_prior.model_weights[2], digits = 4))")

# ## Working with BMA results
#
# The `BMAResult` contains standard `WeightedMixture` marginals, so the full
# Distributions.jl interface works:
intercept_bma = bma_12.latent_marginals[1]
println(
    "BMA intercept: mean = $(round(mean(intercept_bma), digits = 3)), ",
    "95% CI = [$(round(quantile(intercept_bma, 0.025), digits = 3)), ",
    "$(round(quantile(intercept_bma, 0.975), digits = 3))]"
)

# You can also get a summary table:
summary_df(bma_12.latent_marginals)
