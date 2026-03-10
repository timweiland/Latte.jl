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
# We'll fit three models that differ in which covariates they include.
# All share the same IID latent structure and Poisson likelihood.
using IntegratedNestedLaplace
using GaussianMarkovRandomFields

base_obs = ExponentialFamily(Poisson)

# **Model 1**: Intercept + x1 only
A1 = hcat(ones(n), x1)
obs1 = LinearlyTransformedObservationModel(base_obs, A1)
hp1 = @hyperparams begin
    (τ ~ Exponential(1.0), transform = log, space = natural)
end
m1 = INLAModel(hp1, IIDModel(2), obs1)
r1 = inla(m1, y; progress = false)

# **Model 2**: Intercept + x2 only
A2 = hcat(ones(n), x2)
obs2 = LinearlyTransformedObservationModel(base_obs, A2)
hp2 = @hyperparams begin
    (τ ~ Exponential(1.0), transform = log, space = natural)
end
m2 = INLAModel(hp2, IIDModel(2), obs2)
r2 = inla(m2, y; progress = false)

# **Model 3**: Intercept + x1 + x2 (the true model)
A3 = hcat(ones(n), x1, x2)
obs3 = LinearlyTransformedObservationModel(base_obs, A3)
hp3 = @hyperparams begin
    (τ ~ Exponential(1.0), transform = log, space = natural)
end
m3 = INLAModel(hp3, IIDModel(3), obs3)
r3 = inla(m3, y; progress = false)

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

n_lp = length(r1.linear_predictor_marginals)

fig = Figure(size = (800, 400))

## Intercept comparison
ax1 = Axis(fig[1, 1]; title = "Intercept (β₁)", xlabel = "value", ylabel = "density")
plot_dist!(ax1, r1.base_latent_marginals[1]; color = :blue, label = "Model 1 (x1)")
plot_dist!(ax1, r2.base_latent_marginals[1]; color = :red, label = "Model 2 (x2)")
plot_dist!(ax1, bma_12.latent_marginals[n_lp + 1]; color = :black, linewidth = 3, label = "BMA")
vlines!(ax1, β_true[1]; color = :green, linestyle = :dash, label = "truth")
axislegend(ax1; position = :lt, framevisible = false)

## Slope comparison (β₂ from each model — different covariates!)
ax2 = Axis(fig[1, 2]; title = "Slope (β₂)", xlabel = "value", ylabel = "density")
plot_dist!(ax2, r1.base_latent_marginals[2]; color = :blue, label = "Model 1 (coeff of x1)")
plot_dist!(ax2, r2.base_latent_marginals[2]; color = :red, label = "Model 2 (coeff of x2)")
plot_dist!(ax2, bma_12.latent_marginals[n_lp + 2]; color = :black, linewidth = 3, label = "BMA")
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
intercept_bma = bma_12.latent_marginals[n_lp + 1]
println(
    "BMA intercept: mean = $(round(mean(intercept_bma), digits = 3)), ",
    "95% CI = [$(round(quantile(intercept_bma, 0.025), digits = 3)), ",
    "$(round(quantile(intercept_bma, 0.975), digits = 3))]"
)

# You can also get a summary table:
summary_df(bma_12.latent_marginals)
