# # Spatial Modelling with the SPDE Approach
#
# The **SPDE approach** (Lindgren, Rue & Lindström, 2011) is one of the most powerful
# features of INLA. It lets us model continuously indexed spatial random fields using
# Gaussian Markov random fields on a triangular mesh — combining the flexibility of
# geostatistical models with the computational efficiency of sparse precision matrices.
#
# In this tutorial we will:
# 1. Simulate noisy observations from a Matérn spatial field
# 2. Fit an SPDE model to recover the underlying field
# 3. Visualise the posterior mean and uncertainty on a prediction grid

# ## Simulating spatial data
#
# We place 120 sensor locations across a `[0, 10]²` domain using a Sobol sequence
# for uniform coverage. The true spatial field follows a Matérn covariance with
# practical range 3 and marginal std dev 2, plus a constant mean of 1.5.

using Random, LinearAlgebra
Random.seed!(1234)

using Sobol
sob = SobolSeq(2)
skip(sob, 120)  # skip initial points for better uniformity
coords = reduce(vcat, [next!(sob)' for _ in 1:120]) .* 10

n_obs = size(coords, 1)

# True parameters
intercept_true = 1.5
range_true = 3.0
σ_field = 2.0
σ_noise = 0.5

# We simulate from the analytic Matérn covariance to generate "ground truth":
using SpecialFunctions: besselk, gamma

function matern_cov(d; ν = 1, κ = sqrt(8ν) / range_true, σ² = σ_field^2)
    d ≈ 0 && return σ²
    x = κ * d
    return σ² * 2^(1 - ν) / gamma(ν) * x^ν * besselk(ν, x)
end

dists = [norm(coords[i, :] - coords[j, :]) for i in 1:n_obs, j in 1:n_obs]
Σ_true = [matern_cov(dists[i, j]) for i in 1:n_obs, j in 1:n_obs]
Σ_true += 1.0e-6 * I

field_true = cholesky(Symmetric(Σ_true)).L * randn(n_obs)
y = intercept_true .+ field_true .+ σ_noise .* randn(n_obs)

using DataFrames
df = DataFrame(x = coords[:, 1], y_coord = coords[:, 2], y = y, field = field_true)
first(df, 5)

# ## Visualising the observations
#
# The left panel shows the true latent field (what we want to recover), the right
# panel shows the noisy observations (what we actually measure).
using AlgebraOfGraphics, CairoMakie

fig = Figure(size = (800, 350))
ax1 = Axis(fig[1, 1]; title = "True spatial field", xlabel = "x", ylabel = "y", aspect = DataAspect())
sc1 = scatter!(ax1, df.x, df.y_coord; color = df.field, colormap = :RdYlBu, markersize = 8)
Colorbar(fig[1, 2], sc1; label = "u(s)")

ax2 = Axis(fig[1, 3]; title = "Noisy observations", xlabel = "x", ylabel = "y", aspect = DataAspect())
sc2 = scatter!(ax2, df.x, df.y_coord; color = df.y, colormap = :RdYlBu, markersize = 8)
Colorbar(fig[1, 4], sc2; label = "y")
fig

# ## Setting up the SPDE model
#
# The model is:
#
# ``y_i = \alpha + u(s_i) + \varepsilon_i, \qquad \varepsilon_i \sim N(0, \sigma^2)``
#
# where ``u(s)`` is a Matérn spatial field and ``\alpha`` is a global intercept.
#
# With the formula interface, we just specify which columns hold the spatial
# coordinates. The triangular mesh, FEM discretisation, and projector matrix
# are all constructed automatically from the observation locations.

using GaussianMarkovRandomFields, StatsModels
using IntegratedNestedLaplace
using Distributions

f = @formula(y ~ 1 + Matern()(x, y_coord))

# The formula interface renames hyperparameters to avoid collisions when combining
# multiple latent components. For the Matérn model, the field precision becomes
# `τ_matern` and the range parameter becomes `range_matern`.
hp = @hyperparams begin
    (σ ~ PCPrior.Sigma(5.0, α = 0.01), transform = log)
    (τ_matern ~ PCPrior.Precision(1.0, α = 0.01), transform = log)
    (range_matern ~ Exponential(5.0), transform = log, space = natural)
end

# ## Running inference

result = inla(f, hp, df; family = Normal, progress = false)

# ## Hyperparameter posteriors
#
# Let's check how well we recovered the true parameters.
hp_df = summary_df(result.hyperparameter_marginals)
hp_df

#

# The true field precision is τ = 1/σ_field²:
τ_true = 1 / σ_field^2
println("True values: σ = $σ_noise, τ = $τ_true, range = $range_true")

# The posterior distributions show the uncertainty in these estimates:

fig = Figure(size = (1000, 300))
for (i, (name, truth, label)) in enumerate(
        zip(
            [:σ, :τ_matern, :range_matern],
            [σ_noise, τ_true, range_true],
            ["Noise σ", "Field precision τ", "Spatial range"]
        )
    )
    ax = Axis(fig[1, i]; title = label, xlabel = "value", ylabel = "density")
    d = result.hyperparameter_marginals[name]
    μ, s = mean(d), std(d)
    xs = range(max(1.0e-3, μ - 4s), μ + 4s; length = 200)
    lines!(ax, xs, pdf.(d, xs); color = :steelblue, linewidth = 2, label = "posterior")
    vlines!(ax, [truth]; color = :red, linestyle = :dash, linewidth = 2, label = "truth")
    axislegend(ax; position = :rt, framevisible = false)
end
fig

# ## Intercept posterior
#
# Since we simulated with a non-zero mean (α = 1.5), the intercept should
# be recovered from the data. The intercept is the last base latent marginal
# (the fixed effect component of the CombinedModel).
base_model = result.model.latent_prior.base_model
n_mesh = length(base_model.matern)
intercept_marginal = result.base_latent_marginals[n_mesh + 1]

fig = Figure(size = (400, 300))
ax = Axis(fig[1, 1]; title = "Intercept α", xlabel = "value", ylabel = "density")
μ_int, s_int = mean(intercept_marginal), std(intercept_marginal)
xs = range(μ_int - 4s_int, μ_int + 4s_int; length = 200)
lines!(ax, xs, pdf.(intercept_marginal, xs); color = :steelblue, linewidth = 2, label = "posterior")
vlines!(ax, [intercept_true]; color = :red, linestyle = :dash, linewidth = 2, label = "truth (α = $intercept_true)")
axislegend(ax; position = :rt, framevisible = false)
fig

# ## Posterior spatial field
#
# Now we project the estimated field onto a regular grid for visualisation.
# The `predict` function handles all the details: it reuses the trained SPDE
# mesh, builds the FEM projector, includes the intercept, and delegates to
# `linear_combinations` — which gives proper posterior marginals that account
# for hyperparameter uncertainty.

n_grid = 50
xs_grid = range(0, 10; length = n_grid)
ys_grid = range(0, 10; length = n_grid)

pred_df = DataFrame(
    x = vec([x for x in xs_grid, _ in ys_grid]),
    y_coord = vec([y for _ in xs_grid, y in ys_grid])
)
pred_marginals = predict(result, pred_df)

field_mean = [mean(m) for m in pred_marginals]
field_sd = [std(m) for m in pred_marginals]

field_mean_grid = reshape(field_mean, n_grid, n_grid)
field_sd_grid = reshape(field_sd, n_grid, n_grid)

fig = Figure(size = (900, 350))
ax1 = Axis(fig[1, 1]; title = "Posterior mean u(s)", xlabel = "x", ylabel = "y", aspect = DataAspect())
hm1 = heatmap!(ax1, xs_grid, ys_grid, field_mean_grid; colormap = :RdYlBu)
scatter!(ax1, df.x, df.y_coord; color = :black, markersize = 3)
Colorbar(fig[1, 2], hm1)

ax2 = Axis(fig[1, 3]; title = "Posterior std. dev.", xlabel = "x", ylabel = "y", aspect = DataAspect())
hm2 = heatmap!(ax2, xs_grid, ys_grid, field_sd_grid; colormap = :YlOrRd)
scatter!(ax2, df.x, df.y_coord; color = :black, markersize = 3)
Colorbar(fig[1, 4], hm2)
fig

# Notice how the uncertainty (right panel) is lowest near observation locations
# and increases towards the edges of the domain — exactly what we expect from
# a spatial interpolation model.

# ## Model diagnostics

println("Model fit:")
println("  DIC:  $(round(result.accumulators[1].DIC, digits = 1))")
println("  WAIC: $(round(result.accumulators[3].WAIC, digits = 1))")
println("  Log marginal likelihood: $(round(result.exploration.log_normalization_constant, digits = 1))")

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
