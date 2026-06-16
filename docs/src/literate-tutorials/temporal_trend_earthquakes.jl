# # Temporal trend smoothing: Global earthquake activity
#
# Raw counts of rare events are noisy. Is the number of major earthquakes really
# changing over time, or are we just seeing random fluctuation? Here we use INLA
# with random walk models to smooth annual earthquake counts and look for a
# long-term trend.
#
# The tutorial covers a Poisson model with a temporal random effect, the
# difference between first- and second-order random walks (RW1 and RW2), the role
# of the random-walk precision in controlling smoothness, and model comparison via
# DIC and WAIC.
#
# ## The dataset
#
# We use annual counts of major earthquakes (magnitude $\geq$ 7) from 1900 to 2006.
# This dataset appears in [Zucchini et al. (2016)](#ref-zucchini) and the
# [INLA gitbook](https://becarioprecario.bitbucket.io/inla-gitbook/ch-temporal.html).
# With 107 observations it fits in a single code block:
using DataFrames
quake_counts = [
    13, 14, 8, 10, 16, 26, 32, 27, 18, 32, 36, 24, 20, 23, 23, 18, 12,
    20, 22, 19, 13, 26, 13, 14, 22, 24, 21, 22, 26, 21, 23, 24, 20, 24, 24, 22,
    20, 10, 14, 19, 23, 18, 12, 13, 20, 26, 35, 14, 17, 19, 15, 18, 22, 22, 17,
    22, 15, 34, 10, 15, 22, 18, 15, 20, 13, 22, 23, 15, 21, 19, 20, 11, 20, 13,
    10, 8, 15, 18, 15, 9, 13, 13, 14, 9, 13, 16, 15, 8, 5, 11, 13, 7, 15, 12, 23,
    25, 22, 21, 20, 16, 14, 15, 13, 14, 17, 14, 11,
]
eq_data = DataFrame(year = 1900:2006, quakes = quake_counts)
first(eq_data, 5)

# Let's take a first look:
using AlgebraOfGraphics, CairoMakie
draw(
    data(eq_data) *
        mapping(:year => "Year", :quakes => "Major earthquakes (M ≥ 7)") *
        visual(Scatter, markersize = 5, color = :gray40),
    axis = (title = "Annual counts of major earthquakes, 1900–2006",)
)

# The counts bounce around a lot year to year. There seems to be a broad hump in the
# early-to-mid 20th century and perhaps a decline toward the end, but by eye it is
# hard to separate signal from noise. Temporal smoothing is one way to make that
# separation explicit.
#
# ## Poisson model with a first-order random walk (RW1)
#
# We model the annual count $y_t$ as
#
# ```math
# y_t \sim \text{Poisson}(\lambda_t), \qquad \log \lambda_t = \beta_0 + f_t
# ```
#
# where $\beta_0$ is an intercept (overall log-rate) and $f_t$ is a latent temporal
# effect. For the RW1 model, the first differences of $f$ are i.i.d. Gaussian:
#
# ```math
# f_{t+1} - f_t \sim \mathcal{N}(0, \tau^{-1})
# ```
#
# This penalises abrupt jumps. The precision $\tau$ controls the smoothness:
# high $\tau$ means small differences (smooth trend), low $\tau$ allows more
# wiggle (follows the data closely).
#
# We express this as an `@latte` model. The random-walk prior comes from
# [GaussianMarkovRandomFields.jl](https://github.com/timweiland/GaussianMarkovRandomFields.jl)'s
# `RWModel{Order}`, which we call with `(τ = τ_rw)` to produce a GMRF prior that
# `@latte` recognizes as a structured Gaussian. We write one model per order:
# `RWModel{1}` here, `RWModel{2}` below.
using Latte
using Distributions
using GaussianMarkovRandomFields: RWModel
using LinearAlgebra

@latte function quake_rw1(y, n)
    τ_rw ~ PCPrior.Precision(1.0, α = 0.01)
    β ~ MvNormal(zeros(1), 100.0 * I(1))
    f ~ RWModel{1}(n)(τ = τ_rw)
    for i in eachindex(y)
        y[i] ~ Poisson(exp(β[1] + f[i]))
    end
end

# The PC prior `PCPrior.Precision(1.0, α = 0.01)` says "I believe there is only
# a 1% chance that the standard deviation of the first differences exceeds 1".
#
# Now we build the `LatentGaussianModel` (by calling the `@latte` function) and run INLA:
n_years = nrow(eq_data)
lgm_rw1 = quake_rw1(eq_data.quakes, n_years)
result_rw1 = inla(lgm_rw1, eq_data.quakes; progress = false)

# Let's look at the fitted trend. The observation marginals give the posterior
# distribution of $\lambda_t = \exp(\eta_t)$, the expected count per year:
obs_rw1 = observation_marginals(result_rw1)
fit_rw1 = summary_df(obs_rw1)
fit_rw1.year = eq_data.year
first(fit_rw1, 5)

# Overlaying the posterior median and 95% interval on the raw counts:
fig = Figure(size = (800, 400))
ax = Axis(
    fig[1, 1],
    xlabel = "Year", ylabel = "Major earthquakes",
    title = "RW1 smoothed trend"
)
scatter!(ax, eq_data.year, eq_data.quakes, color = :gray70, markersize = 5, label = "Observed")
band!(ax, fit_rw1.year, fit_rw1.q2_5, fit_rw1.q97_5, color = (:steelblue, 0.25), label = "95% CI")
lines!(ax, fit_rw1.year, fit_rw1.median, color = :steelblue, linewidth = 2, label = "Median")
axislegend(ax, position = :rt, framevisible = false)
fig

# The RW1 trend is fairly wiggly and tracks local fluctuations in the data. This
# is characteristic of a first-order random walk: it penalises abrupt jumps in
# level, but changing direction from year to year carries little cost.
#
# ## Poisson model with a second-order random walk (RW2)
#
# The RW2 model penalises second differences instead:
#
# ```math
# f_{t+2} - 2f_{t+1} + f_t \sim \mathcal{N}(0, \tau^{-1})
# ```
#
# This penalises changes in *slope* rather than changes in *level*, producing
# smoother, more slowly varying trends. The RW2 model is the same body with
# `RWModel{2}`:
@latte function quake_rw2(y, n)
    τ_rw ~ PCPrior.Precision(1.0, α = 0.01)
    β ~ MvNormal(zeros(1), 100.0 * I(1))
    f ~ RWModel{2}(n)(τ = τ_rw)
    for i in eachindex(y)
        y[i] ~ Poisson(exp(β[1] + f[i]))
    end
end

lgm_rw2 = quake_rw2(eq_data.quakes, n_years)
result_rw2 = inla(lgm_rw2, eq_data.quakes; progress = false)

# And the fitted trend:
obs_rw2 = observation_marginals(result_rw2)
fit_rw2 = summary_df(obs_rw2)
fit_rw2.year = eq_data.year
first(fit_rw2, 5)

#
fig = Figure(size = (800, 400))
ax = Axis(
    fig[1, 1],
    xlabel = "Year", ylabel = "Major earthquakes",
    title = "RW2 smoothed trend"
)
scatter!(ax, eq_data.year, eq_data.quakes, color = :gray70, markersize = 5, label = "Observed")
band!(ax, fit_rw2.year, fit_rw2.q2_5, fit_rw2.q97_5, color = (:orange, 0.25), label = "95% CI")
lines!(ax, fit_rw2.year, fit_rw2.median, color = :darkorange, linewidth = 2, label = "Median")
axislegend(ax, position = :rt, framevisible = false)
fig

# The RW2 trend is noticeably smoother. Instead of tracking year-to-year noise,
# it captures the broad shape: a rise from 1900 to the 1940s, a gradual plateau,
# and a decline from the 1970s onwards.
#
# ## Side-by-side comparison
#
# The contrast between RW1 and RW2 shows how the smoothness assumption feeds
# through to the fitted trend. Let's overlay both:
fig = Figure(size = (900, 450))
ax = Axis(
    fig[1, 1],
    xlabel = "Year", ylabel = "Major earthquakes",
    title = "RW1 vs RW2: the effect of smoothness assumptions"
)
scatter!(ax, eq_data.year, eq_data.quakes, color = :gray70, markersize = 5, label = "Observed")
lines!(ax, fit_rw1.year, fit_rw1.median, color = :steelblue, linewidth = 2, label = "RW1 (first-order)")
lines!(ax, fit_rw2.year, fit_rw2.median, color = :darkorange, linewidth = 2, label = "RW2 (second-order)")
axislegend(ax, position = :rt, framevisible = false)
fig

# The RW1 trend (blue) hugs the data more closely, while the RW2 trend (orange) is
# smoother and tells a simpler story about the underlying process. Neither is
# "right" in an absolute sense; the choice depends on whether you believe the true
# rate changes erratically or smoothly.
#
# ## Hyperparameter posteriors
#
# The precision $\tau$ controls the bias-variance tradeoff. Let's examine its
# posterior for each model:
fig = Figure(size = (900, 400))
ax1 = Axis(
    fig[1, 1],
    xlabel = "Precision (τ)", ylabel = "Density",
    title = "RW1 precision posterior"
)
ax2 = Axis(
    fig[1, 2],
    xlabel = "Precision (τ)", ylabel = "Density",
    title = "RW2 precision posterior"
)
plot!(ax1, hyperparameter_marginals(result_rw1, :τ_rw)[1])
plot!(ax2, hyperparameter_marginals(result_rw2, :τ_rw)[1])
fig

# Higher precision means smaller differences between consecutive time points,
# which produces a smoother curve. The two posteriors sit in different places:
# the data inform how much smoothing is appropriate within each model class.
#
# The posterior mean of $\tau$ is read directly off the marginal, since
# `@latte` reports hyperparameters on their declared (natural) scale:
τ_rw1 = hyperparameter_marginals(result_rw1, :τ_rw)[1]
τ_rw2 = hyperparameter_marginals(result_rw2, :τ_rw)[1]
mean(τ_rw1), mean(τ_rw2)

# A compact summary table for each model's precision:
summary_df(hyperparameter_marginals(result_rw1))

#
summary_df(hyperparameter_marginals(result_rw2))

# ## Model comparison
#
# Which model fits the data better? INLA computes several model comparison criteria
# as part of inference, and they land in `result.accumulators`. We pull out three.
# The Deviance Information Criterion (DIC) and the Watanabe-Akaike Information
# Criterion (WAIC) both balance fit against complexity, with lower values
# preferred; the log marginal likelihood is a model-selection score where higher
# is preferred. The default accumulator tuple orders them as DIC, log marginal
# likelihood, then WAIC.
comparison = DataFrame(
    model = String[], DIC = Float64[], p_D = Float64[],
    WAIC = Float64[], log_ML = Float64[],
)
for (name, res) in [("RW1", result_rw1), ("RW2", result_rw2)]
    push!(
        comparison, (
            name,
            round(res.accumulators[1].DIC, digits = 1),
            round(res.accumulators[1].p_D, digits = 1),
            round(res.accumulators[3].WAIC, digits = 1),
            round(res.accumulators[2].log_marginal_likelihood, digits = 1),
        )
    )
end
comparison

# DIC and WAIC weigh fit against complexity. The effective number of parameters
# ($p_D$) is lower for RW2, reflecting its smoother fit, so the criteria let us ask
# whether the extra flexibility of RW1 earns its keep on this data.
#
# ## Posterior predictive check
#
# Finally, do the models reproduce the variability we see in the data? We draw
# posterior predictive datasets with `posterior_predictive`, which returns an
# `n_samples × n_obs` matrix where each row is one simulated dataset:
using Random
Random.seed!(42)

n_obs = nrow(eq_data)
n_samples = 200
pp_rw1 = posterior_predictive(result_rw1, n_samples)
pp_rw2 = posterior_predictive(result_rw2, n_samples)
size(pp_rw1), size(pp_rw2)

# For each year, take the 2.5th and 97.5th percentiles over the replicated counts
# (columns index years here):
pp_rw1_lo = [quantile(pp_rw1[:, t], 0.025) for t in 1:n_obs]
pp_rw1_hi = [quantile(pp_rw1[:, t], 0.975) for t in 1:n_obs]
pp_rw2_lo = [quantile(pp_rw2[:, t], 0.025) for t in 1:n_obs]
pp_rw2_hi = [quantile(pp_rw2[:, t], 0.975) for t in 1:n_obs];

fig = Figure(size = (900, 500))
ax1 = Axis(
    fig[1, 1], xlabel = "Year", ylabel = "Count",
    title = "Posterior predictive: RW1"
)
ax2 = Axis(
    fig[1, 2], xlabel = "Year", ylabel = "Count",
    title = "Posterior predictive: RW2"
)
band!(ax1, eq_data.year, pp_rw1_lo, pp_rw1_hi, color = (:steelblue, 0.2))
scatter!(ax1, eq_data.year, eq_data.quakes, color = :gray40, markersize = 4)
band!(ax2, eq_data.year, pp_rw2_lo, pp_rw2_hi, color = (:orange, 0.2))
scatter!(ax2, eq_data.year, eq_data.quakes, color = :gray40, markersize = 4)
fig

# The shaded bands show where 95% of replicated data would fall. If the observed
# counts sit comfortably inside the bands, the model is capturing the data's
# variability well. Points consistently outside the bands would suggest model
# misspecification.
#
# ## Summary
#
# We used INLA to smooth earthquake counts over a century of data, with a few
# points worth carrying forward:
#
# - RW1 penalises first differences and gives a locally adaptive trend that can
#   change direction easily, which suits a process you expect to move irregularly.
# - RW2 penalises second differences for a globally smooth trend, a better match
#   when the underlying rate varies slowly.
# - Within each model class the precision $\tau$ sets the degree of smoothing, and
#   INLA estimates it from the data rather than leaving it for you to pick.
# - DIC, WAIC, and the marginal likelihood give principled ways to compare models
#   with different smoothness assumptions.
#
# Random walks are basic building blocks in INLA ([Rue & Held, 2005](#ref-gmrf)).
# From RW1 and RW2 you can build
# toward AR1 processes, seasonal models, and the separable space-time models
# covered in the spatial disease mapping tutorial.
#
# ## References
#
# ```@raw html
# <div class="ref-grid-2">
# <PaperCite
#   tag="Zucchini"
#   title="Hidden Markov Models for Time Series: An Introduction Using R (2nd ed.)"
#   authors="W. Zucchini, I. L. MacDonald & R. Langrock"
#   venue="Chapman & Hall/CRC" year="2016"
#   doi="10.1201/b20790"
#   url="https://doi.org/10.1201/b20790"
#   abstract="Source of the annual major-earthquake counts (1900–2006) used here as the competing-trend example." />
# <PaperCite
#   tag="GMRF"
#   title="Gaussian Markov Random Fields: Theory and Applications"
#   authors="H. Rue & L. Held"
#   venue="Chapman & Hall/CRC" year="2005"
#   doi="10.1201/9780203492024"
#   url="https://doi.org/10.1201/9780203492024"
#   abstract="The reference on GMRFs, including the random-walk priors (RW1/RW2) used here as temporal smoothers for the latent trend." />
# </div>
# ```
