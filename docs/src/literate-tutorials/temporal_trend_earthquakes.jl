# # Temporal trend smoothing: Global earthquake activity
#
# Raw counts of rare events are noisy. Is the number of major earthquakes really
# changing over time, or are we just seeing random fluctuation? In this tutorial
# we use INLA with random walk models to smooth annual earthquake counts and
# uncover long-term trends.
#
# Along the way you will learn:
# - How to fit **Poisson models** with temporal random effects
# - The difference between **RW1** (first-order) and **RW2** (second-order) random walks
# - How the random walk **precision** controls the smoothness of the estimated trend
# - How to compare models using **DIC** and **WAIC**
#
# ## The dataset
#
# We use annual counts of major earthquakes (magnitude $\geq$ 7) from 1900 to 2006.
# This classic dataset appears in Zucchini et al. (2016) and the
# [INLA gitbook](https://becarioprecario.bitbucket.io/inla-gitbook/ch-temporal.html).
# With only 107 observations it fits comfortably in a code block:
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
# early-to-mid 20th century and perhaps a decline toward the end, but it is hard to
# tell signal from noise by eye. This is exactly the kind of problem where temporal
# smoothing shines.
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
# We express this as an `@latte` model. The random-walk prior is provided by
# [GaussianMarkovRandomFields.jl](https://github.com/timweiland/GaussianMarkovRandomFields.jl)'s
# `RWModel{Order}`, which we call with `(τ = τ_rw)` to produce a GMRF prior that
# `@latte` recognizes as a structured Gaussian. We write one model per order —
# `RWModel{1}` here, `RWModel{2}` below.
using Latte
using DynamicPPL
using Distributions
using GaussianMarkovRandomFields: RWModel
using LinearAlgebra

@latte function quake_rw1(y, n)
    τ_rw ~ PCPrior.Precision(1.0, α = 0.01)
    β ~ MvNormal(zeros(1), 100.0 * I(1))
    f ~ RWModel{1}(n)(τ = τ_rw)
    for i in eachindex(y)
        y[i] ~ Poisson(exp(β[1] + f[i]); check_args = false)
    end
end

# The PC prior `PCPrior.Precision(1.0, α = 0.01)` says "I believe there is only
# a 1% chance that the standard deviation of the first differences exceeds 1".
#
# Now we build the `LatentGaussianModel` (by calling the `@latte` function) and run INLA:
n_years = nrow(eq_data)
lgm_rw1 = quake_rw1(eq_data.quakes, n_years)
result_rw1 = inla(lgm_rw1, eq_data.quakes; progress = false)

# Let's look at the fitted trend. The observation marginals give us the posterior
# distribution of $\lambda_t = \exp(\eta_t)$, the expected count per year:
obs_rw1 = observation_marginals(result_rw1)
fit_rw1 = summary_df(obs_rw1)
fit_rw1.year = eq_data.year

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

# The RW1 trend is fairly wiggly — it tracks local fluctuations in the data.
# This is characteristic of a first-order random walk: it penalises abrupt jumps
# but is happy to change direction frequently.
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
        y[i] ~ Poisson(exp(β[1] + f[i]); check_args = false)
    end
end

lgm_rw2 = quake_rw2(eq_data.quakes, n_years)
result_rw2 = inla(lgm_rw2, eq_data.quakes; progress = false)

# And the fitted trend:
obs_rw2 = observation_marginals(result_rw2)
fit_rw2 = summary_df(obs_rw2)
fit_rw2.year = eq_data.year

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
# The contrast between RW1 and RW2 is a vivid illustration of how model choice
# affects inference. Let's overlay both:
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

# The RW1 trend (blue) hugs the data more closely. The RW2 trend (orange) is
# smoother and more conservative — it tells a simpler story about the underlying
# process. Neither is "right" in an absolute sense; the choice depends on whether
# you believe the true rate changes erratically or smoothly.
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
plot!(ax1, result_rw1.hyperparameter_marginals.τ_rw)
plot!(ax2, result_rw2.hyperparameter_marginals.τ_rw)
fig

# Higher precision means smaller differences between consecutive time points,
# which produces a smoother curve. Notice how the posteriors differ: the data
# inform how much smoothing is appropriate for each model class.
#
# Let's look at the summary statistics:
println("RW1 precision:")
println(summary_df(result_rw1.hyperparameter_marginals))
println("\nRW2 precision:")
println(summary_df(result_rw2.hyperparameter_marginals))

# ## Model comparison
#
# Which model fits the data better? INLA computes several model comparison criteria
# as part of inference. Let's look at three:
# - **DIC** (Deviance Information Criterion): lower is better
# - **WAIC** (Watanabe-Akaike Information Criterion): lower is better
# - **Log marginal likelihood**: higher is better
println("Model comparison:")
println("─"^50)
for (name, res) in [("RW1", result_rw1), ("RW2", result_rw2)]
    dic = res.accumulators[1].DIC
    p_d = res.accumulators[1].p_D
    mll = res.accumulators[2].log_marginal_likelihood
    waic = res.accumulators[3].WAIC
    println(
        "$name: DIC = $(round(dic, digits = 1)) (p_D = $(round(p_d, digits = 1))), " *
            "WAIC = $(round(waic, digits = 1)), " *
            "log ML = $(round(mll, digits = 1))"
    )
end

# The DIC and WAIC tell us which model explains the data better while accounting
# for complexity. The effective number of parameters ($p_D$) is lower for RW2,
# reflecting its smoother fit. Together, these criteria help decide whether the
# extra flexibility of RW1 is justified by the data.
#
# ## Posterior predictive check
#
# Finally, let's check whether our models can reproduce the variability we see in
# the data. We draw posterior predictive samples and compare their distribution
# to the observed counts:
using Random
Random.seed!(42)

n_obs = nrow(eq_data)
n_samples = 200
pp_rw1 = hcat([rand(result_rw1, 1; include_y = true)[1].y[1:n_obs] for _ in 1:n_samples]...)
pp_rw2 = hcat([rand(result_rw2, 1; include_y = true)[1].y[1:n_obs] for _ in 1:n_samples]...)
size(pp_rw1), size(pp_rw2)

# For each year, compute the 2.5th and 97.5th percentiles of the replicated counts:
pp_rw1_lo = [quantile(pp_rw1[t, :], 0.025) for t in 1:n_obs]
pp_rw1_hi = [quantile(pp_rw1[t, :], 0.975) for t in 1:n_obs]
pp_rw2_lo = [quantile(pp_rw2[t, :], 0.025) for t in 1:n_obs]
pp_rw2_hi = [quantile(pp_rw2[t, :], 0.975) for t in 1:n_obs]

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
# In this tutorial we used INLA to smooth earthquake counts over a century of data.
# The key takeaways:
#
# - **RW1** penalises first differences, producing a *locally adaptive* trend that can
#   change direction easily. It is a good default when you expect irregular changes.
# - **RW2** penalises second differences, producing a *globally smooth* trend. It is
#   a good choice when you believe the underlying process varies slowly.
# - The **precision hyperparameter** $\tau$ controls the smoothness within each model
#   class. INLA estimates $\tau$ from the data rather than requiring you to choose it.
# - **DIC**, **WAIC**, and the **marginal likelihood** provide principled ways to
#   compare models with different smoothness assumptions.
#
# Random walks are fundamental building blocks in INLA. Once you are comfortable with
# RW1 and RW2, you can build toward AR1 processes, seasonal models, and the separable
# space-time models covered in the spatial disease mapping tutorial.
#
# ## References
#
# - Zucchini, W., MacDonald, I. L., & Langrock, R. (2016). *Hidden Markov Models for Time Series*. Chapman & Hall/CRC.
# - Rue, H. & Held, L. (2005). *Gaussian Markov Random Fields: Theory and Applications*. Chapman & Hall/CRC.
