# # Nonlinear regression: Bayesian smoothing with INLA
#
# When the relationship between a predictor and a response is nonlinear and of
# unknown form, we need a flexible model that can adapt to the data. Generalized
# additive models (GAMs) are a popular frequentist approach, but they do not
# provide full posterior uncertainty on the smoothness of the curve. In this
# tutorial we use INLA with a second-order random walk (RW2) to fit a nonlinear
# curve with proper Bayesian uncertainty quantification — a fully Bayesian
# alternative to GAMs.
#
# Along the way you will learn:
# - How to fit a **Gaussian** observation model (the first tutorial to use one!)
# - How an **RW2 prior** acts as a nonparametric smooth, analogous to a cubic
#   smoothing spline
# - How two hyperparameters — **observation noise** and **RW2 precision** —
#   jointly control the bias-variance tradeoff
# - How to compare a nonparametric model against a simple linear baseline
#
# ## The dataset
#
# We use the classic `mcycle` dataset (Silverman, 1985): 133 accelerometer
# readings from a simulated motorcycle crash test. The response is head
# acceleration (in g) measured at various times (in milliseconds) after impact.
# This dataset appears in every GAM textbook and is arguably the most iconic
# example in nonparametric regression.
using DataFrames
times = [
    2.4, 2.6, 3.2, 3.6, 4.0, 6.2, 6.6, 6.8, 7.8, 8.2, 8.8, 8.8, 9.6,
    10.0, 10.2, 10.6, 11.0, 11.4, 13.2, 13.6, 13.8, 14.6, 14.6, 14.6,
    14.6, 14.6, 14.6, 14.8, 15.4, 15.4, 15.4, 15.4, 15.6, 15.6, 15.8,
    15.8, 16.0, 16.0, 16.2, 16.2, 16.2, 16.4, 16.4, 16.6, 16.8, 16.8,
    16.8, 17.6, 17.6, 17.6, 17.6, 17.8, 17.8, 18.6, 18.6, 19.2, 19.4,
    19.4, 19.6, 20.2, 20.4, 21.2, 21.4, 21.8, 22.0, 23.2, 23.4, 24.0,
    24.2, 24.2, 24.6, 25.0, 25.0, 25.4, 25.4, 25.6, 26.0, 26.2, 26.2,
    26.4, 27.0, 27.2, 27.2, 27.2, 27.6, 28.2, 28.4, 28.4, 28.6, 29.4,
    30.2, 31.0, 31.2, 32.0, 32.0, 32.8, 33.4, 33.8, 34.4, 34.8, 35.2,
    35.2, 35.4, 35.6, 35.6, 36.2, 36.2, 38.0, 38.0, 39.2, 39.4, 40.0,
    40.4, 41.6, 41.6, 42.4, 42.8, 42.8, 43.0, 44.0, 44.4, 45.0, 46.6,
    47.8, 47.8, 48.8, 50.6, 52.0, 53.2, 55.0, 55.0, 55.4, 57.6,
]
accel = [
    0.0, -1.3, -2.7, 0.0, -2.7, -2.7, -2.7, -1.3, -2.7, -2.7, -1.3,
    -2.7, -2.7, -2.7, -5.4, -2.7, -5.4, 0.0, -2.7, -2.7, 0.0, -13.3,
    -5.4, -5.4, -9.3, -16.0, -22.8, -2.7, -22.8, -32.1, -53.5, -54.9,
    -40.2, -21.5, -21.5, -50.8, -42.9, -26.8, -21.5, -50.8, -61.7, -5.4,
    -80.4, -59.0, -71.0, -91.1, -77.7, -37.5, -85.6, -123.1, -101.9,
    -99.1, -104.4, -112.5, -50.8, -123.1, -85.6, -72.3, -127.2, -123.1,
    -117.9, -134.0, -101.9, -108.4, -123.1, -123.1, -128.5, -112.5,
    -95.1, -81.8, -53.5, -64.4, -57.6, -72.3, -44.3, -26.8, -5.4,
    -107.1, -21.5, -65.6, -16.0, -45.6, -24.2, 9.5, 4.0, 12.0, -21.5,
    37.5, 46.9, -17.4, 36.2, 75.0, 8.1, 54.9, 48.2, 46.9, 16.0, 45.6,
    1.3, 75.0, -16.0, -54.9, 69.6, 34.8, 32.1, -37.5, 22.8, 46.9, 10.7,
    5.4, -1.3, -21.5, -13.3, 30.8, -10.7, 29.4, 0.0, -10.7, 14.7, -1.3,
    0.0, 10.7, 10.7, -26.8, -14.7, -13.3, 0.0, 10.7, -14.7, -2.7, 10.7,
    -2.7, 10.7,
]
# The RW2 model needs integer-valued time indices. We rank the unique time
# values so the random walk operates on evenly spaced discrete positions.
unique_times = sort(unique(times))
time_to_idx = Dict(t => i for (i, t) in enumerate(unique_times))
time_idx = [time_to_idx[t] for t in times]
H = length(unique_times)

df = DataFrame(times = times, time_idx = time_idx, accel = accel)
first(df, 5)

# Let's visualise the data:
using AlgebraOfGraphics, CairoMakie
draw(
    data(df) *
        mapping(:times => "Time after impact (ms)", :accel => "Head acceleration (g)") *
        visual(Scatter, markersize = 5, color = :gray40),
    axis = (title = "Motorcycle crash test: accelerometer data",),
)

# The acceleration curve is highly nonlinear — a sharp negative spike around
# 15–25 ms followed by a rebound and damped oscillation. No polynomial will
# capture this shape well. This is exactly the kind of problem where
# nonparametric smoothing shines, and where a fully Bayesian approach gives us
# proper uncertainty on the estimated curve.
#
# ## Smoothing with random walks
#
# A second-order random walk (RW2) places a Gaussian prior on the *second
# differences* of a discretized function:
#
# ```math
# f_{t+2} - 2f_{t+1} + f_t \sim \mathcal{N}(0, \tau^{-1})
# ```
#
# This is the discrete analogue of penalizing the second derivative — exactly
# the same principle underlying cubic smoothing splines. The precision $\tau$
# plays the role of the smoothing parameter: high $\tau$ forces the function to
# be nearly linear (small second differences), while low $\tau$ allows more
# curvature. The key advantage of the Bayesian approach is that $\tau$ is
# *estimated from the data* rather than chosen by cross-validation.
#
# ## Fitting the model
#
# We model each observation as
#
# ```math
# y_i \sim \mathcal{N}(\mu_i, \sigma^2), \qquad \mu_i = \beta_0 + f(t_i)
# ```
#
# where $\beta_0$ is an intercept and $f$ is the RW2 smooth over time.
using Latte
using DynamicPPL
using Distributions
using GaussianMarkovRandomFields: RWModel
using LinearAlgebra

@latte function rw2_smooth(y, time_idx, H)
    σ ~ PCPrior.Sigma(50.0, α = 0.01)
    τ_rw2 ~ PCPrior.Precision(1.0, α = 0.01)
    β ~ MvNormal(zeros(1), 100.0 * I(1))
    f ~ RWModel{2}(H)(τ = τ_rw2)
    for i in eachindex(y)
        y[i] ~ Normal(β[1] + f[time_idx[i]], σ)
    end
end

# Two hyperparameters:
#
# - `σ` — the observation noise standard deviation. The PC prior
#   `PCPrior.Sigma(50.0, α = 0.01)` says "I believe there is only a 1% chance
#   that the noise SD exceeds 50 g."
# - `τ_rw2` — the RW2 precision. Its PC prior penalises complexity relative to
#   a linear baseline.
#
# We build the LGM (calling the `@latte` function) and run INLA:
lgm = rw2_smooth(df.accel, df.time_idx, H)
result = inla(lgm, df.accel; progress = false)

# ## Visualizing the fit
#
# The observation marginals give us the posterior distribution of $\mu_i$, the
# expected acceleration at each observed time. Let's plot the posterior median
# with a 95% credible band:
obs = observation_marginals(result)
fit = summary_df(obs)
fit.times = df.times

fig = Figure(size = (800, 400))
ax = Axis(
    fig[1, 1],
    xlabel = "Time after impact (ms)", ylabel = "Head acceleration (g)",
    title = "RW2 smooth: posterior fit with 95% credible band",
)
scatter!(ax, df.times, df.accel, color = :gray70, markersize = 5, label = "Observed")
band!(ax, fit.times, fit.q2_5, fit.q97_5, color = (:steelblue, 0.25), label = "95% CI")
lines!(ax, fit.times, fit.median, color = :steelblue, linewidth = 2, label = "Posterior median")
axislegend(ax, position = :rb, framevisible = false)
fig

# The smooth captures the sharp deceleration spike and subsequent rebound.
# Notice how the credible band is narrow where data is dense — the mean curve
# is well-determined there. **Important:** this band shows uncertainty in the
# *mean function* $\mu(t)$, not a prediction interval for new observations.
# Since $y_i \sim N(\mu_i, \sigma^2)$ with $\sigma \approx 25$ g, individual
# data points are expected to scatter well beyond this band. The posterior
# predictive check later in this tutorial shows the wider prediction interval
# that accounts for observation noise.
#
# ## Hyperparameter posteriors
#
# The two hyperparameters have distinct roles. The noise standard deviation
# $\sigma$ controls how much variation we attribute to measurement error, while
# the RW2 precision $\tau_{\text{rw2}}$ controls how smooth the underlying curve
# is. Their ratio determines the effective bias-variance tradeoff: high
# $\tau / \sigma^2$ favours a smooth curve, while low $\tau / \sigma^2$ lets the
# curve track the data more closely.
fig = Figure(size = (900, 400))
ax1 = Axis(
    fig[1, 1],
    xlabel = "σ (noise SD)", ylabel = "Density",
    title = "Observation noise σ",
)
ax2 = Axis(
    fig[1, 2],
    xlabel = "Precision (τ)", ylabel = "Density",
    title = "RW2 precision τ_rw2",
)
plot!(ax1, result.hyperparameter_marginals.σ)
plot!(ax2, result.hyperparameter_marginals.τ_rw2)
fig

# Let's look at the summary statistics:
println("Hyperparameter posteriors:")
println(summary_df(result.hyperparameter_marginals))

# ## Comparison with a linear model
#
# To appreciate the value of the nonparametric smooth, let's fit a simple
# linear model $\mu_i = \beta_0 + \beta_1 t_i$ and compare:
@latte function linear_model(y, x)
    σ ~ PCPrior.Sigma(50.0, α = 0.01)
    β ~ MvNormal(zeros(2), 100.0 * I(2))
    for i in eachindex(y)
        y[i] ~ Normal(β[1] + β[2] * x[i], σ)
    end
end

lgm_linear = linear_model(df.accel, df.times)
result_linear = inla(lgm_linear, df.accel; progress = false)

# Let's overlay both fits:
obs_linear = observation_marginals(result_linear)
fit_linear = summary_df(obs_linear)
fit_linear.times = df.times

fig = Figure(size = (900, 450))
ax = Axis(
    fig[1, 1],
    xlabel = "Time after impact (ms)", ylabel = "Head acceleration (g)",
    title = "Linear model vs RW2 smooth",
)
scatter!(ax, df.times, df.accel, color = :gray70, markersize = 5, label = "Observed")
lines!(ax, fit_linear.times, fit_linear.median, color = :firebrick, linewidth = 2, label = "Linear")
lines!(ax, fit.times, fit.median, color = :steelblue, linewidth = 2, label = "RW2 smooth")
axislegend(ax, position = :rb, framevisible = false)
fig

# The linear model completely misses the nonlinear structure. Let's confirm
# this with model comparison criteria:
println("Model comparison:")
println("─"^50)
for (name, res) in [("Linear", result_linear), ("RW2", result)]
    dic = res.accumulators[1].DIC
    p_d = res.accumulators[1].p_D
    mll = res.accumulators[2].log_marginal_likelihood
    waic = res.accumulators[3].WAIC
    println(
        "$name: DIC = $(round(dic, digits = 1)) (p_D = $(round(p_d, digits = 1))), " *
            "WAIC = $(round(waic, digits = 1)), " *
            "log ML = $(round(mll, digits = 1))",
    )
end

# The RW2 model dramatically outperforms the linear model on every criterion.
# The effective number of parameters ($p_D$) for the RW2 is much larger,
# reflecting its flexibility, but this additional complexity is overwhelmingly
# justified by the data.
#
# ## Posterior predictive check
#
# Finally, let's verify that the RW2 model can reproduce the variability we see
# in the data. We draw posterior predictive samples and check whether the
# observed data falls within the predictive distribution:
using Random
Random.seed!(42)

n_obs = nrow(df)
n_samples = 200
pp = hcat([rand(result, 1; include_y = true)[1].y[1:n_obs] for _ in 1:n_samples]...)
size(pp)

# For each observation, compute the 2.5th and 97.5th percentiles of the
# replicated values:
pp_lo = [quantile(pp[i, :], 0.025) for i in 1:n_obs]
pp_hi = [quantile(pp[i, :], 0.975) for i in 1:n_obs]

fig = Figure(size = (800, 400))
ax = Axis(
    fig[1, 1],
    xlabel = "Time after impact (ms)", ylabel = "Head acceleration (g)",
    title = "Posterior predictive check",
)
band!(ax, df.times, pp_lo, pp_hi, color = (:steelblue, 0.2), label = "95% predictive interval")
scatter!(ax, df.times, df.accel, color = :gray40, markersize = 4, label = "Observed")
axislegend(ax, position = :rb, framevisible = false)
fig

# The observed data sits comfortably within the predictive bands, indicating
# that the model captures both the mean structure and the variability well.
#
# ## Summary
#
# In this tutorial we used INLA with a Gaussian observation model and an RW2
# smooth to perform nonlinear regression on the classic motorcycle crash
# dataset. The key takeaways:
#
# - A **second-order random walk (RW2)** prior is the discrete analogue of a
#   cubic smoothing spline. It penalises second differences, producing a smooth
#   curve that adapts to the data.
# - The **Gaussian family** introduces an observation noise parameter $\sigma$
#   alongside the latent smoothness precision $\tau$. Together they determine the
#   bias-variance tradeoff: how much structure is attributed to the true curve
#   versus measurement noise.
# - INLA estimates both hyperparameters from the data with full posterior
#   uncertainty — no cross-validation or manual tuning required.
# - This approach is a fully Bayesian alternative to frequentist GAMs
#   (e.g., `mgcv::gam()` in R). The Bayesian version provides proper posterior
#   uncertainty on both the fitted curve and the degree of smoothness.
#
# Once you are comfortable with RW2 smoothing, natural extensions include
# heteroscedastic noise models, multiple smooth terms (additive models with
# several covariates), and spatial smoothing via the SPDE approach.
#
# ## References
#
# - Silverman, B. W. (1985). Some aspects of the spline smoothing approach to
#   non-parametric regression curve fitting. *JRSS-B*, 47(1), 1–52.
# - Rue, H. & Held, L. (2005). *Gaussian Markov Random Fields: Theory and
#   Applications*. Chapman & Hall/CRC.
