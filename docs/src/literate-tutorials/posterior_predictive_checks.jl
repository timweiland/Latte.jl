# # Posterior predictive checks
#
# Fitting a model returns a posterior; the posterior is an opinion about
# what the data plausibly looks like. A posterior-predictive check (PPC)
# flips that around: draw new datasets from the fitted model, then ask
# whether they resemble the observed one. If they don't, the model is
# missing something.
#
# Latte exposes a tiny three-function surface for this:
#
# - `posterior_predictive(result, n)` — draw `n` replicate datasets
# - `ppc_stat(T, y, y_rep)` — evaluate a test statistic on each
# - `bayesian_pvalue(T, y, y_rep)` — tail probability
#
# In this tutorial we'll fit a plain Poisson model to data that's
# actually *overdispersed* and watch the PPC catch it.
using Latte
using Distributions
using GaussianMarkovRandomFields: IIDModel
using Random, Statistics
using CairoMakie

Random.seed!(20260424)

# ## Generate overdispersed counts
#
# We simulate from a Negative-Binomial with the same mean as a Poisson
# but with larger variance — classic overdispersion.
n = 80
μ_true = exp.(randn(n) .* 0.3 .+ 1.5)
y = rand.(NegativeBinomial.(2.0, 2.0 ./ (2.0 .+ μ_true)))
println("n = $n, observed mean = $(round(mean(y), digits = 2)), var = $(round(var(y), digits = 2))")

# ## Fit a Poisson model (deliberately wrong)
@latte function poisson_fit(y, n)
    τ ~ PCPrior.Precision(1.0, α = 0.01)
    x ~ IIDModel(n)(τ = τ)
    for i in eachindex(y)
        y[i] ~ Poisson(exp(x[i]))
    end
end
lgm = poisson_fit(y, n)
result = inla(lgm, y; progress = false)

# ## Draw posterior predictive datasets
y_rep = posterior_predictive(result, 1000)
println("y_rep shape: ", size(y_rep))

# ## Check a few statistics
#
# We pick three: `mean`, `std`, and `maximum`. A well-specified model
# reproduces all of them; a Poisson fit to overdispersed counts should
# match the mean but under-estimate spread and tail.
stats = [("mean", mean), ("std", std), ("maximum", maximum)]

fig = Figure(size = (900, 300))
for (k, (name, T)) in enumerate(stats)
    T_obs, T_rep = ppc_stat(T, y, y_rep)
    p = bayesian_pvalue(T, y, y_rep)
    local ax = Axis(
        fig[1, k], xlabel = name, ylabel = "count",
        title = "$name — Bayesian p = $(round(p, digits = 3))"
    )
    hist!(ax, T_rep, bins = 30, color = (:steelblue, 0.6))
    vlines!(ax, [T_obs], color = :red, linewidth = 2)
end
fig

# The red line is the observed statistic; the histogram is its
# posterior-predictive distribution under the Poisson fit. For
# overdispersed data:
#
# - `mean` sits comfortably inside the replicated distribution
#   (p ≈ 0.5) — Poisson captures the location.
# - `std` and `maximum` fall well outside the histogram — the fitted
#   Poisson systematically generates less-variable, lower-tail data
#   than we observed.
#
# A p-value near 0 or 1 is the signal: the model is mis-specified for
# that aspect of the data. Here the fix would be a likelihood that
# accommodates overdispersion (NegativeBinomial, observation-level
# random effect, etc.).
#
# ## Rule of thumb
#
# PPC is qualitative — Bayesian p-values aren't calibrated — but
# "far from 0.5" on a statistic you care about is a strong hint. Pick
# statistics that target the aspect of the data you want the model to
# capture (tails, zero counts, autocorrelation, group-level moments),
# not just `mean` and `std`.
