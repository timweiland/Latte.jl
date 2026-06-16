# # Bayesian Model Averaging
#
# When several candidate models are on the table, committing to a single one
# discards whatever the others have to say. Bayesian model averaging (BMA)
# keeps all of them, combining their posterior marginals with weights set by how
# well each explains the data, so the final inference carries the uncertainty
# about which model is right ([Hoeting et al., 1999](#ref-bma)).
#
# Given $K$ models $M_1, \ldots, M_K$, the averaged posterior for a quantity of
# interest is
#
# ```math
# p(\Delta \mid y) = \sum_{k=1}^{K} p(\Delta \mid y, M_k) \cdot p(M_k \mid y)
# ```
#
# where the model weights are $p(M_k \mid y) \propto p(y \mid M_k) \cdot p(M_k)$.
# The marginal likelihood $p(y \mid M_k)$ is a by-product of the INLA fit, so the
# weights cost nothing extra once the individual models are in hand.
#
# Averaging is only interesting when no single model dominates. If one model has
# a far larger marginal likelihood than the rest, its weight collapses to one and
# the average reduces to that model. The useful case is several models with
# *comparable* support, where the data genuinely cannot decide between them. This
# tutorial works through such a case: two competing smoothness structures for a
# time series whose marginal likelihoods come out nearly tied.
#
# ## The dataset: global earthquake activity
#
# We reuse the annual counts of major earthquakes (magnitude $\geq 7$) from 1900
# to 2006 introduced in the
# [temporal trend tutorial](../tutorials/temporal_trend_earthquakes.md). The data
# come from [Zucchini et al. (2016)](#ref-zucchini); with 107 observations they
# fit in a single block:
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
n_years = nrow(eq_data)
first(eq_data, 5)

# ## Two competing trend models
#
# We model the annual count $y_t$ as Poisson with a log-rate built from an
# intercept and a latent temporal effect $f_t$:
#
# ```math
# y_t \sim \text{Poisson}(\lambda_t), \qquad \log \lambda_t = \beta_0 + f_t.
# ```
#
# The two models differ only in the prior on $f$, i.e. in *what kind of trend*
# they expect. Both place a latent effect of the same length on the same years,
# so their latent fields line up position by position — exactly what averaging
# needs.
#
# **Model A — first-order random walk (RW1).** The first differences of $f$ are
# i.i.d. Gaussian, penalising abrupt jumps in level. This gives a locally
# adaptive trend that can change direction cheaply from year to year.
#
# **Model B — first-order autoregression (AR1).** Here $f$ mean-reverts toward
# zero with correlation $\rho$ between neighbouring years, $f_{t} = \rho f_{t-1} +
# \varepsilon_t$. Instead of a free-floating walk, deviations decay back to the
# overall level.
#
# Both priors come from
# [GaussianMarkovRandomFields.jl](https://github.com/timweiland/GaussianMarkovRandomFields.jl),
# called inside `@latte` so the macro recognizes them as structured Gaussians.
using Latte
using Distributions
using GaussianMarkovRandomFields: RWModel, AR1Model
using LinearAlgebra

@latte function quake_rw1(y, n)
    τ_rw ~ PCPrior.Precision(1.0, α = 0.01)
    β ~ MvNormal(zeros(1), 100.0 * I(1))
    f ~ RWModel{1}(n)(τ = τ_rw)
    for i in eachindex(y)
        y[i] ~ Poisson(exp(β[1] + f[i]))
    end
end

@latte function quake_ar1(y, n)
    τ_ar ~ PCPrior.Precision(1.0, α = 0.01)
    ρ ~ Uniform(-1.0, 1.0)
    β ~ MvNormal(zeros(1), 100.0 * I(1))
    f ~ AR1Model(n)(τ = τ_ar, ρ = ρ)
    for i in eachindex(y)
        y[i] ~ Poisson(exp(β[1] + f[i]))
    end
end

# The precision prior `PCPrior.Precision(1.0, α = 0.01)` is a penalised-complexity
# prior ([Simpson et al., 2017](#ref-pc-priors)): it says "there is only a 1%
# chance the standard deviation of the innovations exceeds 1", shrinking each
# model toward a flat trend unless the data pull it away. We give both models the
# same precision prior so the comparison turns on trend *structure*, not on prior
# scale.
#
# Calling each `@latte` function builds a `LatentGaussianModel`; `inla` runs on it.
lgm_rw1 = quake_rw1(eq_data.quakes, n_years)
result_rw1 = inla(lgm_rw1, eq_data.quakes; progress = false)

lgm_ar1 = quake_ar1(eq_data.quakes, n_years)
result_ar1 = inla(lgm_ar1, eq_data.quakes; progress = false)

# ## Comparing marginal likelihoods
#
# `log_marginal_likelihood(r)` reads off INLA's estimate of $\log p(y \mid M_k)$,
# the quantity the weights are built from. Larger is better-supported:
log_mlls = log_marginal_likelihood.([result_rw1, result_ar1])
DataFrame(model = ["RW1", "AR1"], log_ML = round.(log_mlls, digits = 2))

# The two scores come out within a fraction of a log-unit of each other. Neither
# trend structure is clearly preferred: a random walk and a mean-reverting
# process explain this series about equally well. This is precisely the regime
# where averaging earns its keep.
#
# ## Model averaging
#
# `model_average` turns the marginal likelihoods into posterior model weights and
# blends the two fits position by position. It returns a `BMAResult` whose
# `model_weights` are the posterior model probabilities $p(M_k \mid y)$:
bma = model_average([result_rw1, result_ar1])
DataFrame(model = ["RW1", "AR1"], weight = round.(bma.model_weights, digits = 3))

# The weights are split roughly evenly. The averaged posterior is therefore a
# real blend of both trends, not a thin veneer over one of them.
#
# ## The blended trend
#
# `bma.latent_marginals` holds the model-averaged marginal for each latent
# variable — here the intercept $\beta_0$ followed by the 107 temporal effects
# $f_t$, each a `WeightedMixture` of its RW1 and AR1 counterparts. To plot a
# trend on the count scale we want $\lambda_t = \exp(\beta_0 + f_t)$, which lives
# in the *observation* marginals. We blend those the same way BMA blends the
# latent field: a `WeightedMixture` of the per-year observation marginals under
# each model, weighted by the posterior model weights.
using Latte: WeightedMixture

obs_rw1 = observation_marginals(result_rw1)
obs_ar1 = observation_marginals(result_ar1)
obs_bma = [
    WeightedMixture([obs_rw1[t], obs_ar1[t]], bma.model_weights) for t in 1:n_years
];

# A tidy long-form table of the posterior-median trend for each model plus the
# blend lets AlgebraOfGraphics draw all three with one `mapping`:
using AlgebraOfGraphics, CairoMakie

trend_df = vcat(
    DataFrame(year = eq_data.year, median = median.(obs_rw1), model = "RW1"),
    DataFrame(year = eq_data.year, median = median.(obs_ar1), model = "AR1"),
    DataFrame(year = eq_data.year, median = median.(obs_bma), model = "BMA"),
)
first(trend_df, 5)

# Overlaying the three medians on the raw counts:
pts = data(eq_data) *
    mapping(:year => "Year", :quakes => "Major earthquakes (M ≥ 7)") *
    visual(Scatter, markersize = 5, color = :gray70)
trends = data(trend_df) *
    mapping(:year => "Year", :median => "Major earthquakes (M ≥ 7)", color = :model => "Model") *
    visual(Lines, linewidth = 2)
draw(
    pts + trends,
    axis = (title = "RW1, AR1, and the model-averaged trend",),
)

# The RW1 and AR1 medians trace slightly different paths, and the BMA trend
# (its own colour) runs between them — closest to whichever model carries more
# weight at each point. Because the weights here are near 50/50, the blend sits
# roughly midway rather than hugging either single fit.
#
# ## Uncertainty in the blend
#
# The averaged marginal is a mixture of the two single-model marginals, so its
# variance is the weight-averaged variance of the two plus a term for how far
# apart their means sit:
#
# ```math
# \operatorname{Var}_\text{BMA} = \sum_k w_k \operatorname{Var}_k + \sum_k w_k (\mu_k - \bar\mu)^2.
# ```
#
# That means the blend lands *between* the two models' spreads, nudged upward by
# any disagreement in their centers — it neither copies the sharpest model nor
# automatically exceeds the widest. We can read this off the intercept $\beta_0$,
# the first latent variable:
β_bma = bma.latent_marginals[1]
β_rw1 = latent_marginals(result_rw1, :β)[1]
β_ar1 = latent_marginals(result_ar1, :β)[1]
DataFrame(
    source = ["RW1", "AR1", "BMA"],
    mean = round.(mean.([β_rw1, β_ar1, β_bma]), digits = 3),
    std = round.(std.([β_rw1, β_ar1, β_bma]), digits = 3),
)

# The BMA standard deviation (0.068) sits between the two: well above RW1's tight
# 0.024 and just below AR1's 0.091, since AR1 carries slightly more weight and the
# two means nearly coincide. Committing to RW1 alone would have understated the
# intercept uncertainty almost threefold; the blend keeps the more cautious model's
# contribution in proportion to its support, which is the honest thing to report.
#
# Plotting the three intercept densities shows the blend as a weighted mixture of
# the two. We evaluate each density on a shared grid and assemble a tidy frame for
# AlgebraOfGraphics:
grid = range(2.4, 3.3, length = 300)
dens_df = vcat(
    DataFrame(x = grid, density = pdf.(β_rw1, grid), source = "RW1"),
    DataFrame(x = grid, density = pdf.(β_ar1, grid), source = "AR1"),
    DataFrame(x = grid, density = pdf.(β_bma, grid), source = "BMA"),
)
draw(
    data(dens_df) *
        mapping(:x => "Intercept β₀", :density => "Density", color = :source => "Source") *
        visual(Lines, linewidth = 2),
    axis = (title = "Intercept: single-model marginals vs the BMA blend",),
)

# The BMA curve is a weighted mixture of the two single-model densities: it places
# mass over both locations in proportion to the model weights rather than
# collapsing onto either. Reporting it instead of a single model passes the
# model-selection uncertainty through to the final inference.
#
# ## Using prior model weights
#
# `model_average` assumes equal prior weights unless told otherwise. When prior
# knowledge favours one structure, pass `prior_weights` to tilt the comparison.
# Here we suppose a 3:1 prior preference for the random walk:
bma_prior = model_average([result_rw1, result_ar1]; prior_weights = [0.75, 0.25])
DataFrame(model = ["RW1", "AR1"], weight = round.(bma_prior.model_weights, digits = 3))

# The prior shifts the posterior weights toward RW1, but because the marginal
# likelihoods are so close the data offer little to override it: the prior
# largely carries through.
#
# ## Working with BMA results
#
# Each averaged marginal is a `WeightedMixture`, so the full Distributions.jl
# interface applies directly. Posterior mean and a 95% credible interval for the
# intercept:
mean(β_bma), quantile.(β_bma, (0.025, 0.975))

# `summary_df` tabulates the averaged latent marginals the same way it does for a
# single fit. We show the intercept and the first few temporal effects:
first(summary_df(bma.latent_marginals), 5)

# ## Summary
#
# - BMA is worthwhile when several models have *comparable* marginal likelihoods;
#   a lopsided comparison just recovers the single best model.
# - Here a random walk and an AR1 process explain the earthquake series almost
#   equally well, giving near-even weights and a genuine blend.
# - `model_average` reuses the marginal likelihood INLA already computes, so the
#   weights are free once the individual fits are done.
# - The averaged marginals are `WeightedMixture` distributions: they sit between
#   the single-model fits, with a spread that blends both models' uncertainty and
#   their disagreement rather than committing to one model's confidence.
#
# ## References
#
# ```@raw html
# <div class="ref-grid-2">
# <PaperCite
#   tag="BMA"
#   title="Bayesian Model Averaging: A Tutorial"
#   authors="J. A. Hoeting, D. Madigan, A. E. Raftery & C. T. Volinsky"
#   venue="Statistical Science" year="1999"
#   doi="10.1214/ss/1009212519"
#   url="https://doi.org/10.1214/ss/1009212519"
#   abstract="The standard tutorial reference on Bayesian model averaging: combining inferences across models weighted by their posterior model probabilities to account for model uncertainty." />
# <PaperCite
#   tag="INLA"
#   title="Approximate Bayesian Inference for Latent Gaussian Models by Using Integrated Nested Laplace Approximations"
#   authors="H. Rue, S. Martino & N. Chopin"
#   venue="J. R. Statist. Soc. B" year="2009"
#   doi="10.1111/j.1467-9868.2008.00700.x"
#   url="https://doi.org/10.1111/j.1467-9868.2008.00700.x"
#   abstract="The original INLA paper: deterministic approximate Bayesian inference for latent Gaussian models via nested Laplace approximations and numerical integration over the hyperparameters. The marginal likelihood it reports drives the BMA weights." />
# <PaperCite
#   tag="PC priors"
#   title="Penalising Model Component Complexity: A Principled, Practical Approach to Constructing Priors"
#   authors="D. Simpson, H. Rue, A. Riebler, T. G. Martins & S. H. Sørbye"
#   venue="Statistical Science" year="2017"
#   arxiv="1403.4630"
#   doi="10.1214/16-STS576"
#   url="https://doi.org/10.1214/16-STS576"
#   abstract="Penalised-complexity (PC) priors: weakly informative priors that shrink a model component towards a simpler base model, used here for the random-walk and AR1 precision." />
# <PaperCite
#   tag="Zucchini"
#   title="Hidden Markov Models for Time Series: An Introduction Using R (2nd ed.)"
#   authors="W. Zucchini, I. L. MacDonald & R. Langrock"
#   venue="Chapman & Hall/CRC" year="2016"
#   doi="10.1201/b20790"
#   url="https://doi.org/10.1201/b20790"
#   abstract="Source of the annual major-earthquake counts (1900–2006) used here as the competing-trend example." />
# </div>
# ```
