# # Getting started with Latte.jl
#
# This tutorial walks through a small Bayesian analysis of mortality rates
# following cardiac surgery across twelve hospitals. We write the model once
# and then run it through all three of Latte's inference engines, INLA, TMB,
# and HMC-Laplace, swapping a single function call each time. That "define
# once, run any engine" workflow is the thread this tutorial pulls on.
#
# ## Installation
# If you haven't done so already, you need to install both
# [Julia](https://julialang.org/install/) and this package:
#
# ```julia
# using Pkg
# Pkg.add("Latte")
# ```
#
# ## Latent Gaussian models and the three engines
#
# Latte targets *latent Gaussian models* (LGMs): hierarchical models with a
# Gaussian latent field, such as random intercepts, smooth temporal trends, or
# spatial fields, observed through an exponential-family likelihood. This is a
# broad class. It covers generalized linear mixed models, geostatistics, and
# spatio-temporal models built on Gaussian Markov random fields.
#
# For these models the latent field can be integrated out with a Laplace
# approximation, which is the shared backbone of all three engines Latte
# ships. They differ only in how they handle the remaining *hyperparameters*:
#
# - **INLA** spreads a small grid of hyperparameter points and integrates over
#   them. Deterministic, fast, and the default.
# - **TMB** finds the most likely hyperparameters and fits one Gaussian at that
#   mode. The fastest engine; it reports a mode and standard errors.
# - **HMC-Laplace** samples the hyperparameters with NUTS. The most faithful
#   when the hyperparameter posterior is awkward, and the slowest.
#
# Because they all consume the same `LatentGaussianModel`, you write the model
# once and pick the engine afterwards. The rest of this tutorial does exactly
# that.
#
# ## The dataset
# The data records, for each hospital, the number of operations `n` and the
# number of deaths `r` following cardiac surgery on infants. Here it is:
using DataFrames
surg_data = DataFrame(
    hospital = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"],
    n = [47, 148, 119, 810, 211, 196, 148, 215, 207, 97, 256, 360],
    r = [0, 18, 8, 46, 8, 13, 9, 31, 14, 8, 29, 24]
)

# Let's visualize it using AlgebraOfGraphics.jl:
using AlgebraOfGraphics, CairoMakie
draw(
    data(surg_data) *
        mapping(
        :hospital => "Hospital",
        (:r, :n) => ((r, n) -> r ./ n) => "Observed mortality rate"
    ) *
        visual(BarPlot),
    axis = (
        xticklabelrotation = π / 4, title = "Observed surgery mortality rates by hospital",
    )
)

# ## Modelling
# Latte.jl models are written as `@latte` functions, using the same
# probabilistic programming syntax as
# [DynamicPPL.jl](https://github.com/TuringLang/DynamicPPL.jl) / Turing.jl.
# If you know Turing, this will look very familiar. `@latte` recognizes the
# structured Gaussian priors in the body (here an `IIDModel`) and builds a
# `LatentGaussianModel` that rides Latte's fast inference paths. (Already have
# a plain Turing `@model`? The Turing handoff tutorial covers `latte_from_dppl`.)
#
# For the surgery data, we assume each hospital has its own (logit-scale)
# mortality rate `β + u[h]`, where `β` is an overall intercept and `u[h]`
# is a hospital-specific random effect with a Gaussian prior. The number
# of deaths in each hospital is then Binomial.
#
# A few modelling details are worth calling out. The hospital effect `u` uses
# `IIDModel(H, constraint = :sumtozero)(τ = τ_u)`, an IID Gaussian random effect
# whose sum-to-zero constraint keeps `β` and `u` identified. The random-effect
# precision `τ_u` gets a `Gamma(2, 1)` prior, which places mild mass on moderate
# precisions. The inverse-logit link comes from `logistic` in `StatsFuns`.
using Latte
using Distributions
using GaussianMarkovRandomFields: IIDModel
using StatsFuns: logistic
using LinearAlgebra

@latte function surg_mortality(r, n_trials, hospital_idx, H)
    τ_u ~ Gamma(2.0, 1.0)
    β ~ MvNormal(zeros(1), 100.0 * I(1))
    u ~ IIDModel(H, constraint = :sumtozero)(τ = τ_u)
    for i in eachindex(r)
        r[i] ~ Binomial(
            n_trials[i], logistic(β[1] + u[hospital_idx[i]])
        )
    end
end

# Calling the `@latte` function returns a `LatentGaussianModel`. Latte reads
# off which variables form the latent Gaussian field (the "random effects" in
# mixed-model vocabulary — here `β` and `u`) and which are hyperparameters
# (`τ_u`) straight from the model body, so there's nothing else to specify.
H = length(surg_data.hospital)
hospital_idx = 1:H |> collect
lgm = surg_mortality(surg_data.r, surg_data.n, hospital_idx, H)

# ## Running INLA
# With the model in hand, the default engine is one call away:
inla_result = inla(lgm, surg_data.r)

# The output is an `INLAResult`. It holds the full analysis and prints a short
# summary when displayed.
#
# Every marginal Latte computes implements the Distributions.jl interface, so
# you can call Distributions.jl methods on these objects directly. The
# accessor functions return marginals as blocks: one distribution per
# hyperparameter, one per latent variable. We read them back through these
# accessors rather than the result's fields, since the accessors behave the
# same regardless of how the result was stored internally.
#
# For a quick table of key statistics, `summary_df` collects a block of
# marginals into a DataFrame. Here are the hyperparameter and latent-field
# marginals:
summary_df(hyperparameter_marginals(inla_result))

#-
summary_df(latent_marginals(inla_result))

# We can also map the per-hospital linear predictors `η = β + u[hospital]`
# through the inverse link onto the mortality-rate scale, the posterior
# predictive marginals, and plot them against the data with AlgebraOfGraphics:
pred_df = summary_df(observation_marginals(inla_result))
pred_df.hospital = surg_data.hospital
data(pred_df) * (
    mapping(:hospital => "Hospital", :q2_5, :q97_5) *
        visual(Rangebars, whiskerwidth = 8, color = :gray80) +
        mapping(:hospital => "Hospital", :median => "Median") *
        visual(Scatter, markersize = 10)
) |> draw(;
    axis = (
        xticklabelrotation = π / 4,
        title = "Posterior predictive marginals of mortality rate by hospital",
        ylabel = "Mortality rate",
    )
)

# Hospital H stands out, with the highest fitted rate and an interval well
# above the others. The companion tutorial,
# [Getting familiar with INLA](inla_in_depth.md), goes deeper on these
# accessors, on model-comparison criteria like DIC and WAIC, and on drawing
# joint posterior samples.

# ## The same model through TMB
# `tmb` runs the fast point-estimate engine: it finds the hyperparameter mode
# (the MAP) and fits a single Gaussian there, giving a MAP value and a standard
# error for each hyperparameter. Note that the model object is the same `lgm`,
# only the entry-point function changes:
tmb_result = tmb(lgm, surg_data.r)

# The same accessors work on the result, returning the same Distributions.jl
# marginals. TMB's marginal for `τ_u` is a single Gaussian (in the working,
# here log, space) reported back on the natural scale, so its median is the MAP
# estimate and its spread the standard error:
τ_tmb = hyperparameter_marginals(tmb_result, :τ_u)[1]
median(τ_tmb), std(τ_tmb)

# TMB is exact when the hyperparameter posterior is close to Gaussian in the
# working (here log) space, which is often the case for well-identified models.
# Its single Gaussian cannot bend to follow a skewed posterior, though, which
# is the trade-off against INLA's grid.

# ## The same model through HMC-Laplace
# `hmc_laplace` keeps the inner Laplace approximation for the latent field but
# replaces the deterministic hyperparameter treatment with NUTS, sampling the
# hyperparameter posterior directly. It is the most faithful engine when that
# posterior is genuinely non-Gaussian, at the cost of running an MCMC chain. We
# seed the RNG for a reproducible chain:
using Random
hmc_result = hmc_laplace(
    lgm, surg_data.r; n_samples = 500, n_warmup = 200, rng = MersenneTwister(0)
)

# Check that the chain behaved before trusting the marginals, then read `τ_u`
# back through the same accessor. Here the marginal is an empirical one built
# from the NUTS draws, summarised by its median and a 95% credible interval:
converged(hmc_result), divergences(hmc_result)

#-
τ_hmc = hyperparameter_marginals(hmc_result, :τ_u)[1]
median(τ_hmc), quantile(τ_hmc, 0.025), quantile(τ_hmc, 0.975)

# ## Which engine, when?
# All three approximate the same posterior; they differ in how they handle the
# hyperparameters and therefore in speed and faithfulness.
#
# - Reach for **`inla`** by default. It is fast and accurate on the great
#   majority of latent Gaussian models, and it recovers the hyperparameter skew
#   that TMB misses.
# - Reach for **`tmb`** when you want the quickest fit and a MAP-plus-standard-error
#   summary is enough, or when the model has many hyperparameters and you are
#   content with empirical Bayes at the mode.
# - Reach for **`hmc_laplace`** when the hyperparameter posterior is curved or
#   strongly skewed (scale-versus-correlation trade-offs, parameters pinned
#   against a boundary) and you need faithful tails, and there are few enough
#   hyperparameters that sampling stays affordable.
#
# The dedicated tutorials work each engine on a model built for its strengths:
# [age-structured stock assessment (SAM)](age_structured_sam.md) for TMB, and
# [sampling hyperparameters](hmc_laplace_when.md) for HMC-Laplace. The
# [TMB](../engines/tmb.md) and [HMC-Laplace](../engines/hmc_laplace.md) engine
# pages cover the methods and their tuning in more detail.

# ## Conclusion
# That covers the core workflow: write one `@latte` model, then run `inla`,
# `tmb`, or `hmc_laplace` on it and read the marginals back through the same
# accessor functions. Next, [Getting familiar with INLA](inla_in_depth.md)
# stays with this surgery model and digs into the result object: marginal
# accessors in depth, model-comparison criteria, and posterior sampling. The
# other tutorials build on this with spatial fields, temporal smoothing, custom
# likelihoods, and more.

# ## References
#
# ```@raw html
# <div class="ref-grid-2">
# <PaperCite
#   tag="INLA"
#   title="Approximate Bayesian Inference for Latent Gaussian Models by Using Integrated Nested Laplace Approximations"
#   authors="H. Rue, S. Martino & N. Chopin"
#   venue="J. R. Statist. Soc. B" year="2009"
#   doi="10.1111/j.1467-9868.2008.00700.x"
#   url="https://doi.org/10.1111/j.1467-9868.2008.00700.x"
#   abstract="The original INLA paper: deterministic approximate Bayesian inference for latent Gaussian models via nested Laplace approximations and numerical integration over the hyperparameters." />
# <PaperCite
#   tag="TMB"
#   title="TMB: Automatic Differentiation and Laplace Approximation"
#   authors="K. Kristensen, A. Nielsen, C. W. Berg, H. Skaug & B. M. Bell"
#   venue="Journal of Statistical Software" year="2016"
#   doi="10.18637/jss.v070.i05"
#   url="https://doi.org/10.18637/jss.v070.i05"
#   abstract="The TMB R package: fast Laplace approximation of the marginal likelihood for latent-variable models, with automatic differentiation for the gradients and Hessians." />
# <PaperCite
#   tag="Embedded Laplace + HMC"
#   title="Hamiltonian Monte Carlo using an Adjoint-differentiated Laplace Approximation"
#   authors="C. C. Margossian, A. Vehtari, D. Simpson & R. Agrawal"
#   venue="Advances in Neural Information Processing Systems (NeurIPS)" year="2020"
#   arxiv="2004.12550"
#   url="https://arxiv.org/abs/2004.12550"
#   abstract="Hamiltonian Monte Carlo over the hyperparameters, with the latent Gaussian field marginalised by an embedded Laplace approximation and the gradient propagated through that inner solve. The method this engine implements." />
# </div>
# ```
