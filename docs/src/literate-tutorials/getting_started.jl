# # Getting started with Latte.jl
#
# Welcome!
# In this hands-on tutorial we will walk through a very simple Bayesian analysis
# of mortality rates following surgery in some hospitals.
# In the process, you will learn the basics of Latte.jl enabling
# you to get started with your own analyses.
#
# ## Installation
# If you haven't done so already, you need to install both Julia and this package.
# See the instructions on the home page.
#
# ## What are integrated nested Laplace approximations?
#
# Integrated Nested Laplace Approximations (INLA) is a computational method for
# fast approximate Bayesian inference in latent Gaussian models. Instead of using
# slow MCMC sampling, INLA uses a combination of Laplace approximations and numerical
# integration to quickly compute posterior marginal distributions for parameters and
# hyperparameters.
#
# INLA is particularly well-suited for:
# - **Hierarchical models** with structured latent effects (random intercepts, spatial fields, etc.)
# - **Generalized linear mixed models** with exponential family likelihoods
# - **Geostatistical and spatio-temporal models** with Gaussian Markov random fields
# - **Problems requiring speed** where you need results in seconds rather than hours
#
# This package, Latte.jl, provides a modern Julia implementation
# of INLA with emphasis on clarity, flexibility, and composability.
#
# ## The dataset
# We're going to analyze a very simple dataset that contains mortality rates following
# cardiac surgery on babies in twelve different hospitals.
#
# This is our dataset:
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
    axis = (xticklabelrotation = π / 4, title = "Observed surgery mortality rates by hospital")
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
# A couple of modelling details worth calling out:
# - `u` uses Latte's `IIDModel(H, constraint = :sumtozero)(τ = τ_u)` — an
#   IID Gaussian random effect with a sum-to-zero constraint, which keeps
#   `β` and `u` identified.
# - We give `τ_u` (the random-effect precision) a `Gamma(2, 1)` prior. It
#   puts mild mass on moderate precisions; nothing fancy.
# - We use `logistic` from `StatsFuns` for the inverse-logit link, mirroring
#   the Turing idiom.
using Latte
using DynamicPPL
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
# We have our data, we've specified the model, and we've chosen a prior for its hyperparameter.
# Without further ado, let's run INLA:
inla_result = inla(lgm, surg_data.r)

# The output is of type `INLAResult`, which contains the results of the analysis
# (and spits out a nice summary).
#
# Let's dissect the results.
#
# ## Exploring the Results
#
# All marginals computed by Latte.jl implement the Distributions.jl interface.
# This means that you can simply call methods from Distributions.jl directly on these objects.
# Let's try this for the marginal of the IID model's precision:
τ_marginal = inla_result.hyperparameter_marginals.τ_u

# Let's compute the median and a confidence interval:
median(τ_marginal), quantile(τ_marginal, 0.025), quantile(τ_marginal, 0.975)

# We can also sample:
rand(τ_marginal, 3)

# If you'd prefer to get a quick summary of key statistics, the helper method `summary_df` is your friend:
summary_df(inla_result.hyperparameter_marginals)

# We can do the same for the marginals of the latent field. These are exactly the
# variables we specified in the model: the intercept `β` plus the 12 hospital
# effects, 13 in total.
summary_df(inla_result.latent_marginals)

# We can also ask for the per-hospital linear predictors `η = β + u[hospital]`.
# These aren't latent variables we sampled directly, so Latte derives them from
# the latent posterior on demand:
summary_df(linear_predictor_marginals(inla_result))

# And the posterior predictive marginals, which map the linear predictors through
# the inverse link function onto the mortality-rate scale:
pred_df = summary_df(observation_marginals(inla_result))

# We can use AlgebraOfGraphics again to visualize our results nicely:
pred_df.hospital = surg_data.hospital
data(pred_df) * (
    mapping(:hospital => "Hospital", :q2_5, :q97_5) * visual(Rangebars, whiskerwidth = 8, color = :gray80) +
        mapping(:hospital => "Hospital", :median => "Median") * visual(Scatter, markersize = 10)
) |> draw(;
    axis = (
        xticklabelrotation = π / 4,
        title = "Posterior predictive marginals of mortality rate by hospital",
        ylabel = "Mortality rate",
    )
)

# ... so it'd be wise to avoid Hospital H :)
#
# Lastly, let's take a look at accumulators.
# These get called for each integration point of the hyperparameter posterior.
# Two accumulators are included by default, to compute the Deviance Information Criterion (DIC) and the marginal likelihood.
# You saw their output in the `INLAResult` summary, but you can also access their results directly:
inla_result.accumulators[1]

#
inla_result.accumulators[1].DIC, inla_result.accumulators[1].p_D

#
inla_result.accumulators[2]

# These two quantities are most valuable for Bayesian model selection.
# If we run INLA on the same data for two different models, these quantities help us decide between the two.

# ## Conclusion
# Congratulations! You just learned the basic usage of Latte.jl.
#
# Curious to learn more? Check our other tutorials.
