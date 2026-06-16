# # Getting familiar with INLA
#
# The [Getting started](getting_started.md) tutorial introduced the surgery
# model and ran it through all three engines in a few lines each. This tutorial
# stays with that same model and digs into the INLA result object: how to read
# the marginals back through the accessor functions, how to use the
# model-comparison criteria INLA computes along the way, and how to draw joint
# posterior samples.
#
# ## The surgery model
#
# We rebuild the model from [Getting started](getting_started.md) unchanged: a
# Binomial likelihood for the per-hospital death counts, an overall logit-scale
# intercept `β`, and a sum-to-zero IID hospital effect `u` with precision `τ_u`.
using DataFrames
surg_data = DataFrame(
    hospital = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"],
    n = [47, 148, 119, 810, 211, 196, 148, 215, 207, 97, 256, 360],
    r = [0, 18, 8, 46, 8, 13, 9, 31, 14, 8, 29, 24]
)

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

H = length(surg_data.hospital)
hospital_idx = 1:H |> collect
lgm = surg_mortality(surg_data.r, surg_data.n, hospital_idx, H)

inla_result = inla(lgm, surg_data.r)

# ## Marginals in depth
#
# Every marginal Latte computes implements the Distributions.jl interface, so
# you can call Distributions.jl methods on these objects directly. The rule of
# thumb is to read marginals back out through the accessor functions rather
# than the result's fields: the accessors work the same way regardless of how
# the result was stored internally.
#
# Each accessor returns a *block* of marginals. `hyperparameter_marginals`
# takes a hyperparameter name and returns the marginals for that named block,
# here a single distribution for `τ_u`:
τ_marginal = hyperparameter_marginals(inla_result, :τ_u)[1]

# Because it is an ordinary Distributions.jl object, the usual methods apply.
# Read off the median and a 95% credible interval:
median(τ_marginal), quantile(τ_marginal, 0.025), quantile(τ_marginal, 0.975)

# Sample from it:
rand(τ_marginal, 3)

# Or evaluate its density and CDF at a point:
pdf(τ_marginal, 1.0), cdf(τ_marginal, 1.0)

# Calling an accessor without a name returns every block at once. `summary_df`
# turns a block into a tidy DataFrame of key statistics — the quickest way to
# scan a whole posterior:
summary_df(hyperparameter_marginals(inla_result))

# The same works for the latent field. These are exactly the variables in the
# model: the intercept `β` plus the 12 hospital effects, 13 in total. Pass a
# name to slice out a single latent block:
summary_df(latent_marginals(inla_result, :u))

# We can also ask for the per-hospital linear predictors `η = β + u[hospital]`.
# These aren't latent variables we sampled directly, so Latte derives them from
# the latent posterior on demand:
summary_df(linear_predictor_marginals(inla_result))

# And the posterior predictive marginals, which map the linear predictors through
# the inverse link function onto the mortality-rate scale:
summary_df(observation_marginals(inla_result))

# ## Model-comparison criteria
#
# INLA computes several model-comparison quantities along the way, evaluated at
# each integration point of the hyperparameter posterior. By default Latte
# attaches four accumulators: the Deviance Information Criterion (DIC), the
# marginal likelihood, WAIC, and CPO. Their values appear in the `INLAResult`
# summary, and you can also reach them directly through the `accumulators`
# field.
#
# The first accumulator carries the DIC and its effective number of parameters:
inla_result.accumulators[1].DIC, inla_result.accumulators[1].p_D

# The marginal likelihood has its own accessor:
log_marginal_likelihood(inla_result)

# Quantities like these are the basis of Bayesian model selection: fit competing
# models to the same data and compare them on DIC or WAIC (lower is better), or on
# the marginal likelihood (higher is better). The marginal likelihood is also what
# turns several candidate models into a weighted average. The
# [Bayesian model averaging tutorial](../tutorials/bayesian_model_averaging.md)
# works through a comparison of two competing trend models on a shared dataset,
# built from exactly these quantities.

# ## Posterior sampling
#
# The marginals above are one-dimensional summaries. When you need *joint*
# draws — for instance to propagate posterior uncertainty through a downstream
# function of several latent variables — `rand` draws from the full approximate
# posterior. Each draw picks a hyperparameter configuration from the integration
# grid, then a joint latent field from the inner Gaussian at that configuration.
using Random
Random.seed!(0)
samples = rand(inla_result, 1000);

# `rand(result, n)` returns a `PosteriorSamples` object holding row-aligned
# matrices: one row per draw, with `θ` the hyperparameters and `x` the latent
# field. Their shapes:
size(samples.θ), size(samples.x)

# From the joint draws we can compute any functional of the posterior. The
# columns of `samples.x` follow the model's latent layout, which
# `latent_groups` reports as a name-to-column-range map. We use it to locate
# the hospital block rather than hard-coding offsets:
u_cols = latent_groups(inla_result)[:u]

# With that, the posterior probability that hospital H (the eighth) has a
# higher logit-scale effect than hospital A (the first):
mean(samples.x[:, u_cols[8]] .> samples.x[:, u_cols[1]])

# Passing `include_y = true` additionally draws posterior-predictive observations,
# the basis of posterior predictive checks. The
# [posterior predictive checks tutorial](../tutorials/posterior_predictive_checks.md)
# walks through that workflow end to end.

# ## Conclusion
#
# That covers the INLA result object in depth: the marginal accessors and their
# Distributions.jl interface, the DIC / WAIC / marginal-likelihood accumulators
# for model comparison, and joint posterior sampling. The same accessors work
# on `tmb` and `hmc_laplace` results too, so once you are comfortable here the
# other engines read the same way.

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
#   tag="Review"
#   title="Bayesian Computing with INLA: A Review"
#   authors="H. Rue, A. Riebler, S. H. Sørbye, J. B. Illian, D. P. Simpson & F. K. Lindgren"
#   venue="Annual Review of Statistics and Its Application" year="2017"
#   doi="10.1146/annurev-statistics-060116-054045"
#   url="https://doi.org/10.1146/annurev-statistics-060116-054045"
#   abstract="A modern review of the INLA methodology, the SPDE approach, and the R-INLA ecosystem." />
# </div>
# ```
