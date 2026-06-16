# # Handoff: same model, INLA and Turing
#
# Latte uses [DynamicPPL](#ref-dynamicppl)'s `@model` macro as its modelling
# surface. So does [Turing.jl](#ref-turing). A model you write for `inla()` can
# therefore be handed straight to `Turing.sample` for MCMC, with no rewriting in
# between.
#
# This tutorial shows the handoff on a small Poisson model with an
# IID-Normal latent field, and compares the posteriors the two engines return.
#
# The shared model is useful in a few situations:
# - When you suspect [INLA](#ref-inla)'s Laplace approximation might be off on an
#   unusual model, a short NUTS run on the same `@model` is the most direct check.
# - INLA is fast and approximate; HMC is slow and asymptotically exact.
#   Keeping one model definition lets you pick the engine per problem.
# - Latte also exposes `hmc_laplace(lgm, y)`, which runs [NUTS](#ref-nuts) over the
#   hyperparameters with an [inner Laplace step](#ref-embedded-laplace-hmc) on the
#   latent field. It sits between `inla` and full NUTS.
#
# ## A small Poisson model
using Latte
using DynamicPPL: @model
using Distributions
using GaussianMarkovRandomFields: IIDModel
using Turing
using Random, Statistics
using CairoMakie

Random.seed!(20260424)

@model function iid_poisson(y, n)
    τ ~ PCPrior.Precision(1.0, α = 0.01)
    x ~ IIDModel(n)(τ = τ)
    for i in eachindex(y)
        y[i] ~ Poisson(exp(x[i]); check_args = false)
    end
end

n = 15
true_x = randn(n) .* 0.4 .+ 0.8
y = rand.(Poisson.(exp.(true_x)))
first(y, 5)

# ## Fit with INLA
#
# `latte_from_dppl` packages the DPPL model into a `LatentGaussianModel` that
# `inla()` understands. It is the same object `tmb()` and `hmc_laplace()` take.
model = iid_poisson(y, n)
lgm = latte_from_dppl(model; random = (:x,))
result = inla(lgm, y; progress = false)

# Read the latent field back through the accessor, keyed by the name it has in
# the model (`:x`), rather than reaching into the result's fields:
x_inla = mean.(latent_marginals(result, :x))
first(x_inla, 5)

# ## Fit with Turing NUTS
#
# Turing consumes the `@model` directly, so there is no packaging step here.
chain = sample(model, NUTS(), 2000; progress = false)
x_nuts = [mean(chain[Symbol("x[$i]")]) for i in 1:n];

# ## Compare posteriors
#
# Latent posterior means from the two engines line up on the diagonal.
# Raw Makie here: an identity-reference-line scatter (the dashed y = x line spans
# the shared data range), which AoG's tidy-data idiom does not express cleanly.
fig = Figure(size = (500, 500))
ax = Axis(
    fig[1, 1],
    xlabel = "INLA posterior mean", ylabel = "Turing NUTS posterior mean",
    title = "Latent posterior means (n = $n)"
)
scatter!(ax, x_inla, x_nuts)
lims = extrema(vcat(x_inla, x_nuts))
lines!(ax, [lims...], [lims...], color = :gray, linestyle = :dash)
fig

# Quantitatively:
println("max |x_INLA − x_NUTS| = ", round(maximum(abs.(x_inla .- x_nuts)), digits = 3))
println("mean |x_INLA − x_NUTS| = ", round(mean(abs.(x_inla .- x_nuts)), digits = 3))

# ## Caveats
#
# Sharing one `@model` keeps the specification identical across engines, but a
# few things are worth knowing before you lean on the comparison:
#
# - Constrained GMRF priors (Besag ICAR with sum-to-zero, rank-deficient random
#   walks, and the like) do not sample cleanly under plain NUTS without a
#   reparameterisation onto the constraint manifold. NUTS samples the
#   unconstrained base density and ignores the constraint, which is not the
#   posterior you specified. For a constrained prior, `inla()`, `tmb()`, and
#   `hmc_laplace()` impose the constraint via Kriging conditioning.
# - Plain NUTS on a latent Gaussian model is usually slower than INLA, since
#   the posterior geometry is awkward for a generic sampler. In practice, run
#   `inla()` first and turn to `hmc_laplace()` when you want a sampled
#   hyperparameter posterior.
# - Heavy tails in hyperparameter marginals (a precision under weak data, say)
#   will differ between the two: INLA extrapolates the tail with a spline, while
#   NUTS only reflects what it visited. Compare medians and credible intervals
#   near the mode, where both are on firm ground.

# ## References
#
# ```@raw html
# <div class="ref-grid-2">
# <PaperCite
#   tag="Turing"
#   title="Turing: A Language for Flexible Probabilistic Inference"
#   authors="H. Ge, K. Xu & Z. Ghahramani"
#   venue="AISTATS (PMLR 84)" year="2018"
#   url="https://proceedings.mlr.press/v84/ge18b.html"
#   abstract="The Turing.jl probabilistic programming language: a flexible Julia system for MCMC-based Bayesian inference. Latte shares its DynamicPPL @model surface, so the same model hands off to Turing.sample." />
# <PaperCite
#   tag="DynamicPPL"
#   title="DynamicPPL: Stan-like Speed for Dynamic Probabilistic Models"
#   authors="M. Tarek, K. Xu, M. Trapp, H. Ge & Z. Ghahramani"
#   venue="arXiv preprint" year="2020"
#   arxiv="2002.02702"
#   url="https://arxiv.org/abs/2002.02702"
#   abstract="The modelling-language layer behind Turing and the @model surface Latte builds on: a performant runtime for dynamic probabilistic models." />
# <PaperCite
#   tag="INLA"
#   title="Approximate Bayesian Inference for Latent Gaussian Models by Using Integrated Nested Laplace Approximations"
#   authors="H. Rue, S. Martino & N. Chopin"
#   venue="J. R. Statist. Soc. B" year="2009"
#   doi="10.1111/j.1467-9868.2008.00700.x"
#   url="https://doi.org/10.1111/j.1467-9868.2008.00700.x"
#   abstract="The original INLA paper: deterministic approximate Bayesian inference for latent Gaussian models via nested Laplace approximations and numerical integration over the hyperparameters." />
# <PaperCite
#   tag="NUTS"
#   title="The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo"
#   authors="M. D. Hoffman & A. Gelman"
#   venue="Journal of Machine Learning Research" year="2014"
#   arxiv="1111.4246"
#   url="https://arxiv.org/abs/1111.4246"
#   abstract="The No-U-Turn Sampler, the adaptive Hamiltonian Monte Carlo algorithm Turing's NUTS sampler and Latte's HMC-Laplace engine both run." />
# <PaperCite
#   tag="Embedded Laplace + HMC"
#   title="Hamiltonian Monte Carlo using an Adjoint-differentiated Laplace Approximation"
#   authors="C. C. Margossian, A. Vehtari, D. Simpson & R. Agrawal"
#   venue="Advances in Neural Information Processing Systems (NeurIPS)" year="2020"
#   arxiv="2004.12550"
#   url="https://arxiv.org/abs/2004.12550"
#   abstract="Hamiltonian Monte Carlo over the hyperparameters, with the latent Gaussian field marginalised by an embedded Laplace approximation and the gradient propagated through that inner solve. The method Latte's hmc_laplace engine implements." />
# </div>
# ```
