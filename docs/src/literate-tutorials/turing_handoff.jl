# # Handoff: same model, INLA and Turing
#
# Latte uses DynamicPPL's `@model` macro as its modelling surface. So does
# Turing.jl. That means a model you write for `inla()` can also be fed
# directly into `Turing.sample` for gold-standard MCMC — no rewriting.
#
# This tutorial shows the handoff on a small Poisson model with an
# IID-Normal latent field, and compares the posteriors.
#
# Why bother? A few reasons:
# - **Validation.** If you suspect INLA's Laplace approximation might be
#   misbehaving on an unusual model, a quick NUTS run on the same `@model`
#   is the most direct sanity check you can do.
# - **Choice of engine per problem.** INLA is fast but approximate. HMC is
#   slow but exact. Having both available with zero duplication lets you
#   pick per-model.
# - **Gradual exactness.** Latte also exposes `hmc_laplace(lgm, y)`, which
#   is full HMC on hyperparameters combined with an inner Laplace on the
#   latent. That's a nice middle ground between `inla` and full NUTS.
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

# ## Fit with INLA
#
# `latte_from_dppl` packages the DPPL model into a `LatentGaussianModel`
# that `inla()` understands — same object you'd use for `tmb()` or
# `hmc_laplace()`.
model = iid_poisson(y, n)
lgm = latte_from_dppl(model; random = (:x,))
result = inla(lgm, y; progress = false)

x_inla = [mean(m) for m in result.latent_marginals[1:n]]

# ## Fit with Turing NUTS
#
# No packaging needed — Turing consumes the `@model` directly.
chain = sample(model, NUTS(), 2000; progress = false)
x_nuts = [mean(chain[Symbol("x[$i]")]) for i in 1:n]

# ## Compare posteriors
#
# Latent posterior means from the two engines line up on the diagonal:
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
# The handoff is exact in the sense that the *same* `@model` is shared
# between engines. A few things to keep in mind if you rely on it:
#
# - **Constrained GMRF priors** (Besag ICAR with sum-to-zero, rank-deficient
#   random walks, etc.) don't sample cleanly under plain NUTS without a
#   reparameterisation onto the constraint manifold. NUTS will happily
#   sample the *unconstrained* base density, silently ignoring the
#   constraint, which breaks the posterior you actually wanted. If you
#   need a constrained prior, use `inla()` / `tmb()` / `hmc_laplace()` —
#   they handle the constraint correctly via Kriging conditioning.
# - **Speed.** Plain NUTS on a latent Gaussian model is typically much
#   slower than INLA, because the posterior geometry is bad for generic
#   samplers. That's literally the reason INLA exists. For production
#   use, reach for `inla()` first and `hmc_laplace()` when you need an
#   exact hyperparameter posterior.
# - **Tails of heavy-tailed hyperparameter marginals** (precisions on
#   weak data) will disagree between INLA and NUTS, because INLA
#   extrapolates via a spline while NUTS only samples what it visits.
#   Medians and credible intervals near the mode are the honest
#   cross-check.
