# Canonical SBC test models, defined once and reused across testsets.
#
# Each distinct `@model` is a distinct type, so it triggers its own first-call
# compile of the inference pipeline. Most SBC testsets only vary the engine,
# seed, executor, or target — not the model — so they share these definitions
# rather than redefining an identical model per testset (which multiplied the
# block's compile cost). Testsets that exercise genuinely different paths
# (Gaussian likelihood, scalar/no latent, Gamma prior, the `@latte` recognition
# path) keep their own definitions where they are used.

using Latte
using DynamicPPL: @model
using Distributions
using GaussianMarkovRandomFields: IIDModel

# Poisson observations over an IID latent field with a PC prior on its precision.
@model function sbc_pois(y, n)
    τ ~ PCPrior.Precision(1.0, α = 0.01)
    x ~ IIDModel(n)(τ = τ)
    for i in eachindex(y)
        y[i] ~ Poisson(exp(x[i]); check_args = false)
    end
end

# As `sbc_pois`, plus a Normal fixed effect `β` in the linear predictor.
@model function sbc_pois_beta(y, n)
    τ ~ PCPrior.Precision(1.0, α = 0.01)
    β ~ Normal(0, 1)
    x ~ IIDModel(n)(τ = τ)
    for i in eachindex(y)
        y[i] ~ Poisson(exp(β + x[i]); check_args = false)
    end
end
