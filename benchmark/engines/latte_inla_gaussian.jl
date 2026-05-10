# Latte INLA engine — Gaussian latent marginalization (cheapest, no
# skew correction). Each x[i] marginal is approximated by the local
# Laplace Gaussian; tail asymmetry is ignored.

using Latte

const ENGINE_ID = "latte_inla_gaussian"
_LATENT_METHOD() = GaussianMarginal()

Base.include(@__MODULE__, joinpath(@__DIR__, "_latte_inla_core.jl"))
