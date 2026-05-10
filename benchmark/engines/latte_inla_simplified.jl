# Latte INLA engine — SimplifiedLaplace latent marginalization.
# First-order skew correction on every x[i]; cheaper than full
# Laplace, more accurate than the bare Gaussian approximation.

using Latte

const ENGINE_ID = "latte_inla_simplified"
_LATENT_METHOD() = SimplifiedLaplace()

Base.include(@__MODULE__, joinpath(@__DIR__, "_latte_inla_core.jl"))
