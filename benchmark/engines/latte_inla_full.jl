# Latte INLA engine — full LaplaceMarginal latent marginalization.
# Most expensive option, runs the full skew-corrected Laplace
# approximation on every x[i] regardless of whether the SKLD
# threshold would have triggered an upgrade in adaptive mode.

using Latte

const ENGINE_ID = "latte_inla_full"
_LATENT_METHOD() = LaplaceMarginal()

Base.include(@__MODULE__, joinpath(@__DIR__, "_latte_inla_core.jl"))
