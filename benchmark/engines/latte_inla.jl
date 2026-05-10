# Latte INLA engine — adaptive latent marginalization (default).
#
# Starts with `SimplifiedLaplace`, escalates each x[i] whose
# Gaussian-vs-SimplifiedLaplace SKLD exceeds the adaptive threshold
# to full `LaplaceMarginal`. Most accurate option that doesn't pay
# for full Laplace on every variable.

using Latte

const ENGINE_ID = "latte_inla"
_LATENT_METHOD() = AdaptiveMarginal()

Base.include(@__MODULE__, joinpath(@__DIR__, "_latte_inla_core.jl"))
