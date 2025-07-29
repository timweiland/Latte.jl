module IntegratedNestedLaplace

include("hyperparameters/hyperparameters.jl")
include("observation_models/observation_models.jl")
include("inla_model.jl")
include("gaussian_approximation/gaussian_approximation.jl")
include("latent_marginalization/marginalization_module.jl")
include("hyperparameter_posterior/hyperparameter_posterior.jl")
include("distributions/distributions.jl")
include("main_interface/main_interface.jl")

end
