module IntegratedNestedLaplace

include("hyperparameters/hyperparameters.jl")
include("observation_models/observation_models.jl")
include("gaussian_approximation/gaussian_approximation.jl")
include("marginalization/marginalization_module.jl")
include("inla_model.jl")
include("hyperparameter_posterior/hyperparameter_posterior.jl")
include("distributions/distributions.jl")

end
