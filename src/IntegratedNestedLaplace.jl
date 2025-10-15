module IntegratedNestedLaplace

# Import observation model functionality from GaussianMarkovRandomFields.jl v0.4+
using GaussianMarkovRandomFields: ObservationModel, ObservationLikelihood,
    ExponentialFamily, BinomialObservations, LinearlyTransformedObservationModel,
    CompositeObservationModel, loglik, loggrad, loghessian, hyperparameters,
    latent_dimension, gaussian_approximation, successes, trials,
    IdentityLink, LogLink, LogitLink, LinkFunction

# Re-export observation model types and functions for user convenience
export ObservationModel, ObservationLikelihood, ExponentialFamily,
    BinomialObservations, LinearlyTransformedObservationModel,
    CompositeObservationModel, gaussian_approximation,
    loglik, loggrad, loghessian, hyperparameters, latent_dimension,
    successes, trials, IdentityLink, LogLink, LogitLink

# Include INLA-specific modules
include("hyperparameters/hyperparameters.jl")
include("inla_model.jl")
include("latent_marginalization/marginalization_module.jl")
include("hyperparameter_posterior/hyperparameter_posterior.jl")
include("distributions/distributions.jl")
include("main_interface/main_interface.jl")

end
