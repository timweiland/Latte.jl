module IntegratedNestedLaplace

# Import observation model functionality from GaussianMarkovRandomFields.jl v0.4+
using GaussianMarkovRandomFields: ObservationModel, ObservationLikelihood,
    ExponentialFamily, BinomialObservations, PoissonObservations,
    LinearlyTransformedObservationModel,
    CompositeObservationModel, loglik, loggrad, loghessian, hyperparameters,
    latent_dimension, gaussian_approximation, successes, trials,
    IdentityLink, LogLink, LogitLink, LinkFunction,
    conditional_distribution, apply_link, apply_invlink

# Re-export observation model types and functions for user convenience
export ObservationModel, ObservationLikelihood, ExponentialFamily,
    BinomialObservations, PoissonObservations, LinearlyTransformedObservationModel,
    CompositeObservationModel, gaussian_approximation,
    loglik, loggrad, loghessian, hyperparameters, latent_dimension,
    successes, trials, IdentityLink, LogLink, LogitLink, LinkFunction,
    conditional_distribution, apply_link, apply_invlink

# Include INLA-specific modules
include("utils/selinv.jl")
include("utils/distribution_summaries.jl")
include("utils/plotting_stubs.jl")
include("hyperparameters/hyperparameters.jl")
include("latent_augmentation/latent_augmentation_module.jl")
include("inla_model.jl")
include("latent_marginalization/marginalization_module.jl")
include("hyperparameter_posterior/hyperparameter_posterior.jl")
include("distributions/distributions.jl")
include("observation_models/observation_models.jl")
include("posterior_accumulators/posterior_accumulators.jl")
include("main_interface/main_interface.jl")

end
