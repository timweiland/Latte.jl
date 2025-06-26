using Distributions
using GaussianMarkovRandomFields

export INLAModel, latent_gmrf, log_joint_density

"""
    INLAModel{D, F, O}

A complete INLA model specification with hyperparameter prior, latent field prior, and observation model.

# Type Parameters
- `D <: Distribution`: Type of the hyperparameter prior distribution
- `F`: Type of the latent prior function (θ -> GMRF)  
- `O <: ObservationModel`: Type of the observation model

# Fields
- `hyperparameter_prior::D`: Prior distribution over hyperparameters θ
- `latent_prior::F`: Function mapping θ_named::NamedTuple -> GMRF for the latent field prior
- `observation_model::O`: Observation model linking observations to latent field

# Example
```julia
# Define hyperparameter prior
hp_prior = HyperparameterPrior((σ = InverseGamma(2, 1),))

# Define latent field prior (function of named hyperparameters)
function latent_gmrf(θ_named)
    σ = θ_named.σ
    n = 100
    Q = spdiagm(0 => fill(1/σ^2, n))  # Simple white noise
    μ = zeros(n)
    return GMRF(μ, Q, CholeskySolverBlueprint())
end

# Define observation model
obs_model = ExponentialFamily(Normal)

# Create INLA model
model = INLAModel(hp_prior, latent_gmrf, obs_model)
```
"""
struct INLAModel{HP, F, O<:ObservationModel}
    hyperparameter_prior::HP
    latent_prior::F
    observation_model::O
    
    function INLAModel(hp_prior::HyperparameterPrior{FreeNames, AllNames}, latent_prior::F, observation_model::O) where {FreeNames, AllNames, F, O<:ObservationModel}
        # Validation: check all required hyperparameters are provided (both free and fixed)
        required = Set(hyperparameters(observation_model))
        provided = Set(AllNames)  # Check all parameters (free + fixed)
        
        missing_params = setdiff(required, provided)
        if !isempty(missing_params)
            error("Missing required hyperparameters for $(typeof(observation_model)): $(collect(missing_params))")
        end
        
        new{typeof(hp_prior), F, O}(hp_prior, latent_prior, observation_model)
    end
end

"""
    latent_gmrf(model::INLAModel, θ_named)

Get the latent field GMRF for given hyperparameters θ_named.
"""
function latent_gmrf(model::INLAModel, θ_named)
    return model.latent_prior(θ_named)
end

"""
    log_joint_density(model::INLAModel, x, θ, y)

Evaluate the joint log-density log π(x, θ, y) for the INLA model.

This computes: log π(θ) + log π(x | θ) + log π(y | x, θ)
"""
function log_joint_density(model::INLAModel, x, θ, y)
    # Convert hyperparameter vector to named tuple for clean parameter access
    θ_named = to_named(θ, model.hyperparameter_prior)
    
    # Hyperparameter prior contribution
    log_prior_θ = logpdf(model.hyperparameter_prior.free_distribution, θ)
    
    # Latent field prior contribution  
    x_prior = latent_gmrf(model, θ_named)
    log_prior_x = logpdf(x_prior, x)
    
    # Observation model contribution
    log_likelihood = loglik(model.observation_model, x, θ_named, y)
    
    return log_prior_θ + log_prior_x + log_likelihood
end

"""
    Base.show(io::IO, model::INLAModel)

Pretty printing for INLA models.
"""
function Base.show(io::IO, model::INLAModel{D, F, O}) where {D, F, O}
    println(io, "INLAModel")
    println(io, "  Hyperparameter prior:\n    $(repr(model.hyperparameter_prior))")
    println(io, "  Latent prior function: ", typeof(model.latent_prior))
    println(io, "  Observation model: ", typeof(model.observation_model))
end
