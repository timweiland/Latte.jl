using Distributions
using GaussianMarkovRandomFields
using Random

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
    return GMRF(μ, Q)
end

# Define observation model
obs_model = ExponentialFamily(Normal)

# Create INLA model
model = INLAModel(hp_prior, latent_gmrf, obs_model)
```
"""
struct INLAModel{HP, F, O <: ObservationModel}
    hyperparameter_prior::HP
    latent_prior::F
    observation_model::O

    function INLAModel(hp_prior::HyperparameterPrior{FreeNames, AllNames}, latent_prior::F, observation_model::O) where {FreeNames, AllNames, F, O <: ObservationModel}
        # Validation: check all required hyperparameters are provided (both free and fixed)
        required = Set(hyperparameters(observation_model))
        provided = Set(AllNames)  # Check all parameters (free + fixed)

        missing_params = setdiff(required, provided)
        if !isempty(missing_params)
            error("Missing required hyperparameters for $(typeof(observation_model)): $(collect(missing_params))")
        end

        return new{typeof(hp_prior), F, O}(hp_prior, latent_prior, observation_model)
    end
end

"""
    latent_gmrf(model::INLAModel, θ_named)

Get the latent field GMRF for given hyperparameters θ_named.
"""
function latent_gmrf(model::INLAModel, θ_named)
    return model.latent_prior(; θ_named...)
end

"""
    log_joint_density(model::INLAModel, x, θ, y)

Evaluate the joint log-density log π(x, θ, y) for the INLA model.

This computes: log π(θ) + log π(x | θ) + log π(y | x, θ)

# Arguments
- `model::INLAModel`: The INLA model
- `x`: Latent field values
- `θ`: Hyperparameters (as Vector)
- `y`: Observations

This convenience method creates the latent GMRF and materializes the observation likelihood internally.
"""
function log_joint_density(model::INLAModel, x, θ, y)
    # Convert θ to named tuple
    θ_named = to_named(θ, model.hyperparameter_prior)

    # Create latent GMRF and materialized observation likelihood
    latent_prior = latent_gmrf(model, θ_named)
    obs_lik = model.observation_model(y; θ_named...)

    return log_joint_density(model, x, θ, latent_prior, obs_lik)
end

"""
    log_joint_density(model::INLAModel, x, θ, latent_gmrf, obs_lik)

Evaluate the joint log-density log π(x, θ, y) for the INLA model using precomputed components.

This computes: log π(θ) + log π(x | θ) + log π(y | x, θ)

# Arguments
- `model::INLAModel`: The INLA model
- `x`: Latent field values
- `θ`: Hyperparameters (as Vector)
- `latent_gmrf`: Precomputed latent field GMRF
- `obs_lik`: Precomputed materialized observation likelihood

This method is more efficient when the latent GMRF and observation likelihood are already available.
"""
function log_joint_density(model::INLAModel, x, θ, latent_gmrf, obs_lik)
    # Hyperparameter prior contribution
    log_prior_θ = logpdf(model.hyperparameter_prior.free_distribution, θ)

    # Latent field prior contribution
    log_prior_x = logpdf(latent_gmrf, x)

    # Observation model contribution
    log_likelihood = loglik(x, obs_lik)

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
    return println(io, "  Observation model: ", typeof(model.observation_model))
end

"""
    Random.rand([rng], model::INLAModel)

Sample from an INLAModel, returning a NamedTuple with hyperparameters θ, latent field x, and observations y.

The sampling process:
1. Sample hyperparameters θ from the hyperparameter prior
2. Generate the latent GMRF given θ and sample x from it  
3. Sample observations y given x using the observation model

# Arguments
- `rng`: Optional random number generator (defaults to global RNG)
- `model`: The INLAModel to sample from

# Returns
A NamedTuple `(θ = θ_vec, x = x_vec, y = y_vec)` where:
- `θ_vec`: Vector of free hyperparameter values
- `x_vec`: Vector of latent field values
- `y_vec`: Vector of observation values
"""
function Random.rand(rng::AbstractRNG, model::INLAModel)
    # Sample free hyperparameters
    θ_free = rand(rng, model.hyperparameter_prior.free_distribution)

    # Convert to full named tuple (free + fixed parameters)
    θ_named = to_named(θ_free, model.hyperparameter_prior)

    # Generate latent GMRF and sample from it
    gmrf = latent_gmrf(model, θ_named)
    x = rand(rng, gmrf)

    # Sample observations given latent field using GMRF's conditional_distribution
    y_dist = GaussianMarkovRandomFields.conditional_distribution(model.observation_model, x; θ_named...)
    y = rand(rng, y_dist)

    return (θ = θ_free, x = x, y = y)
end

# Default to global RNG
Random.rand(model::INLAModel) = rand(Random.default_rng(), model)
