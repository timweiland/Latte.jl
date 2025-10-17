using Distributions
using GaussianMarkovRandomFields
using Random

export INLAModel, latent_gmrf, log_joint_density

"""
    INLAModel{HP, F, O}

A complete INLA model specification with hyperparameter prior, latent field prior, and observation model.

# Type Parameters
- `HP`: Type of the hyperparameter specification (HyperparameterSpec)
- `F`: Type of the latent prior function (θ -> GMRF)
- `O <: ObservationModel`: Type of the observation model

# Fields
- `hyperparameter_spec::HP`: Hyperparameter specification with transformations
- `latent_prior::F`: Function mapping θ_named::NamedTuple -> GMRF for the latent field prior (receives natural-space parameters)
- `observation_model::O`: Observation model linking observations to latent field

# Example
```julia
using Bijectors

# Define hyperparameter specification
hp_spec = HyperparameterSpec(
    free = [Hyperparameter(:σ, Exponential(1.0), transform=elementwise(log), prior_space=:natural)],
    fixed = NamedTuple()
)

# Define latent field prior (function of named hyperparameters in natural space)
function latent_gmrf(; σ, kwargs...)
    n = 100
    Q = spdiagm(0 => fill(1/σ^2, n))  # Simple white noise
    μ = zeros(n)
    return GMRF(μ, Q)
end

# Define observation model
obs_model = ExponentialFamily(Normal)

# Create INLA model
model = INLAModel(hp_spec, latent_gmrf, obs_model)
```
"""
struct INLAModel{HP, F, O <: ObservationModel}
    hyperparameter_spec::HP
    latent_prior::F
    observation_model::O

    function INLAModel(hp_spec::HyperparameterSpec{FreeNT, FixedNT}, latent_prior::F, observation_model::O) where {FreeNT, FixedNT, F, O <: ObservationModel}
        # Validation: check all required hyperparameters are provided (both free and fixed)
        required = Set(hyperparameters(observation_model))
        provided = Set(fieldnames(FreeNT)) ∪ Set(fieldnames(FixedNT))
        #provided = Set(AllNames)  # Check all parameters (free + fixed)

        missing_params = setdiff(required, provided)
        if !isempty(missing_params)
            error("Missing required hyperparameters for $(typeof(observation_model)): $(collect(missing_params))")
        end

        return new{typeof(hp_spec), F, O}(hp_spec, latent_prior, observation_model)
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
- `θ`: Hyperparameters (as Vector, in working space for HyperparameterSpec)
- `y`: Observations

This convenience method creates the latent GMRF and materializes the observation likelihood internally.
"""
function log_joint_density(model::INLAModel, x, θ, y)
    spec = model.hyperparameter_spec

    # Convert θ vector to named tuples
    θ_working = to_named_tuple(θ, spec)
    θ_natural = to_natural(θ_working, spec)  # Includes fixed parameters

    # Create latent GMRF and materialized observation likelihood in natural space
    latent_prior = latent_gmrf(model, θ_natural)
    obs_lik = model.observation_model(y; θ_natural...)

    return log_joint_density(model, x, θ, latent_prior, obs_lik)
end

"""
    log_joint_density(model::INLAModel, x, θ, latent_gmrf, obs_lik)

Evaluate the joint log-density log π(x, θ, y) for the INLA model using precomputed components.

This computes: log π(θ) + log π(x | θ) + log π(y | x, θ)

# Arguments
- `model::INLAModel`: The INLA model
- `x`: Latent field values
- `θ`: Hyperparameters (as Vector, in working space for HyperparameterSpec)
- `latent_gmrf`: Precomputed latent field GMRF
- `obs_lik`: Precomputed materialized observation likelihood

This method is more efficient when the latent GMRF and observation likelihood are already available.
"""
function log_joint_density(model::INLAModel, x, θ, latent_gmrf, obs_lik)
    spec = model.hyperparameter_spec

    # Hyperparameter prior contribution (in working space with Jacobian)
    θ_working = to_named_tuple(θ, spec)
    log_prior_θ = logpdf_prior(θ_working, spec)

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
    println(io, "  Hyperparameter spec:\n    $(repr(model.hyperparameter_spec))")
    println(io, "  Latent prior function: ", typeof(model.latent_prior))
    return println(io, "  Observation model: ", typeof(model.observation_model))
end

"""
    Random.rand([rng], model::INLAModel)

Sample from an INLAModel, returning a NamedTuple with hyperparameters θ, latent field x, and observations y.

The sampling process:
1. Sample hyperparameters θ from the hyperparameter prior (in working space internally)
2. Convert to natural space and generate the latent GMRF
3. Sample x from the latent GMRF
4. Sample observations y given x using the observation model

# Arguments
- `rng`: Optional random number generator (defaults to global RNG)
- `model`: The INLAModel to sample from

# Returns
A NamedTuple `(θ = θ_named, x = x_vec, y = y_vec)` where:
- `θ_named`: NamedTuple of hyperparameter values in natural space (what users should see)
- `x_vec`: Vector of latent field values
- `y_vec`: Vector of observation values
"""
function Random.rand(rng::AbstractRNG, model::INLAModel)
    spec = model.hyperparameter_spec

    # Sample free hyperparameters in working space
    θ_working_vec = [rand(rng, hp.prior) for hp in values(spec.free)]

    # Convert to named tuples
    θ_working = to_named_tuple(θ_working_vec, spec)
    θ_natural = to_natural(θ_working, spec)  # Includes fixed parameters - this is what user sees

    # Generate latent GMRF and sample from it
    gmrf = latent_gmrf(model, θ_natural)
    x = rand(rng, gmrf)

    # Sample observations given latent field using GMRF's conditional_distribution
    y_dist = GaussianMarkovRandomFields.conditional_distribution(model.observation_model, x; θ_natural...)
    y = rand(rng, y_dist)

    return (θ = θ_natural, x = x, y = y)
end

# Default to global RNG
Random.rand(model::INLAModel) = rand(Random.default_rng(), model)
