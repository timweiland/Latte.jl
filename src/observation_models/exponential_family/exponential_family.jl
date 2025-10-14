using Distributions
using Distributions: product_distribution
using Random

export ExponentialFamily, data_distribution

"""
    ExponentialFamily{F<:Distribution, L<:LinkFunction} <: ObservationModel

Observation model for exponential family distributions with link functions.

This struct represents observation models where the observations come from an exponential 
family distribution (Normal, Poisson, Bernoulli, Binomial) and the mean parameter is 
related to the latent field through a link function.

# Mathematical Model
For observations yᵢ and latent field values xᵢ:
- Linear predictor: ηᵢ = xᵢ
- Mean parameter: μᵢ = g⁻¹(ηᵢ) where g is the link function
- Observations: yᵢ ~ F(μᵢ, θ) where F is the distribution family

# Fields
- `family::Type{F}`: The distribution family (e.g., `Poisson`, `Bernoulli`)
- `link::L`: The link function connecting mean parameters to linear predictors

# Type Parameters
- `F`: A subtype of `Distribution` from Distributions.jl
- `L`: A subtype of `LinkFunction`

# Constructors
```julia
# Use canonical link (recommended)
ExponentialFamily(Poisson)        # Uses LogLink()
ExponentialFamily(Bernoulli)      # Uses LogitLink()
ExponentialFamily(Normal)         # Uses IdentityLink()

# Specify custom link function
ExponentialFamily(Poisson, IdentityLink())  # Non-canonical
```

# Supported Combinations
- `Normal` with `IdentityLink` (canonical) or `LogLink`
- `Poisson` with `LogLink` (canonical) or `IdentityLink`  
- `Bernoulli` with `LogitLink` (canonical) or `LogLink`
- `Binomial` with `LogitLink` (canonical) or `IdentityLink`

# Hyperparameters (θ)
Different families require different hyperparameters:
- `Normal`: `θ = [σ]` (standard deviation)
- `Poisson`: `θ = []` (no hyperparameters)
- `Bernoulli`: `θ = []` (no hyperparameters)
- `Binomial`: `θ = [n]` (number of trials)

# Examples
```julia
# Poisson model for count data
model = ExponentialFamily(Poisson)
x = [1.0, 2.0]        # Latent field (log scale due to LogLink)
θ = Float64[]         # No hyperparameters  
y = [2, 7]           # Count observations

ll = loglik(model, x, θ, y)
dist = data_distribution(model, x, θ)  # Returns Product distribution

# Bernoulli model for binary data
model = ExponentialFamily(Bernoulli)
x = [0.0, 1.0]       # Latent field (logit scale due to LogitLink)
y = [0, 1]           # Binary observations
```

# Performance Notes
Canonical link functions have optimized implementations that avoid redundant computations.
Non-canonical links use general chain rule formulations which may be slower.

See also: [`LinkFunction`](@ref), [`loglik`](@ref), [`data_distribution`](@ref)
"""
struct ExponentialFamily{F <: Distribution, L <: LinkFunction, I} <: ObservationModel
    family::Type{F}
    link::L
    indices::I  # Can be Nothing, UnitRange, or Vector{Int}
end

"""
    ExponentialFamily(family::Type{<:Distribution}) -> ExponentialFamily

Create an exponential family observation model with the canonical link function.

This constructor automatically selects the appropriate canonical link function for 
the given distribution family:
- `Normal` → `IdentityLink()`
- `Poisson` → `LogLink()`  
- `Bernoulli` → `LogitLink()`
- `Binomial` → `LogitLink()`

# Arguments
- `family`: A distribution type from Distributions.jl

# Returns
An `ExponentialFamily` instance with the canonical link function

# Examples
```julia
poisson_model = ExponentialFamily(Poisson)    # Uses LogLink
normal_model = ExponentialFamily(Normal)      # Uses IdentityLink
bernoulli_model = ExponentialFamily(Bernoulli) # Uses LogitLink
```

See also: [`ExponentialFamily`](@ref)
"""
# Main constructor with optional indices
ExponentialFamily(family::Type{<:Distribution}; indices = nothing) = ExponentialFamily(family, _default_link(family), indices)

# Constructor with explicit link function and optional indices
ExponentialFamily(family::Type{<:Distribution}, link::LinkFunction; indices = nothing) = ExponentialFamily(family, link, indices)

_default_link(::Type{<:Normal}) = IdentityLink()
_default_link(::Type{<:Poisson}) = LogLink()
_default_link(::Type{<:Bernoulli}) = LogitLink()
_default_link(::Type{<:Binomial}) = LogitLink()

"""
    data_distribution(obs_model::ExponentialFamily, x, θ_named) -> Distribution

Construct the data-generating distribution p(y | x, θ).

This function returns a Distribution object that represents the probability 
distribution over observations y given latent field values x and hyperparameters θ.
It is used for sampling new observations.

# Arguments
- `obs_model`: An ExponentialFamily observation model
- `x`: Latent field values (vector)  
- `θ_named`: Hyperparameters as a NamedTuple

# Returns
Distribution object that can be used with `rand()` to generate observations

# Example
```julia
model = ExponentialFamily(Poisson)
x = [1.0, 2.0]
θ_named = NamedTuple()
dist = data_distribution(model, x, θ_named)
y = rand(dist)  # Sample observations
```

Note: For likelihood evaluation L(x|y,θ), use the materialized API:
```julia
obs_lik = obs_model(y; θ_named...)
ll = loglik(obs_lik, x)
```
"""
function data_distribution(obs_model::ExponentialFamily, x, θ_named)
    η = x  # Linear predictor
    μ = apply_invlink.(Ref(obs_model.link), η)
    return _data_distribution_family(obs_model.family, μ, θ_named)
end

function _data_distribution_family(::Type{<:Normal}, μ, θ_named)
    σ = θ_named.σ  # Extract standard deviation
    return product_distribution(Normal.(μ, σ))
end

function _data_distribution_family(::Type{<:Poisson}, μ, θ_named)
    return product_distribution(Poisson.(μ))
end

function _data_distribution_family(::Type{<:Bernoulli}, μ, θ_named)
    return product_distribution(Bernoulli.(μ))
end

function _data_distribution_family(::Type{<:Binomial}, μ, θ_named)
    n = θ_named.n  # Extract number of trials
    return product_distribution(Binomial.(n, μ))
end


# =======================================================================================
# FACTORY PATTERN: Make ExponentialFamily callable to create materialized likelihoods
# =======================================================================================

"""
    (obs_model::ExponentialFamily{Normal, L})(y; σ, kwargs...) -> NormalLikelihood{L}

Create a materialized Normal likelihood with precomputed hyperparameters.

# Arguments
- `y`: Observed data
- `σ`: Standard deviation hyperparameter
- `kwargs...`: Additional keyword arguments (ignored)

# Returns
A `NormalLikelihood` with the same link function as the factory model.
"""
function (obs_model::ExponentialFamily{Normal, L, I})(y; σ, kwargs...) where {L, I}
    return NormalLikelihood(obs_model.link, Float64.(y), Float64(σ), 1.0 / (σ^2), log(σ), obs_model.indices)
end

"""
    (obs_model::ExponentialFamily{Poisson, L})(y; kwargs...) -> PoissonLikelihood{L}

Create a materialized Poisson likelihood.

# Arguments  
- `y`: Count observations
- `kwargs...`: Additional keyword arguments (ignored)

# Returns
A `PoissonLikelihood` with the same link function as the factory model.
"""
function (obs_model::ExponentialFamily{Poisson, L, I})(y; kwargs...) where {L, I}
    return PoissonLikelihood(obs_model.link, Int.(y), obs_model.indices)
end

"""
    (obs_model::ExponentialFamily{Bernoulli, L})(y; kwargs...) -> BernoulliLikelihood{L}

Create a materialized Bernoulli likelihood.

# Arguments
- `y`: Binary observations (0 or 1)
- `kwargs...`: Additional keyword arguments (ignored)

# Returns  
A `BernoulliLikelihood` with the same link function as the factory model.
"""
function (obs_model::ExponentialFamily{Bernoulli, L, I})(y; kwargs...) where {L, I}
    return BernoulliLikelihood(obs_model.link, Int.(y), obs_model.indices)
end

"""
    (obs_model::ExponentialFamily{Binomial, L})(y::BinomialObservations; kwargs...) -> BinomialLikelihood{L}

Create a materialized Binomial likelihood from BinomialObservations.

# Arguments
- `y::BinomialObservations`: Combined successes and trials data
- `kwargs...`: Additional keyword arguments (ignored)

# Returns
A `BinomialLikelihood` with the same link function as the factory model.

# Example
```julia
# Create BinomialObservations and use with model
y = BinomialObservations([3, 1, 4], [5, 8, 6])
obs_lik = obs_model(y)
```
"""
function (obs_model::ExponentialFamily{Binomial, L, I})(y::BinomialObservations; kwargs...) where {L, I}
    return BinomialLikelihood(obs_model.link, successes(y), trials(y), obs_model.indices)
end

# Hyperparameter interface implementations
hyperparameters(::ExponentialFamily{<:Normal}) = (:σ,)
hyperparameters(::ExponentialFamily{<:Bernoulli}) = ()
hyperparameters(::ExponentialFamily{<:Binomial}) = ()  # No hyperparameters - trials are data
hyperparameters(::ExponentialFamily{<:Poisson}) = ()

"""
    latent_dimension(ef::ExponentialFamily, y::AbstractVector) -> Int

Return the latent field dimension for exponential family models.

For ExponentialFamily models, there is a direct 1:1 mapping between observations
and latent field components, so the latent dimension equals the observation dimension.
"""
latent_dimension(ef::ExponentialFamily, y::AbstractVector) = length(y)

# Sampling interface implementation
function Random.rand(rng::AbstractRNG, obs_model::ExponentialFamily; x, θ_named)
    dist = data_distribution(obs_model, x, θ_named)
    return rand(rng, dist)
end
