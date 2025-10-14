export ExponentialFamilyLikelihood, NormalLikelihood, PoissonLikelihood, BernoulliLikelihood, BinomialLikelihood

"""
    ExponentialFamilyLikelihood{L, I} <: ObservationLikelihood

Abstract type for exponential family observation likelihoods.

This intermediate type allows for generic implementations that work across all 
exponential family distributions while still allowing specialized methods for 
specific combinations.

# Type Parameters
- `L`: Link function type
- `I`: Index type (Nothing for non-indexed, UnitRange or Vector for indexed)
"""
abstract type ExponentialFamilyLikelihood{L, I} <: ObservationLikelihood end

"""
    NormalLikelihood{L<:LinkFunction} <: ObservationLikelihood

Materialized Normal observation likelihood with precomputed hyperparameters.

# Fields
- `link::L`: Link function connecting latent field to mean parameter
- `y::Vector{Float64}`: Observed data  
- `σ::Float64`: Standard deviation hyperparameter
- `inv_σ²::Float64`: Precomputed 1/σ² for performance
- `log_σ::Float64`: Precomputed log(σ) for log-likelihood computation

# Example
```julia
obs_model = ExponentialFamily(Normal)
obs_lik = obs_model([1.0, 2.0, 1.5]; σ=0.5)  # NormalLikelihood{IdentityLink}
ll = loglik(obs_lik, [0.9, 2.1, 1.4])
```
"""
struct NormalLikelihood{L <: LinkFunction, I} <: ExponentialFamilyLikelihood{L, I}
    link::L
    y::Vector{Float64}
    σ::Float64
    inv_σ²::Float64
    log_σ::Float64
    indices::I  # Can be Nothing, UnitRange, or Vector{Int}
end

"""
    PoissonLikelihood{L<:LinkFunction} <: ObservationLikelihood

Materialized Poisson observation likelihood.

# Fields  
- `link::L`: Link function connecting latent field to rate parameter
- `y::Vector{Int}`: Count observations

# Example
```julia
obs_model = ExponentialFamily(Poisson)  # Uses LogLink by default
obs_lik = obs_model([1, 3, 0, 2])      # PoissonLikelihood{LogLink}
ll = loglik(obs_lik, [0.0, 1.1, -2.0, 0.7])  # x values on log scale
```
"""
struct PoissonLikelihood{L <: LinkFunction, I} <: ExponentialFamilyLikelihood{L, I}
    link::L
    y::Vector{Int}
    indices::I  # Can be Nothing, UnitRange, or Vector{Int}
end

"""
    BernoulliLikelihood{L<:LinkFunction} <: ObservationLikelihood

Materialized Bernoulli observation likelihood for binary data.

# Fields
- `link::L`: Link function connecting latent field to probability parameter  
- `y::Vector{Int}`: Binary observations (0 or 1)

# Example
```julia
obs_model = ExponentialFamily(Bernoulli)  # Uses LogitLink by default
obs_lik = obs_model([1, 0, 1, 0])        # BernoulliLikelihood{LogitLink}
ll = loglik(obs_lik, [0.5, -0.2, 1.1, -0.8])  # x values on logit scale
```
"""
struct BernoulliLikelihood{L <: LinkFunction, I} <: ExponentialFamilyLikelihood{L, I}
    link::L
    y::Vector{Int}
    indices::I  # Can be Nothing, UnitRange, or Vector{Int}
end

"""
    BinomialLikelihood{L<:LinkFunction} <: ObservationLikelihood

Materialized Binomial observation likelihood.

# Fields
- `link::L`: Link function connecting latent field to probability parameter
- `y::Vector{Int}`: Number of successes for each trial
- `n::Vector{Int}`: Number of trials per observation (can vary across observations)

# Example  
```julia
obs_model = ExponentialFamily(Binomial)  # Uses LogitLink by default
obs_lik = obs_model([3, 1, 4]; trials=[5, 8, 6])  # BinomialLikelihood{LogitLink}
ll = loglik(obs_lik, [0.2, -1.0, 0.8])  # x values on logit scale
```
"""
struct BinomialLikelihood{L <: LinkFunction, I} <: ExponentialFamilyLikelihood{L, I}
    link::L
    y::Vector{Int}
    n::Vector{Int}  # Changed from Int to Vector{Int}
    indices::I  # Can be Nothing, UnitRange, or Vector{Int}
end
