using ForwardDiff
using SparseDiffTools
using Symbolics
using SparseArrays
using Random

export ObservationModel, ObservationLikelihood, loglik, loggrad, loghessian, hyperparameters

"""
    ObservationModel

Abstract base type for all observation models in INLA.

An observation model defines the relationship between observations `y` and the latent field `x`,
typically through a likelihood function. ObservationModel types serve as factories for creating
ObservationLikelihood instances via callable syntax.

# Usage Pattern
```julia
# Step 1: Create observation model (factory)
obs_model = ExponentialFamily(Normal)

# Step 2: Materialize with data and hyperparameters  
obs_lik = obs_model(y; σ=1.2)  # Creates ObservationLikelihood

# Step 3: Use materialized likelihood in hot loops
ll = loglik(obs_lik, x)  # Fast x-only evaluation
```

See also: [`ObservationLikelihood`](@ref), [`ExponentialFamily`](@ref)
"""
abstract type ObservationModel end

"""
    ObservationLikelihood

Abstract base type for materialized observation likelihoods.

Observation likelihoods are created by materializing an observation model with specific
hyperparameters θ and observed data y. They provide efficient evaluation methods that 
only depend on the latent field x, eliminating the need to repeatedly pass θ and y.

This design provides major performance benefits in optimization loops and cleaner 
automatic differentiation boundaries.

# Usage Pattern
```julia
# Step 1: Configure observation model (factory)
obs_model = ExponentialFamily(Normal)

# Step 2: Materialize with data and hyperparameters  
obs_lik = obs_model(y; σ=1.2)

# Step 3: Fast evaluation in hot loops
ll = loglik(obs_lik, x)      # Only x argument needed!
grad = loggrad(obs_lik, x)   # Fast x-only evaluation
```

"""
abstract type ObservationLikelihood end

"""
    hyperparameters(obs_model::ObservationModel) -> Tuple{Vararg{Symbol}}

Return a tuple of required hyperparameter names for this observation model.

This method defines which hyperparameters the observation model expects to receive
when materializing an ObservationLikelihood instance.

# Arguments  
- `obs_model`: An observation model implementing the `ObservationModel` interface

# Returns
- `Tuple{Vararg{Symbol}}`: Tuple of parameter names (e.g., `(:σ,)` or `(:α, :β)`)

# Example
```julia
hyperparameters(ExponentialFamily(Normal)) == (:σ,)
hyperparameters(ExponentialFamily(Bernoulli)) == ()
```

# Implementation
All observation models should implement this method. The default returns an empty tuple.
"""
hyperparameters(obs_model::ObservationModel) = ()


# =======================================================================================
# FALLBACK IMPLEMENTATIONS FOR NEW OBSERVATIONLIKELIHOOD API
# =======================================================================================

"""
    loggrad(obs_lik::ObservationLikelihood, x) -> Vector{Float64}

Automatic differentiation fallback for ObservationLikelihood gradient computation.
"""
function loggrad(obs_lik::ObservationLikelihood, x)
    return ForwardDiff.gradient(xi -> loglik(obs_lik, xi), x)
end

"""
    loghessian(obs_lik::ObservationLikelihood, x) -> AbstractMatrix{Float64}

Automatic differentiation fallback for ObservationLikelihood Hessian computation.
"""
function loghessian(obs_lik::ObservationLikelihood, x)
    try
        f(xi) = loglik(obs_lik, xi)
        sparsity_pattern = Symbolics.hessian_sparsity(f, x)
        sparsity_pattern = Float64.(sparsity_pattern)
        colors = matrix_colors(sparsity_pattern)
        return forwarddiff_color_hessian(f, x, colors)
    catch e
        return ForwardDiff.hessian(xi -> loglik(obs_lik, xi), x)
    end
end

"""
    Random.rand(rng::AbstractRNG, obs_model::ObservationModel; x, θ_named) -> Vector

Sample observations y from the observation model given latent field x and hyperparameters θ_named.

# Arguments
- `rng`: Random number generator
- `obs_model`: The observation model to sample from
- `x`: Latent field vector
- `θ_named`: Named tuple of hyperparameters

# Returns
- `y`: Vector of sampled observations, same length as x

Concrete observation model types should implement this method for efficient sampling.
"""
function Random.rand(rng::AbstractRNG, obs_model::ObservationModel; x, θ_named)
    error("Sampling not implemented for observation model type $(typeof(obs_model))")
end
