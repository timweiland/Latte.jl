using ForwardDiff
using SparseDiffTools
using Symbolics
using SparseArrays

export ObservationModel, loglik, loggrad, loghessian, hyperparameters

"""
    ObservationModel

Abstract base type for all observation models in INLA.

An observation model defines the relationship between observations `y` and the latent field `x`,
typically through a likelihood function. All concrete observation models must implement the 
`loglik(model, x, θ, y)` method.

# Interface Requirements

Concrete subtypes must implement:
- `loglik(model::YourModel, x, θ, y)`: Log-likelihood function

Optional implementations for better performance:
- `loggrad(model::YourModel, x, θ, y)`: Gradient of log-likelihood w.r.t. `x`
- `loghessian(model::YourModel, x, θ, y)`: Hessian of log-likelihood w.r.t. `x`

If not implemented, automatic differentiation fallbacks are provided.

# Arguments for interface methods
- `x`: Latent field values (typically a vector)
- `θ`: Hyperparameters for the observation model (vector, can be empty)
- `y`: Observed data

# Example
```julia
struct CustomModel <: ObservationModel
    σ::Float64
end

function loglik(model::CustomModel, x, θ, y)
    return sum(logpdf.(Normal.(x, model.σ), y))
end
```

See also: [`ExponentialFamily`](@ref), [`loglik`](@ref), [`loggrad`](@ref), [`loghessian`](@ref)
"""
abstract type ObservationModel end

"""
    hyperparameters(obs_model::ObservationModel) -> Tuple{Vararg{Symbol}}

Return a tuple of required hyperparameter names for this observation model.

This method defines which hyperparameters the observation model expects to receive
in the hyperparameter vector θ when calling loglik, loggrad, or loghessian.

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

"""
    loglik(obs_model::ObservationModel, x, θ_named, y) -> Float64

Compute the log-likelihood of observations `y` given latent field `x` and hyperparameters `θ_named`.

This is the core method that all observation models must implement. It computes the 
log-likelihood function ℓ(x, θ; y) = log p(y | x, θ).

# Arguments
- `obs_model`: An observation model implementing the `ObservationModel` interface
- `x`: Latent field values (vector of length n)
- `θ_named`: Hyperparameters as a NamedTuple (e.g., `(σ = 0.5,)`)
- `y`: Observed data (vector of length n)

# Returns
- `Float64`: The log-likelihood value

# Example
```julia
model = ExponentialFamily(Poisson)
x = [1.0, 2.0, 0.5]        # Latent field (will be exponentiated for Poisson)
θ_named = NamedTuple()     # No hyperparameters for Poisson
y = [1, 3, 0]              # Count observations

ll = loglik(model, x, θ_named, y)
```

See also: [`loggrad`](@ref), [`loghessian`](@ref), [`likelihood`](@ref)
"""
function loglik(obs_model::ObservationModel, x, θ_named, y)
    error("loglik not implemented for $(typeof(obs_model))")
end

"""
    loggrad(obs_model::ObservationModel, x, θ_named, y) -> Vector{Float64}

Compute the gradient of the log-likelihood with respect to the latent field `x`.

This function computes ∇ₓ ℓ(x, θ; y) = ∇ₓ log p(y | x, θ). If not implemented by 
the specific observation model, an automatic differentiation fallback using 
ForwardDiff.jl is provided.

# Arguments
- `obs_model`: An observation model implementing the `ObservationModel` interface
- `x`: Latent field values (vector of length n)
- `θ_named`: Hyperparameters as a NamedTuple (e.g., `(σ = 0.5,)`)
- `y`: Observed data (vector of length n)

# Returns
- `Vector{Float64}`: Gradient vector of the same length as `x`

# Example
```julia
model = ExponentialFamily(Bernoulli)
x = [0.0, 1.0, -0.5]       # Latent field (logit scale)
θ_named = NamedTuple()     # No hyperparameters for Bernoulli
y = [0, 1, 0]              # Binary observations

grad = loggrad(model, x, θ_named, y)
```

# Performance Note
For better performance, observation models can provide specialized implementations.
The automatic differentiation fallback is convenient but may be slower for large problems.

See also: [`loglik`](@ref), [`loghessian`](@ref)
"""
function loggrad(obs_model::ObservationModel, x, θ_named, y)
    return ForwardDiff.gradient(xi -> loglik(obs_model, xi, θ_named, y), x)
end

"""
    loghessian(obs_model::ObservationModel, x, θ_named, y) -> AbstractMatrix{Float64}

Compute the Hessian matrix of the log-likelihood with respect to the latent field `x`.

This function computes ∇²ₓ ℓ(x, θ; y) = ∇²ₓ log p(y | x, θ). If not implemented by 
the specific observation model, an automatic differentiation fallback is provided
that attempts to exploit sparsity when possible.

# Arguments
- `obs_model`: An observation model implementing the `ObservationModel` interface
- `x`: Latent field values (vector of length n)
- `θ_named`: Hyperparameters as a NamedTuple (e.g., `(σ = 0.5,)`)
- `y`: Observed data (vector of length n)

# Returns
- `AbstractMatrix{Float64}`: Hessian matrix of size n×n

# Example
```julia
model = ExponentialFamily(Poisson)
x = [1.0, 2.0]             # Latent field
θ_named = NamedTuple()     # No hyperparameters
y = [1, 3]                 # Count observations

hess = loghessian(model, x, θ_named, y)
```

# Performance Note
The fallback implementation attempts to detect sparsity patterns using Symbolics.jl
and SparseDiffTools.jl for efficient computation. If sparsity detection fails,
it falls back to dense ForwardDiff.jl Hessian computation.

For independent observations, the Hessian is typically diagonal, which specialized
implementations can exploit for better performance.

See also: [`loglik`](@ref), [`loggrad`](@ref)
"""
function loghessian(obs_model::ObservationModel, x, θ_named, y)
    try
        f(xi) = loglik(obs_model, xi, θ_named, y)
        sparsity_pattern = Symbolics.hessian_sparsity(f, x)
        sparsity_pattern = Float64.(sparsity_pattern)
        colors = matrix_colors(sparsity_pattern)
        return forwarddiff_color_hessian(f, x, colors)
    catch e
        return ForwardDiff.hessian(xi -> loglik(obs_model, xi, θ_named, y), x)
    end
end
