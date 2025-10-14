using LinearAlgebra
using SparseArrays

export LinearlyTransformedObservationModel, LinearlyTransformedLikelihood

"""
    LinearlyTransformedObservationModel{M, A} <: ObservationModel

Observation model that applies a linear transformation to the latent field before 
passing to a base observation model. This enables GLM-style modeling with design 
matrices while maintaining full compatibility with existing observation models.

# Mathematical Foundation
The wrapper transforms the full latent field x_full to linear predictors η via a 
design matrix A:
- η = A * x_full  
- Base model operates on η as usual: p(y | η, θ)
- Chain rule applied for gradients/Hessians: 
  - ∇_{x_full} ℓ = A^T ∇_η ℓ
  - ∇²_{x_full} ℓ = A^T ∇²_η ℓ A

# Type Parameters
- `M <: ObservationModel`: Type of the base observation model
- `A`: Type of the design matrix (typically AbstractMatrix)

# Fields
- `base_model::M`: The underlying observation model that operates on linear predictors
- `design_matrix::A`: Matrix mapping full latent field to observation-specific linear predictors

# Usage Pattern
```julia
# Step 1: Create base observation model
base_model = ExponentialFamily(Poisson)  # LogLink by default

# Step 2: Create design matrix (maps latent field to linear predictors)
# For: y ~ intercept + temperature + group_effects
A = [1.0  20.0  1.0  0.0  0.0;   # obs 1: intercept + temp + group1
     1.0  25.0  1.0  0.0  0.0;   # obs 2: intercept + temp + group1  
     1.0  30.0  0.0  1.0  0.0;   # obs 3: intercept + temp + group2
     1.0  15.0  0.0  0.0  1.0]   # obs 4: intercept + temp + group3

# Step 3: Create wrapped model
obs_model = LinearlyTransformedObservationModel(base_model, A)

# Step 4: Use in INLAModel - latent field now includes all components
# x_full = [β₀, β₁, u₁, u₂, u₃]  # intercept, slope, group effects

# Step 5: Materialize with data and hyperparameters
obs_lik = obs_model(y; σ=1.2)  # Creates LinearlyTransformedLikelihood

# Step 6: Fast evaluation in optimization loops
ll = loglik(obs_lik, x_full)
```

# Hyperparameters
All hyperparameters come from the base observation model. The design matrix 
introduces no new hyperparameters - it's a fixed linear transformation.

See also: [`LinearlyTransformedLikelihood`](@ref), [`ExponentialFamily`](@ref), [`ObservationModel`](@ref)
"""
struct LinearlyTransformedObservationModel{M <: ObservationModel, A} <: ObservationModel
    base_model::M
    design_matrix::A

    function LinearlyTransformedObservationModel(base_model::M, design_matrix::A) where {M <: ObservationModel, A}
        # Validate that design matrix is appropriate
        if size(design_matrix, 1) == 0
            error("Design matrix must have at least one row (observation)")
        end
        if size(design_matrix, 2) == 0
            error("Design matrix must have at least one column (latent component)")
        end

        return new{M, A}(base_model, design_matrix)
    end
end

"""
    LinearlyTransformedLikelihood{L, A} <: ObservationLikelihood

Materialized likelihood for LinearlyTransformedObservationModel with precomputed 
base likelihood and design matrix.

This is created by calling a LinearlyTransformedObservationModel instance with 
data and hyperparameters, following the factory pattern used throughout the package.

# Type Parameters
- `L <: ObservationLikelihood`: Type of the materialized base likelihood
- `A`: Type of the design matrix

# Fields
- `base_likelihood::L`: Materialized base observation likelihood (contains y and θ)
- `design_matrix::A`: Design matrix mapping full latent field to linear predictors

# Usage
This type is typically created automatically:
```julia
ltom = LinearlyTransformedObservationModel(base_model, design_matrix)
ltlik = ltom(y; σ=1.2)  # Creates LinearlyTransformedLikelihood
ll = loglik(ltlik, x_full)  # Fast evaluation
```
"""
struct LinearlyTransformedLikelihood{L <: ObservationLikelihood, A} <: ObservationLikelihood
    base_likelihood::L
    design_matrix::A
end

# =======================================================================================
# FACTORY PATTERN: Make LinearlyTransformedObservationModel callable
# =======================================================================================

"""
    (ltom::LinearlyTransformedObservationModel)(y; kwargs...) -> LinearlyTransformedLikelihood

Factory method to create materialized LinearlyTransformedLikelihood.

Delegates hyperparameter handling to the base model, then wraps the result 
with the design matrix.

# Arguments
- `y`: Observed data
- `kwargs...`: Hyperparameters passed through to base model

# Returns
- `LinearlyTransformedLikelihood`: Materialized likelihood ready for fast evaluation
"""
function (ltom::LinearlyTransformedObservationModel)(y; kwargs...)
    # Create materialized base likelihood
    base_likelihood = ltom.base_model(y; kwargs...)

    # Wrap with design matrix
    return LinearlyTransformedLikelihood(base_likelihood, ltom.design_matrix)
end

# =======================================================================================
# HYPERPARAMETER INTERFACE DELEGATION
# =======================================================================================

"""
    hyperparameters(ltom::LinearlyTransformedObservationModel) -> Tuple{Vararg{Symbol}}

Return required hyperparameters by delegating to the base model.

The design matrix introduces no new hyperparameters - all parameters come from 
the base observation model.
"""
hyperparameters(ltom::LinearlyTransformedObservationModel) = hyperparameters(ltom.base_model)

"""
    latent_dimension(ltom::LinearlyTransformedObservationModel, y::AbstractVector) -> Int

Return the latent field dimension for a linearly transformed observation model.

The latent dimension is the number of columns in the design matrix, representing
the dimension of the full latent field (not the linear predictors).
"""
latent_dimension(ltom::LinearlyTransformedObservationModel, y::AbstractVector) = size(ltom.design_matrix, 2)

# =======================================================================================
# CORE LIKELIHOOD EVALUATION METHODS
# =======================================================================================

"""
    loglik(ltlik::LinearlyTransformedLikelihood, x_full) -> Float64

Evaluate log-likelihood for materialized LinearlyTransformedLikelihood.

This is the performance-critical method used in optimization loops. It applies the 
linear transformation η = A * x_full and delegates to the base likelihood.

# Arguments
- `ltlik`: MaterializedLinearlyTransformedLikelihood instance  
- `x_full`: Full latent field vector (length: n_latent_components)

# Returns
- `Float64`: Log-likelihood value

# Mathematical Details
Computes: log p(y | η, θ) where η = A * x_full
"""
function loglik(ltlik::LinearlyTransformedLikelihood, x_full)
    η = ltlik.design_matrix * x_full
    return loglik(ltlik.base_likelihood, η)
end

"""
    loggrad(ltlik::LinearlyTransformedLikelihood, x_full) -> Vector{Float64}

Compute gradient of log-likelihood with respect to full latent field using chain rule.

Applies the chain rule: ∇_{x_full} ℓ = A^T ∇_η ℓ where A is the design matrix
and ∇_η ℓ is the gradient computed by the base likelihood.

# Arguments
- `ltlik`: MaterializedLinearlyTransformedLikelihood instance
- `x_full`: Full latent field vector (length: n_latent_components)

# Returns
- `Vector{Float64}`: Gradient vector of same length as x_full

# Mathematical Details
1. Transform: η = A * x_full
2. Compute base gradient: grad_η = ∇_η log p(y | η, θ)  
3. Apply chain rule: grad_x = A^T * grad_η

# Performance Notes
- Preserves sparsity if A is sparse
- Efficiently handles both dense and sparse design matrices
- Reuses optimized gradient computations from base likelihood
"""
function loggrad(ltlik::LinearlyTransformedLikelihood, x_full)
    η = ltlik.design_matrix * x_full
    grad_η = loggrad(ltlik.base_likelihood, η)
    return ltlik.design_matrix' * grad_η  # Chain rule: A^T * grad_η
end

"""
    loghessian(ltlik::LinearlyTransformedLikelihood, x_full) -> AbstractMatrix{Float64}

Compute Hessian of log-likelihood with respect to full latent field using chain rule.

Applies the chain rule for Hessians: ∇²_{x_full} ℓ = A^T ∇²_η ℓ A where A is the 
design matrix and ∇²_η ℓ is the Hessian computed by the base likelihood.

# Arguments
- `ltlik`: MaterializedLinearlyTransformedLikelihood instance
- `x_full`: Full latent field vector (length: n_latent_components)

# Returns
- `AbstractMatrix{Float64}`: Hessian matrix of size (n_latent_components, n_latent_components)

# Mathematical Details
1. Transform: η = A * x_full
2. Compute base Hessian: hess_η = ∇²_η log p(y | η, θ)
3. Apply chain rule: hess_x = A^T * hess_η * A

# Performance Notes
- **Sparsity preservation**: If A and hess_η are sparse, result will be sparse
- **Efficient sandwich computation**: Uses optimized matrix multiplication
- **Memory scaling**: Result size is (n_latent, n_latent), larger than base (n_obs, n_obs)
- **Fill-in warning**: A^T * hess_η * A can create new non-zeros even if inputs are sparse

# Common Case Optimization
For diagonal base Hessians (common with canonical links), the computation
A^T * Diagonal(d) * A can be computed efficiently without forming the full diagonal matrix.
"""
function loghessian(ltlik::LinearlyTransformedLikelihood, x_full)
    η = ltlik.design_matrix * x_full
    hess_η = loghessian(ltlik.base_likelihood, η)
    A = ltlik.design_matrix

    # Chain rule: A^T * hess_η * A
    # This preserves sparsity patterns efficiently
    return A' * hess_η * A
end

# =======================================================================================
# SAMPLING INTERFACE
# =======================================================================================

"""
    Random.rand(rng::AbstractRNG, ltom::LinearlyTransformedObservationModel; x_full, θ_named) -> Vector

Sample observations from the LinearlyTransformedObservationModel.

Transforms the full latent field to linear predictors η = A * x_full, then 
delegates sampling to the base observation model.

# Arguments
- `rng`: Random number generator
- `ltom`: LinearlyTransformedObservationModel instance
- `x_full`: Full latent field vector (keyword argument)
- `θ_named`: Named tuple of hyperparameters (keyword argument)

# Returns
- `Vector`: Sampled observations of length n_observations

# Mathematical Details
1. Transform latent field: η = A * x_full
2. Sample from base model: y ~ base_model(η, θ_named)

# Usage
```julia
ltom = LinearlyTransformedObservationModel(base_model, design_matrix)
x_full = [β₀, β₁, u₁, u₂, u₃]  # Full latent field
θ_named = (σ = 1.2,)            # Hyperparameters
y = rand(ltom; x_full=x_full, θ_named=θ_named)
```
"""
function Random.rand(rng::AbstractRNG, ltom::LinearlyTransformedObservationModel; x_full, θ_named)
    η = ltom.design_matrix * x_full
    return Random.rand(rng, ltom.base_model; x = η, θ_named = θ_named)
end
