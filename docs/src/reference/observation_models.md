# [Observation Models](@id observation-models)

Observation models define the relationship between observations and the latent field in INLA. They specify how observations `y` are generated from latent field values `x` through probability distributions and link functions.

## Overview

The observation model interface provides a flexible framework for connecting observations to latent fields. The design uses a **factory pattern** for efficiency:

1. **Create observation model template**: Define the distribution and link function with `ExponentialFamily(Distribution, LinkFunction)`
2. **Materialize with data**: Create observation likelihood with `obs_lik = obs_model(y; θ_named...)`
3. **Fast evaluation**: Compute log-likelihood efficiently with `loglik(obs_lik, x)`

This approach eliminates redundant parameter passing and provides significant performance benefits by pre-materializing the observation likelihood with data and hyperparameters.

The package provides built-in support for exponential family distributions through the [`ExponentialFamily`](@ref) struct, which covers most common use cases in statistical modeling.

## Basic Usage

```julia
using IntegratedNestedLaplace
using Distributions

# Create a Poisson observation model (canonical link)
obs_model = ExponentialFamily(Poisson)

# Observations and latent field values
y = [2, 7, 1]  # Count observations  
x = [1.0, 2.0, 0.5]  # Latent field values (log scale due to LogLink)

# Materialize observation likelihood with data (no hyperparameters for Poisson)
obs_lik = obs_model(y)

# Fast evaluation using materialized likelihood
ll = loglik(obs_lik, x)      # Log-likelihood
grad = loggrad(obs_lik, x)   # Gradient w.r.t. x
hess = loghessian(obs_lik, x) # Hessian w.r.t. x

# Get data distribution as Distribution object
dist = data_distribution(obs_lik, x)
samples = rand(dist, 100)    # Generate synthetic data
```

## Exponential Family Models

The [`ExponentialFamily`](@ref) struct provides ready-to-use models for standard distributions:

### Supported Distributions

| Distribution | Canonical Link | Hyperparameters | Use Case |
|:-------------|:---------------|:----------------|:---------|
| [`Normal`](https://juliastats.org/Distributions.jl/stable/univariate/#Distributions.Normal) | [`IdentityLink`](@ref) | `θ = [σ]` | Continuous data |
| [`Poisson`](https://juliastats.org/Distributions.jl/stable/univariate/#Distributions.Poisson) | [`LogLink`](@ref) | `θ = []` | Count data |
| [`Bernoulli`](https://juliastats.org/Distributions.jl/stable/univariate/#Distributions.Bernoulli) | [`LogitLink`](@ref) | `θ = []` | Binary data |
| [`Binomial`](https://juliastats.org/Distributions.jl/stable/univariate/#Distributions.Binomial) | [`LogitLink`](@ref) | `θ = [n]` | Binomial data |

### Examples by Distribution

#### Poisson Model (Count Data)
```julia
obs_model = ExponentialFamily(Poisson)  # Uses LogLink automatically

# Observations and latent field
y = [1, 4, 2]                       # Count observations
x = [log(2.0), log(5.0), log(1.0)]  # Latent field on log scale (rates [2, 5, 1])

# Materialize and evaluate
obs_lik = obs_model(y)  # No hyperparameters
ll = loglik(obs_lik, x)
```

#### Bernoulli Model (Binary Data)  
```julia
obs_model = ExponentialFamily(Bernoulli)  # Uses LogitLink automatically

# Observations and latent field  
y = [0, 1, 0]         # Binary observations
x = [0.0, 1.0, -1.0]  # Latent field on logit scale (probabilities [0.5, 0.73, 0.27])

# Materialize and evaluate
obs_lik = obs_model(y)  # No hyperparameters
ll = loglik(obs_lik, x)
```

#### Normal Model (Continuous Data)
```julia
obs_model = ExponentialFamily(Normal)  # Uses IdentityLink automatically

# Observations and latent field
y = [0.1, 1.2, -0.4]  # Continuous observations
x = [0.0, 1.0, -0.5]  # Latent field directly as means

# Materialize with hyperparameter
obs_lik = obs_model(y; σ = 0.5)  # Standard deviation
ll = loglik(obs_lik, x)
```

#### Binomial Model
```julia
obs_model = ExponentialFamily(Binomial)  # Uses LogitLink automatically

# Observations and latent field
y = [5, 7]     # Number of successes
x = [0.0, 0.5] # Latent field on logit scale

# Materialize with hyperparameter
obs_lik = obs_model(y; n = 10.0)  # Number of trials
ll = loglik(obs_lik, x)
```

### Non-Canonical Links

You can specify custom link functions for specialized applications:

```julia
# Poisson with identity link (non-canonical)
# Requires x values to be positive
obs_model = ExponentialFamily(Poisson, IdentityLink())
y = [1, 4, 2]
x = [2.0, 5.0, 1.0]  # Directly as rates (must be positive)
obs_lik = obs_model(y)
ll = loglik(obs_lik, x)

# Bernoulli with log link (non-canonical)  
# Requires x values such that exp(x) ∈ (0,1)
obs_model = ExponentialFamily(Bernoulli, LogLink())
y = [0, 1]
x = [log(0.3), log(0.7)]  # log-probabilities
obs_lik = obs_model(y)
ll = loglik(obs_lik, x)
```

## Link Functions

Link functions `g(μ)` connect the mean parameter μ to the linear predictor η via the relationship μ = g⁻¹(η).

### Available Link Functions

```@docs
IdentityLink
LogLink  
LogitLink
```

### Working with Link Functions

```julia
# Create link functions
identity = IdentityLink()
log_link = LogLink()
logit = LogitLink()

# Forward transformation: μ → η
η₁ = apply_link(log_link, 2.718)    # ≈ 1.0
η₂ = apply_link(logit, 0.5)         # = 0.0

# Inverse transformation: η → μ  
μ₁ = apply_invlink(log_link, 1.0)   # ≈ 2.718
μ₂ = apply_invlink(logit, 0.0)      # = 0.5
```

## Data Distributions

The [`data_distribution`](@ref) function returns Distribution objects compatible with Distributions.jl:

```julia
obs_model = ExponentialFamily(Poisson)
y = [2, 7]  # Observations 
x = [1.0, 2.0]  # Latent field values

# Materialize observation likelihood
obs_lik = obs_model(y)

# Get data distribution given latent field
dist = data_distribution(obs_lik, x)

# Use with Distributions.jl
synthetic_data = rand(dist, 100)    # Generate samples
mean_val = mean(dist)               # Compute moments  
var_val = var(dist)
prob = logpdf(dist, y)              # Evaluate data probability

# Equivalent to direct loglik call
prob2 = loglik(obs_lik, x)  # prob ≈ prob2
```

## Custom Observation Models

For specialized applications, you can implement custom observation models:

### Basic Custom Model

```julia
# Define custom struct
struct CustomNormalModel <: ObservationModel
    σ::Float64
end

# Declare hyperparameters (none in this example)
hyperparameters(::CustomNormalModel) = ()

# Implement factory pattern: callable syntax returns materialized likelihood
function (model::CustomNormalModel)(y; kwargs...)
    return MaterializedCustomModel(y, model.σ)
end

# Materialized likelihood struct
struct MaterializedCustomModel{Y}
    y::Y
    σ::Float64
end

# Implement loglik for materialized version
function loglik(obs_lik::MaterializedCustomModel, x)
    return sum(logpdf.(Normal.(x, obs_lik.σ), obs_lik.y))
end

# Usage
obs_model = CustomNormalModel(0.5)
y = [0.1, 1.2, -0.4]
obs_lik = obs_model(y)  # Materialize
x = [0.0, 1.0, -0.5]
ll = loglik(obs_lik, x)
grad = loggrad(obs_lik, x)  # Automatic differentiation
```

### Optimized Custom Model

For better performance, you can provide specialized gradient and Hessian implementations:

```julia
struct OptimizedModel <: ObservationModel
    σ²::Float64
end

# Declare hyperparameters (none in this example)
hyperparameters(::OptimizedModel) = ()

# Factory pattern implementation
function (model::OptimizedModel)(y; kwargs...)
    return MaterializedOptimizedModel(y, model.σ²)
end

struct MaterializedOptimizedModel{Y}
    y::Y
    σ²::Float64
end

# Optimized implementations for materialized likelihood
function loglik(obs_lik::MaterializedOptimizedModel, x)
    y, σ² = obs_lik.y, obs_lik.σ²
    return -0.5 * sum((y .- x).^2) / σ² - 0.5 * length(y) * log(2π * σ²)
end

function loggrad(obs_lik::MaterializedOptimizedModel, x)
    y, σ² = obs_lik.y, obs_lik.σ²
    return (y .- x) ./ σ²  # Analytical gradient
end

function loghessian(obs_lik::MaterializedOptimizedModel, x)
    σ² = obs_lik.σ²
    return Diagonal(-ones(length(x)) ./ σ²)  # Analytical Hessian
end
```

## Performance Considerations

### Canonical vs Non-Canonical Links

- **Canonical links** (default choices) have optimized implementations that avoid redundant computations
- **Non-canonical links** use general chain rule formulations which may be slower
- For large problems, prefer canonical links when possible

### Automatic Differentiation

The package provides intelligent AD fallbacks:

1. **Sparsity detection**: Attempts to detect sparse Hessian patterns using Symbolics.jl
2. **Colored differentiation**: Uses SparseDiffTools.jl for efficient sparse AD when possible  
3. **Dense fallback**: Falls back to ForwardDiff.jl for dense computation

For custom models with known structure, providing analytical derivatives can significantly improve performance.

## API Reference

### Core Interface

```@docs
ObservationModel
ObservationLikelihood
hyperparameters
loglik
loggrad
loghessian
```

### Exponential Family

```@docs
ExponentialFamily
data_distribution
```

### Link Functions

```@docs
LinkFunction
apply_link
apply_invlink
```
