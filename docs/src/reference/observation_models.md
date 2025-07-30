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

## Composite Observation Models

Composite observation models allow you to combine multiple observation models to handle heterogeneous data within a single INLA inference. This is useful when you have different types of observations (e.g., continuous and count data) that should be modeled together.

### Basic Usage

```julia
using IntegratedNestedLaplace
using Distributions

# Create individual observation models
gaussian_model = ExponentialFamily(Normal, indices = 1:3)    # First 3 elements
poisson_model = ExponentialFamily(Poisson, indices = 4:6)    # Next 3 elements

# Create composite model
composite_model = CompositeObservationModel((gaussian_model, poisson_model))

# Prepare observation data
y_gaussian = [1.0, 2.0, 1.5]  # 3 Gaussian observations
y_poisson = [2, 3, 1]         # 3 Poisson observations
y_composite = CompositeObservations((y_gaussian, y_poisson))

# Materialize with hyperparameters
composite_lik = composite_model(y_composite; σ = 0.5)  # σ for Gaussian component

# Evaluate (automatically sums contributions from all components)
x = [0.5, 1.2, 0.8, 1.1, 0.9, 0.3]  # 6 latent field values
ll = loglik(composite_lik, x)
grad = loggrad(composite_lik, x)
hess = loghessian(composite_lik, x)
```

### Composite Data Structure

The `CompositeObservations` type combines multiple observation vectors while maintaining an `AbstractVector` interface:

```julia
# Create composite observations
y1 = [1.0, 2.0, 3.0]  # First component (3 observations)
y2 = [4.0, 5.0]       # Second component (2 observations)
y_composite = CompositeObservations((y1, y2))

# Acts like a regular vector
length(y_composite)    # 5
y_composite[1]         # 1.0
y_composite[4]         # 4.0
collect(y_composite)   # [1.0, 2.0, 3.0, 4.0, 5.0]
```

### Hyperparameter Handling

Each component observation model can have different hyperparameters. The composite model passes all provided hyperparameters to each component, and each component extracts what it needs:

```julia
# Multiple hyperparameters for different components
gaussian_model = ExponentialFamily(Normal, indices = 1:2)
binomial_model = ExponentialFamily(Binomial, indices = 3:4)
composite_model = CompositeObservationModel((gaussian_model, binomial_model))

y_composite = CompositeObservations(([1.0, 2.0], [7, 8]))

# Pass hyperparameters for both components
composite_lik = composite_model(y_composite; σ = 0.5, n = 10.0)
# σ is used by Gaussian component, n is used by Binomial component
```

### Advanced Example: Mixed Model Types

```julia
# Create a complex composite model with different distributions
normal_model = ExponentialFamily(Normal, IdentityLink(), indices = 1:2)
poisson_model = ExponentialFamily(Poisson, LogLink(), indices = 3:4) 
bernoulli_model = ExponentialFamily(Bernoulli, LogitLink(), indices = 5:6)

composite_model = CompositeObservationModel((normal_model, poisson_model, bernoulli_model))

# Heterogeneous observation data
y_normal = [0.5, -1.2]      # Continuous data
y_poisson = [3, 7]          # Count data  
y_bernoulli = [1, 0]        # Binary data
y_composite = CompositeObservations((y_normal, y_poisson, y_bernoulli))

# Materialize with appropriate hyperparameters
composite_lik = composite_model(y_composite; σ = 0.8)  # Only Normal needs σ

# Latent field values (different scales due to different link functions)
x = [0.5, -1.0,           # Identity scale for Normal
     log(4.0), log(8.0),  # Log scale for Poisson
     0.0, -1.0]            # Logit scale for Bernoulli

# Evaluate composite likelihood
ll = loglik(composite_lik, x)  # Sum of all component log-likelihoods
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
