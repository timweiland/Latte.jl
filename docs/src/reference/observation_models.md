# [Observation Models](@id observation-models)

Observation models define the relationship between observations and the latent field in INLA. They specify how observations `y` are generated from latent field values `x` through probability distributions and link functions.

## Overview

The observation model interface provides a flexible framework for connecting observations to latent fields. All observation models implement the [`ObservationModel`](@ref) interface, which requires:

- **Log-likelihood computation**: `loglik(model, x, θ, y)` 
- **Automatic differentiation fallbacks**: For gradient and Hessian computation

The package provides built-in support for exponential family distributions through the [`ExponentialFamily`](@ref) struct, which covers most common use cases in statistical modeling.

## Basic Usage

```julia
using IntegratedNestedLaplace
using Distributions

# Create a Poisson observation model (canonical link)
model = ExponentialFamily(Poisson)

# Latent field values (log scale due to LogLink)
x = [1.0, 2.0, 0.5]
θ = Float64[]  # No hyperparameters for Poisson
y = [2, 7, 1]  # Count observations

# Core interface methods
ll = loglik(model, x, θ, y)      # Log-likelihood
grad = loggrad(model, x, θ, y)   # Gradient w.r.t. x
hess = loghessian(model, x, θ, y) # Hessian w.r.t. x

# Get likelihood as Distribution object
dist = likelihood(model, x, θ)
samples = rand(dist, 100)        # Generate synthetic data
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
model = ExponentialFamily(Poisson)  # Uses LogLink automatically

# Latent field on log scale
x = [log(2.0), log(5.0), log(1.0)]  # Corresponds to rates [2, 5, 1]
y = [1, 4, 2]                       # Count observations

ll = loglik(model, x, Float64[], y)
```

#### Bernoulli Model (Binary Data)  
```julia
model = ExponentialFamily(Bernoulli)  # Uses LogitLink automatically

# Latent field on logit scale
x = [0.0, 1.0, -1.0]  # Corresponds to probabilities [0.5, 0.73, 0.27]
y = [0, 1, 0]         # Binary observations

ll = loglik(model, x, Float64[], y)
```

#### Normal Model (Continuous Data)
```julia
model = ExponentialFamily(Normal)  # Uses IdentityLink automatically

# Latent field directly as means
x = [0.0, 1.0, -0.5]
θ = [0.5]             # Standard deviation
y = [0.1, 1.2, -0.4]  # Continuous observations

ll = loglik(model, x, θ, y)
```

#### Binomial Model
```julia
model = ExponentialFamily(Binomial)  # Uses LogitLink automatically

# Latent field on logit scale
x = [0.0, 0.5]
θ = [10.0]     # Number of trials
y = [5, 7]     # Number of successes

ll = loglik(model, x, θ, y)
```

### Non-Canonical Links

You can specify custom link functions for specialized applications:

```julia
# Poisson with identity link (non-canonical)
# Requires x values to be positive
model = ExponentialFamily(Poisson, IdentityLink())
x = [2.0, 5.0, 1.0]  # Directly as rates (must be positive)
y = [1, 4, 2]

# Bernoulli with log link (non-canonical)  
# Requires x values such that exp(x) ∈ (0,1)
model = ExponentialFamily(Bernoulli, LogLink())
x = [log(0.3), log(0.7)]  # log-probabilities
y = [0, 1]
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

## Likelihood Distributions

The [`likelihood`](@ref) function returns Distribution objects compatible with Distributions.jl:

```julia
model = ExponentialFamily(Poisson)
x = [1.0, 2.0]
θ = Float64[]

# Get likelihood as Distribution
dist = likelihood(model, x, θ)

# Use with Distributions.jl
synthetic_data = rand(dist, 100)    # Generate samples
mean_val = mean(dist)               # Compute moments  
var_val = var(dist)
prob = logpdf(dist, [2, 7])        # Evaluate likelihood

# Equivalent to direct loglik call
prob2 = loglik(model, x, θ, [2, 7])  # prob ≈ prob2
```

## Custom Observation Models

For specialized applications, you can implement custom observation models:

### Basic Custom Model

```julia
# Define custom struct
struct CustomNormalModel <: ObservationModel
    σ::Float64
end

# Implement required interface
function loglik(model::CustomNormalModel, x, θ, y)
    return sum(logpdf.(Normal.(x, model.σ), y))
end

# Gradient and Hessian computed automatically via ForwardDiff
model = CustomNormalModel(0.5)
ll = loglik(model, x, θ, y)
grad = loggrad(model, x, θ, y)  # Automatic differentiation
```

### Optimized Custom Model

For better performance, you can provide specialized gradient and Hessian implementations:

```julia
struct OptimizedModel <: ObservationModel
    σ²::Float64
end

function loglik(model::OptimizedModel, x, θ, y)
    return -0.5 * sum((y .- x).^2) / model.σ² - 0.5 * length(y) * log(2π * model.σ²)
end

function loggrad(model::OptimizedModel, x, θ, y)
    return (y .- x) ./ model.σ²  # Analytical gradient
end

function loghessian(model::OptimizedModel, x, θ, y)
    return Diagonal(-ones(length(x)) ./ model.σ²)  # Analytical Hessian
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
loglik
loggrad
loghessian
```

### Exponential Family

```@docs
ExponentialFamily
likelihood
```

### Link Functions

```@docs
LinkFunction
apply_link
apply_invlink
```
