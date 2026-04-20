# [Hyperparameters](@id hyperparameters)

The hyperparameter system in Latte.jl provides type-safe management of model parameters with support for both free (optimized) and fixed parameters, including automatic parameter transformations between constrained and unconstrained spaces.

## Overview

INLA models typically have hyperparameters that control the latent field prior (e.g., smoothness, variance) and observation model (e.g., noise level). The hyperparameter system provides:

- **Declarative specification** using the `@hyperparams` macro for readable model definitions
- **Automatic transformations** between natural (constrained) and working (unconstrained) spaces with Jacobian corrections
- **Type-safe parameter access** using NamedTuples instead of positional vectors
- **Mixed free/fixed parameters** for flexible model specification
- **Built-in bijector support** for common transformations (log, logit) and custom transformations

## Basic Usage

### Simple Hyperparameter Prior

```julia
using Latte
using Distributions

# Define prior over a single hyperparameter
spec = @hyperparams begin
    (σ ~ Gamma(2, 3), transform = log, space = natural)
end

# Work with parameter vectors and conversions
θ_w = WorkingHyperparameters([1.5], spec)  # Parameter in working space (log(σ))
θ_n = convert(NaturalHyperparameters, θ_w)  # Transform to natural space: σ = exp(1.5)
θ_nt = convert(NamedTuple, θ_n)             # Convert to NamedTuple: (σ = exp(1.5),)
θ_back = θ_n.θ                               # Access underlying vector: [exp(1.5)]
```

### Multiple Parameters with Transformations

```julia
# Multiple free parameters with different transformations
spec = @hyperparams begin
    (σ ~ Gamma(2, 3), transform = log, space = natural)          # PC prior on σ in natural space
    (ρ ~ Beta(2, 2), transform = logit, space = natural)        # Logit transform for correlation
    (τ ~ InverseGamma(1, 1), transform = log, space = natural) # Log transform for precision
end

θ_w = WorkingHyperparameters([0.5, -1.2, 0.8], spec)  # Parameters in working space
θ_n = convert(NaturalHyperparameters, θ_w)
θ_nt = convert(NamedTuple, θ_n)
# Result: (ρ = logistic(-1.2), σ = exp(0.5), τ = exp(0.8))
```

## Transformation and Space Specification

The `@hyperparams` macro supports flexible parameter transformations:

```julia
spec = @hyperparams begin
    # Identity transform (working space = natural space)
    (μ ~ Normal(0, 1), transform = identity, space = working)

    # Log transform for positive parameters
    (σ ~ LogNormal(0, 1), transform = log, space = working)

    # Custom bijector for bounded parameters
    (ρ ~ Uniform(0, 1), transform = Bijectors.Logit(0.0, 1.0), space = natural)
end
```

**Parameters:**
- `transform`: Bijector mapping natural → working space. Common options: `log`, `logit`, `identity`, or custom bijectors
- `space`: Where the prior is specified: `natural` (user-space) or `working` (optimization space)

The system automatically handles Jacobian corrections when the prior is specified in natural space.

## Mixed Free and Fixed Parameters

A powerful feature is the ability to fix some parameters while optimizing others:

```julia
# Fix some parameters, optimize others
spec = @hyperparams begin
    (ρ ~ Beta(1, 1), transform = logit, space = natural)           # Free: correlation
    (τ ~ Gamma(2, 1), transform = log, space = natural)            # Free: precision
    σ = 0.5                                                         # Fixed: observation noise
end

# Only free parameters go in the vector
θ_w = WorkingHyperparameters([1.2, -0.5], spec)  # [transformed ρ, transformed τ]

# Convert to natural space and includes fixed parameters
θ_n = convert(NaturalHyperparameters, θ_w)
θ_nt = convert(NamedTuple, θ_n)
# Result: (ρ = logistic(1.2), σ = 0.5, τ = exp(-0.5))

# Access parameters by name using dot notation
ρ_natural = θ_n.ρ         # logistic(1.2) (natural space)
σ_fixed = θ_n.σ           # 0.5 (fixed value)
```

## Integrating with INLA Models

The `@hyperparams` macro integrates seamlessly with `LatentGaussianModel`:

```julia
using GaussianMarkovRandomFields

# Define hyperparameters
spec = @hyperparams begin
    (σ ~ Gamma(2, 3), transform = log, space = natural)
    (ρ ~ Beta(2, 2), transform = logit, space = natural)
end

# Define latent model using keyword arguments
function spatial_gmrf(; σ, ρ, kwargs...)
    n = 50
    # Simple AR(1) precision matrix scaled by σ
    Q = spdiagm(-1 => -ρ*ones(n-1), 0 => (1+ρ^2)*ones(n), 1 => -ρ*ones(n-1))
    return GMRF(zeros(n), Q / σ^2)
end

# Observation model
obs_model = ExponentialFamily(Normal)

# Create INLA model
model = LatentGaussianModel(spec, spatial_gmrf, obs_model)

# Sample from model to generate synthetic data
θ, x, y = rand(model)  # Sample hyperparameters, latent field, and observations
```

**Key insight:** Model functions receive parameters in **natural space** via keyword arguments, even though optimization happens in working space. The system automatically handles all conversions.

## Prior Space Transformation

When you specify a prior in natural space, the system automatically transforms it to working space with Jacobian correction:

```julia
# This prior on σ in natural space (PC prior):
spec = @hyperparams begin
    (σ ~ Exponential(1.0), transform = log, space = natural)
end

# Is equivalent to a prior on log(σ) in working space:
# p(log_σ) = p(σ=exp(log_σ)) * |dσ/d(log_σ)| = Exponential(1) * σ = Exponential(1) * exp(log_σ)
```

This Jacobian correction is handled automatically.

## Complete Example: Spatial Model with Transformations

Here's a complete example showing AR(1) model with mixed parameters and transformations:

```julia
using Latte
using GaussianMarkovRandomFields
using Distributions
using LinearAlgebra
using SparseArrays

# Define hyperparameters with transformations
spec = @hyperparams begin
    (σ ~ Exponential(1.0), transform = log, space = natural)      # Marginal std dev (log scale)
    (ρ ~ Beta(2, 2), transform = logit, space = natural)         # Autocorrelation (logit scale)
end

# Define latent GMRF using keyword arguments
function ar1_gmrf(; σ, ρ, kwargs...)
    k = 100
    # AR(1) precision matrix
    Q = spdiagm(
        -1 => -ρ*ones(k-1),
         0 => (1 + ρ^2)*ones(k),
         1 => -ρ*ones(k-1)
    )
    return GMRF(zeros(k), Q / σ^2)
end

# Observation model (Normal with unknown σ parameter)
obs_model = ExponentialFamily(Normal)

# Create INLA model
model = LatentGaussianModel(spec, ar1_gmrf, obs_model)

# Generate synthetic data
θ_true = (σ = 2.0, ρ = 0.8)  # Natural space
x_true = rand(ar1_gmrf(; σ = θ_true.σ, ρ = θ_true.ρ))
y_obs = rand(conditional_distribution(obs_model, x_true; σ = 0.1))

# Run INLA inference
result = inla(model, y_obs)

# Results are in natural space
println("Posterior mode for σ: ", result.hyperparameter_mode[1])  # In natural space
```

## Type Stability and Performance

The hyperparameter system is designed for type stability:

```julia
spec = @hyperparams begin
    (σ ~ Gamma(2, 3), transform = log, space = natural)
    (ρ ~ Beta(1, 1), transform = logit, space = natural)
end

θ_w = WorkingHyperparameters([1.5, 0.3], spec)

# These operations are type-stable
@inferred convert(NaturalHyperparameters, θ_w)
@inferred convert(NamedTuple, θ_w)
@inferred logpdf_prior(θ_w)  # Evaluates prior in working space
```

All conversions and prior evaluations have concrete return types determined at compile time.

## API Reference

```@docs
@hyperparams
HyperparameterSpec
Hyperparameter
WorkingHyperparameters
NaturalHyperparameters
logpdf_prior
logdetjac
```
