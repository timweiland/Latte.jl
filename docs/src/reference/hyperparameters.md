# [Hyperparameters](@id hyperparameters)

The hyperparameter system in IntegratedNestedLaplace.jl provides type-safe management of model parameters with support for both free (optimized) and fixed parameters.

## Overview

INLA models typically have hyperparameters that control the latent field prior (e.g., smoothness, variance) and observation model (e.g., noise level). The hyperparameter system provides:

- **Type-safe parameter access** using NamedTuples instead of positional vectors
- **Mixed free/fixed parameters** for flexible model specification
- **Automatic validation** ensuring all required parameters are provided
- **Efficient parameter transformations** between vectors and named representations

## Basic Usage

### Simple Hyperparameter Prior

```julia
using IntegratedNestedLaplace
using Distributions

# Define prior over a single hyperparameter
hp_prior = HyperparameterPrior((σ = Gamma(2, 3),))

# Work with parameter vectors and named tuples
θ_vec = [1.5]                           # Parameter vector
θ_named = to_named(θ_vec, hp_prior)     # Convert to NamedTuple: (σ = 1.5,)
θ_back = to_vector(θ_named, hp_prior)   # Convert back: [1.5]

# Access individual parameters
σ_value = get_hyperparameter(θ_vec, hp_prior, :σ)  # Extract σ = 1.5
set_hyperparameter!(θ_vec, hp_prior, :σ, 2.0)      # Set σ = 2.0
```

### Multiple Parameters

```julia
# Multiple free parameters
hp_prior = HyperparameterPrior((
    σ = Gamma(2, 3),      # Noise standard deviation
    ρ = Beta(2, 2),       # Spatial correlation
    τ = InverseGamma(1, 1) # Precision parameter
))

θ_vec = [1.2, 0.7, 0.8]
θ_named = to_named(θ_vec, hp_prior)  # (ρ = 0.7, σ = 1.2, τ = 0.8)

# Extract subset of parameters
subset = extract_hyperparameters(θ_vec, hp_prior, (:σ, :τ))  # (σ = 1.2, τ = 0.8)
```

## Mixed Free and Fixed Parameters

A powerful feature is the ability to fix some parameters while optimizing others:

```julia
# Fix some parameters, optimize others
hp_prior = HyperparameterPrior(
    (ρ = Beta(1, 1), τ = Gamma(2, 1)),  # Free parameters
    fixed = (σ = 0.5, μ = 0.0)          # Fixed parameters
)

# Only free parameters go in the vector
θ_free = [0.3, 1.2]  # [ρ, τ]

# But named tuples include all parameters
θ_named = to_named(θ_free, hp_prior)
# Result: (μ = 0.0, ρ = 0.3, σ = 0.5, τ = 1.2)

# Access works for both free and fixed parameters
ρ_val = get_hyperparameter(θ_free, hp_prior, :ρ)  # 0.3 (free)
σ_val = get_hyperparameter(θ_free, hp_prior, :σ)  # 0.5 (fixed)

# Can only modify free parameters
set_hyperparameter!(θ_free, hp_prior, :ρ, 0.8)  # ✓ Works
set_hyperparameter!(θ_free, hp_prior, :σ, 1.0)  # ✗ Error: σ is fixed
```

## Advanced Construction

### Correlated Parameters

For parameters that should be modeled jointly:

```julia
using LinearAlgebra

# Define correlated prior for two parameters
corr_matrix = [1.0 0.5; 0.5 1.0]
joint_dist = MvNormal([0.0, 0.0], corr_matrix)

hp_prior = HyperparameterPrior{(:ρ, :τ)}(
    joint_dist,
    fixed = (σ = 0.5,)
)

# The free parameters are now correlated
θ_sample = rand(hp_prior.free_distribution)  # Correlated [ρ, τ] sample
```

### Integration with Observation Models

Hyperparameter priors automatically validate against observation model requirements:

```julia
# Normal observation model requires σ parameter
obs_model = ExponentialFamily(Normal)
required_params = hyperparameters(obs_model)  # (:σ,)

# Valid: provides required σ
hp_prior_valid = HyperparameterPrior((σ = Gamma(2, 3), ρ = Beta(1, 1)))

# Valid: σ is fixed but provided
hp_prior_fixed = HyperparameterPrior(
    (ρ = Beta(1, 1),),
    fixed = (σ = 0.5,)
)

# Invalid: missing required σ
hp_prior_invalid = HyperparameterPrior((ρ = Beta(1, 1),))

# Create INLA model with validation
latent_gmrf = θ_named -> GMRF(zeros(10), I)  # Dummy latent prior

model_valid = INLAModel(hp_prior_valid, latent_gmrf, obs_model)     # ✓ Works
model_fixed = INLAModel(hp_prior_fixed, latent_gmrf, obs_model)     # ✓ Works  
model_invalid = INLAModel(hp_prior_invalid, latent_gmrf, obs_model) # ✗ Error
```

## Performance Considerations

### Type Stability

The hyperparameter system is designed for type stability:

```julia
hp_prior = HyperparameterPrior((σ = Gamma(2, 3), ρ = Beta(1, 1)))
θ = [1.5, 0.3]

# These operations are type-stable
@inferred get_hyperparameter(θ, hp_prior, :σ)
@inferred to_named(θ, hp_prior)
@inferred to_vector((σ = 1.5, ρ = 0.3), hp_prior)
```

### Parameter Name Encoding

Parameter names are encoded in the type signature for compile-time optimization:

```julia
hp_prior = HyperparameterPrior((σ = Gamma(2, 3), ρ = Beta(1, 1)))

# Type encodes parameter information
typeof(hp_prior)  # HyperparameterPrior{(:σ, :ρ), (:ρ, :σ), ...}
```

## Working with Distributions

The hyperparameter prior integrates seamlessly with Distributions.jl:

```julia
hp_prior = HyperparameterPrior((σ = Gamma(2, 3), ρ = Beta(1, 1)))

# Standard distribution operations work on free parameters
θ_mode = mode(hp_prior.free_distribution)      # Modal values
θ_sample = rand(hp_prior.free_distribution)    # Random sample
log_dens = logpdf(hp_prior.free_distribution, θ_sample)  # Log density
```

## Complete Example

Here's a complete example showing a spatial model with mixed free and fixed parameters:

```julia
using IntegratedNestedLaplace
using Distributions
using LinearAlgebra

# Define hyperparameter prior: optimize correlation, fix observation noise
hp_prior = HyperparameterPrior(
    (ρ = Beta(2, 2), τ = Gamma(2, 1)),     # Free: correlation and latent precision
    fixed = (σ = 0.1,)                     # Fixed: observation noise
)

# Define latent GMRF that uses all parameters
function spatial_gmrf(θ_named)
    ρ, τ = θ_named.ρ, θ_named.τ  # Extract free parameters
    
    # Simple AR(1) precision matrix scaled by τ
    n = 50
    Q = τ * spdiagm(-1 => fill(-ρ, n-1), 0 => ones(n), 1 => fill(-ρ, n-1))
    return GMRF(zeros(n), Q)
end

# Normal observation model (requires σ parameter)
obs_model = ExponentialFamily(Normal)

# Create complete INLA model
model = INLAModel(hp_prior, spatial_gmrf, obs_model)

# Example parameter vector and conversion
θ_free = [0.7, 2.5]  # [ρ, τ] values
θ_named = to_named(θ_free, hp_prior)  # (ρ = 0.7, σ = 0.1, τ = 2.5)

# Get latent GMRF for these parameters
latent_prior = spatial_gmrf(θ_named)
```

## API Reference

```@docs
HyperparameterPrior
get_hyperparameter
set_hyperparameter!
to_named
to_vector
extract_hyperparameters
```
