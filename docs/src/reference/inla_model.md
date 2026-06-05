# INLA Model

The `LatentGaussianModel` provides a complete specification for INLA inference, combining hyperparameter priors, latent field structure, and observation models into a single coherent framework.

!!! tip "Prefer `@latte` for most models"
    Most models are written with the `@latte` macro, which builds a
    `LatentGaussianModel` for you (see [Main Interface](@ref main-interface) and
    the tutorials). Construct one directly, as shown below, when you need full
    control over the latent prior.

## Overview

An INLA model consists of three key components:

1. **Hyperparameter Prior**: Distributions over model hyperparameters θ
2. **Latent Field Prior**: Function mapping hyperparameters to a GMRF structure  
3. **Observation Model**: Links observations to the latent field

## Basic Usage

```julia
using Latte, GaussianMarkovRandomFields, Distributions, SparseArrays

# Hyperparameter spec — free parameters via `~`, in a transformed working space
spec = @hyperparams begin
    (σ ~ InverseGamma(2, 1), transform = log, space = natural)
end

# Latent field as a function of the (keyword) hyperparameters
function ar1_latent(; σ, kwargs...)
    n = 50
    ϕ = 0.8  # AR(1) coefficient
    
    # Build AR(1) precision matrix
    diag_main = [1.0; fill(1 + ϕ^2, n-2); 1.0] ./ σ^2
    diag_off = fill(-ϕ, n-1) ./ σ^2
    Q = spdiagm(0 => diag_main, -1 => diag_off, 1 => diag_off)
    
    return (zeros(n), Q)  # FunctionLatentModel expects (mean, precision)
end

# Define observation model
obs_model = ExponentialFamily(Binomial)

# Complete INLA model — wrap the latent function with its dimension
model = LatentGaussianModel(spec, FunctionLatentModel(ar1_latent, 50), obs_model)
```

## Key Functions

### Model Construction

```@docs
LatentGaussianModel
```

### Model Utilities

```@docs
latent_gmrf
log_joint_density
```

## Multiple Hyperparameters

Models can have multiple hyperparameters affecting both the latent field and observation model:

```julia
# Multiple hyperparameters
spec = @hyperparams begin
    (σ_spatial ~ InverseGamma(2, 1), transform = log, space = natural)   # Spatial SD
    (σ ~ InverseGamma(2, 0.5), transform = log, space = natural)         # Observation noise
    (ρ ~ Beta(2, 2), transform = logit, space = natural)                 # Spatial correlation
end

function spatial_latent(; σ_spatial, ρ, kwargs...)
    n = 100
    # ... build the spatial precision Q from σ_spatial and ρ
    return (zeros(n), Q)
end

obs_model = ExponentialFamily(Normal)  # Uses σ

model = LatentGaussianModel(spec, FunctionLatentModel(spatial_latent, 100), obs_model)
```

## Fixed and Free Parameters

The hyperparameter system supports both free parameters (to be estimated) and fixed parameters (held constant):

```julia
spec = @hyperparams begin
    (σ ~ InverseGamma(2, 1), transform = log, space = natural)  # Free parameter
    df = 3.0                                                     # Fixed parameter
end

function robust_latent(; σ, df, kwargs...)
    # Both the free (σ) and fixed (df) parameters arrive as keyword arguments
    n = 50
    # ... build the precision Q from σ and df
    return (zeros(n), Q)
end
```

## Validation

The `LatentGaussianModel` constructor validates that all required hyperparameters for the observation model are provided. Missing required parameters will raise an error:

```julia
# This will error: Normal requires a σ hyperparameter
spec = @hyperparams begin
    (τ ~ Gamma(2, 1), transform = log, space = natural)  # no σ provided
end
obs_model = ExponentialFamily(Normal)
model = LatentGaussianModel(spec, FunctionLatentModel(latent_fn, 50), obs_model)  # ERROR: missing σ
```

Additional hyperparameters beyond those required by the observation model are allowed, as they may be used by the latent field prior function.

## Integration with INLA Pipeline

The `LatentGaussianModel` serves as the foundation for the complete INLA inference pipeline.