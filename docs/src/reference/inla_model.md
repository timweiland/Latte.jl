# INLA Model

The `INLAModel` provides a complete specification for INLA inference, combining hyperparameter priors, latent field structure, and observation models into a single coherent framework.

## Overview

An INLA model consists of three key components:

1. **Hyperparameter Prior**: Distributions over model hyperparameters θ
2. **Latent Field Prior**: Function mapping hyperparameters to a GMRF structure  
3. **Observation Model**: Links observations to the latent field

## Basic Usage

```julia
using IntegratedNestedLaplace, GaussianMarkovRandomFields, Distributions

# Define hyperparameter prior
hp_prior = HyperparameterPrior((σ = InverseGamma(2, 1),))

# Define latent field structure as function of hyperparameters
function ar1_latent(θ_named)
    σ = θ_named.σ
    n = 50
    ϕ = 0.8  # AR(1) coefficient
    
    # Build AR(1) precision matrix
    diag_main = [1.0; fill(1 + ϕ^2, n-2); 1.0] ./ σ^2
    diag_off = fill(-ϕ, n-1) ./ σ^2
    Q = spdiagm(0 => diag_main, -1 => diag_off, 1 => diag_off)
    
    return GMRF(zeros(n), Q, CholeskySolverBlueprint())
end

# Define observation model
obs_model = ExponentialFamily(Binomial)

# Create complete INLA model
model = INLAModel(hp_prior, ar1_latent, obs_model)
```

## Key Functions

### Model Construction

```@docs
INLAModel
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
hp_prior = HyperparameterPrior((
    σ_spatial = InverseGamma(2, 1),     # Spatial variance
    σ = InverseGamma(2, 0.5),           # Observation noise  
    ρ = Beta(2, 2)                      # Spatial correlation
))

function spatial_latent(θ_named)
    σ_spatial = θ_named.σ_spatial
    ρ = θ_named.ρ
    # ... build spatial GMRF using σ_spatial and ρ
end

obs_model = ExponentialFamily(Normal)  # Uses σ

model = INLAModel(hp_prior, spatial_latent, obs_model)
```

## Fixed and Free Parameters

The hyperparameter system supports both free parameters (to be estimated) and fixed parameters (held constant):

```julia
hp_prior = HyperparameterPrior(
    (σ = InverseGamma(2, 1),);           # Free parameter
    fixed = (df = 3.0,)                  # Fixed parameter
)

function robust_latent(θ_named)
    σ = θ_named.σ
    df = θ_named.df  # Access fixed parameter
    # ... build GMRF with both parameters
end
```

## Validation

The `INLAModel` constructor validates that all required hyperparameters for the observation model are provided. Missing required parameters will raise an error:

```julia
# This will error: Normal requires σ hyperparameter
hp_prior = HyperparameterPrior((μ = Normal(0, 1),))  # Missing σ
obs_model = ExponentialFamily(Normal)
model = INLAModel(hp_prior, latent_fn, obs_model)  # ERROR!
```

Additional hyperparameters beyond those required by the observation model are allowed, as they may be used by the latent field prior function.

## Integration with INLA Pipeline

The `INLAModel` serves as the foundation for the complete INLA inference pipeline.