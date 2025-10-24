# [Observation Models](@id observation-models)

!!! note "Provided by GaussianMarkovRandomFields.jl"
    Observation models are implemented in [GaussianMarkovRandomFields.jl](https://github.com/timweiland/GaussianMarkovRandomFields.jl) v0.4+ and re-exported by IntegratedNestedLaplace.jl for user convenience. **For detailed API documentation**, see the [GaussianMarkovRandomFields.jl documentation](https://timweiland.github.io/GaussianMarkovRandomFields.jl/).

This guide shows how to use observation models with INLA. Observation models define the relationship between observations `y` and the latent field `x` through probability distributions and link functions.

## Quick Start

```julia
using IntegratedNestedLaplace
using Distributions

# 1. Create observation model (Poisson with canonical log link)
obs_model = ExponentialFamily(Poisson)

# 2. Use in INLA model
spec = @hyperparams begin
    (σ ~ Gamma(2, 1), transform = log, space = natural)
end

model = INLAModel(spec, my_latent_function, obs_model)

# 3. Run inference
result = inla(model, y_observed)
```

## Overview

Observation models use a **factory pattern** for efficiency:
1. **Create template**: `obs_model = ExponentialFamily(Distribution)`
2. **Use in INLA**: Pass to `INLAModel(spec, latent_fn, obs_model)`
3. **INLA handles the rest**: Automatic materialization with data and hyperparameters

The `ExponentialFamily` struct supports most common statistical distributions with their canonical link functions.

## Common Observation Models

| Distribution | Canonical Link | Hyperparameters | Use Case |
|:-------------|:---------------|:----------------|:---------|
| `Normal` | Identity | `σ` (std dev) | Continuous data |
| `Poisson` | Log | None | Count data |
| `Bernoulli` | Logit | None | Binary data |
| `Binomial` | Logit | `n` (trials) | Binomial trials |

### Usage Examples

#### Poisson Model (Count Data)
```julia
using IntegratedNestedLaplace, GaussianMarkovRandomFields, Distributions

# Poisson uses log link: η = log(λ) where λ is the rate
obs_model = ExponentialFamily(Poisson)

spec = @hyperparams begin
    (ρ ~ Beta(2, 2), transform = logit, space = natural)
end

# Latent field defines log-rates
latent_fn(; ρ, kwargs...) = GMRF(zeros(100), ar1_precision(100, ρ))

model = INLAModel(spec, latent_fn, obs_model)
```

#### Bernoulli Model (Binary Data)
```julia
# Bernoulli uses logit link: η = logit(p) where p is the probability
obs_model = ExponentialFamily(Bernoulli)

# Use in INLA model (no observation hyperparameters needed)
model = INLAModel(spec, latent_fn, obs_model)
```

#### Normal Model (Continuous Data)
```julia
# Normal uses identity link: η = μ directly
obs_model = ExponentialFamily(Normal)

# Normal requires observation noise σ as hyperparameter
spec = @hyperparams begin
    (σ_latent ~ Gamma(2, 1), transform = log, space = natural)
    (σ_obs ~ Gamma(2, 1), transform = log, space = natural)
end

# The σ_obs will be passed to the observation model automatically
model = INLAModel(spec, latent_fn, obs_model)
```

#### Binomial Model
```julia
# Binomial uses logit link for success probability
obs_model = ExponentialFamily(Binomial)

# Binomial requires n (number of trials) as hyperparameter
spec = @hyperparams begin
    (ρ ~ Beta(2, 2), transform = logit, space = natural)
    n = 10.0  # Fixed number of trials
end

model = INLAModel(spec, latent_fn, obs_model)
```

### Non-Canonical Links

You can specify custom link functions when needed:

```julia
# Poisson with identity link (latent field = rate directly, must be positive)
obs_model = ExponentialFamily(Poisson, IdentityLink())

# Bernoulli with log link (latent field = log-probability)
obs_model = ExponentialFamily(Bernoulli, LogLink())
```

For most applications, the canonical links (default) are recommended.

## Composite Observation Models

Composite observation models combine multiple observation types (e.g., continuous and count data) in a single INLA model:

```julia
using IntegratedNestedLaplace

# Create models with index ranges
normal_model = ExponentialFamily(Normal, indices = 1:3)    # First 3 latent values
poisson_model = ExponentialFamily(Poisson, indices = 4:6)  # Next 3 latent values

composite_model = CompositeObservationModel((normal_model, poisson_model))

# Prepare heterogeneous observations
y_normal = [1.0, 2.0, 1.5]
y_poisson = [2, 3, 1]
y_composite = CompositeObservations((y_normal, y_poisson))

# Use with INLA
spec = @hyperparams begin
    (ρ ~ Beta(2, 2), transform = logit, space = natural)
    (σ ~ Gamma(2, 1), transform = log, space = natural)  # For Normal component
end

model = INLAModel(spec, latent_fn, composite_model)
result = inla(model, y_composite)
```

Each component uses its specified latent field indices and extracts its required hyperparameters automatically.

## Custom Observation Models

For specialized applications beyond the built-in exponential family models, you can implement custom observation models. This requires:

1. Defining a struct that subtypes `ObservationModel`
2. Implementing the factory pattern with `(model::YourModel)(y; kwargs...)`
3. Implementing `loglik(obs_lik, x)` for the materialized likelihood

See the [GaussianMarkovRandomFields.jl documentation](https://timweiland.github.io/GaussianMarkovRandomFields.jl/) for detailed implementation guides and advanced features like:
- Analytical gradient and Hessian implementations for performance
- Automatic differentiation with sparsity detection
- Custom link function definitions

## Further Reading

For complete API documentation, advanced features, and implementation details, refer to:
- **[GaussianMarkovRandomFields.jl Documentation](https://timweiland.github.io/GaussianMarkovRandomFields.jl/)** - Complete observation model API reference
- The examples in `examples/` directory for practical INLA usage patterns
