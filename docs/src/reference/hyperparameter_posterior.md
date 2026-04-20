# [Hyperparameter Posterior](@id hyperparameter-posterior)

The hyperparameter posterior module implements the INLA approximation to the hyperparameter posterior π(θ | y), including mode finding, posterior exploration, and marginalization.

## Overview

INLA approximates the hyperparameter posterior using the Laplace approximation around the mode θ*. The hyperparameter posterior system provides:

- **Mode finding** using optimization methods to find θ* = argmax π(θ | y)
- **Posterior exploration** around the mode via grid or CCD design
- **Hyperparameter marginalization** via spline-based strategies for O(1) queries
- **Numerical stability** through proper handling of log-space computations

## Basic Usage

### Mode Finding

```julia
using Latte
using Distributions
using GaussianMarkovRandomFields

# Set up model with @hyperparams macro
spec = @hyperparams begin
    (σ ~ InverseGamma(2, 1), transform = log, space = natural)
end

function latent_gmrf(; σ, kwargs...)
    GMRF(zeros(5), spdiagm(0 => fill(1/σ^2, 5)))
end

obs_model = ExponentialFamily(Normal)
model = LatentGaussianModel(spec, latent_gmrf, obs_model)

# Observed data
y = [0.5, -0.2, 0.8, -0.1, 0.3]

# Find hyperparameter mode
θ_star, mode_points, mode_logdensities = find_hyperparameter_mode(model, y)
```

### Full Inference

```julia
# Run full INLA (mode finding + exploration + marginalization)
result = inla(model, y)

# Access hyperparameter marginals
summary_df(result.hyperparameter_marginals)

# Access latent marginals
mean.(result.latent_marginals)
```

### Custom Exploration and Marginalization

```julia
# Use CCD exploration with custom scaling
result = inla(model, y,
    exploration_strategy = CCDExplorationStrategy(f0 = 1.3)
)

# Use grid exploration with custom parameters
result = inla(model, y,
    exploration_strategy = GridExplorationStrategy(
        max_log_drop = 3.0,
        interpolation_subdivisions = 2
    )
)
```

## Mathematical Background

### INLA Hyperparameter Approximation

The INLA approach approximates the hyperparameter posterior as:

```math
\tilde{\pi}(\theta | y) \propto \frac{\pi(x^*(\theta), \theta, y)}{\tilde{\pi}_G(x^*(\theta) | \theta, y)}
```

where:
- x*(θ) is the mode of the latent field for given θ
- π̃_G is the Gaussian approximation to the latent field posterior

### Posterior Exploration

The exploration builds grids for integration by:

1. **Reparameterization**: Transform θ → z using eigendecomposition of the Hessian
2. **Design point placement**: Grid (Cartesian product) or CCD (Central Composite Design)
3. **Evaluation**: Compute log-density and latent marginals at each design point
4. **Normalization**: Normalize using integration weights

### Hyperparameter Marginalization

1D marginal posteriors are computed by profiling or summation, then fit with cubic splines:

```math
\tilde{\pi}(\theta_j | y) = \int \tilde{\pi}(\theta | y) d\theta_{-j}
```

All downstream queries (logpdf, cdf, quantile) are O(1) spline lookups.

## API Reference

### Mode Finding

```@docs
hyperparameter_logpdf
find_hyperparameter_mode
```

### Exploration

```@docs
explore_hyperparameter_posterior
```
