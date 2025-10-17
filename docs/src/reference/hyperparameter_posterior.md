# [Hyperparameter Posterior](@id hyperparameter-posterior)

The hyperparameter posterior module implements the INLA approximation to the hyperparameter posterior π(θ | y), including mode finding, posterior exploration, interpolation, and marginalization.

## Overview

INLA approximates the hyperparameter posterior using the Laplace approximation around the mode θ*. The hyperparameter posterior system provides:

- **Mode finding** using optimization methods to find θ* = argmax π(θ | y)
- **Posterior exploration** around the mode to build integration and interpolation grids
- **Fast interpolation** for evaluating the posterior at arbitrary points
- **Marginal computation** by integrating over other hyperparameters
- **Numerical stability** through proper handling of log-space computations

## Basic Usage

### Mode Finding

```julia
using IntegratedNestedLaplace
using Distributions
using GaussianMarkovRandomFields

# Set up model with new @hyperparams macro
spec = @hyperparams begin
    (σ ~ InverseGamma(2, 1), transform = log, space = natural)
end

function latent_gmrf(; σ, kwargs...)
    GMRF(zeros(5), spdiagm(0 => fill(1/σ^2, 5)))
end

obs_model = ExponentialFamily(Normal)
model = INLAModel(spec, latent_gmrf, obs_model)

# Observed data
y = [0.5, -0.2, 0.8, -0.1, 0.3]

# Find hyperparameter mode
θ_star, mode_points, mode_logdensities = find_hyperparameter_mode(model, y)
```

### Posterior Exploration and Interpolation

```julia
# Explore posterior around the mode
exploration = explore_hyperparameter_posterior(
    model, y, θ_star, GaussianMarginal(), 1:5
)

# Build interpolant for fast evaluation
posterior_approx = build_posterior_interpolant(exploration)

# Evaluate posterior at arbitrary points (in working space)
test_θ_working = [2.0]
logpdf_val = posterior_approx(test_θ_working)
```

### Marginal Computation

```julia
# For multidimensional hyperparameter space
spec_2d = @hyperparams begin
    (μ ~ Normal(0, 1), transform = identity, space = working)
    (σ ~ InverseGamma(2, 1), transform = log, space = natural)
end
# ... set up 2D model ...

# Compute marginal posterior for first dimension
test_value = 0.5
marginal_logpdf = hyperparameter_marginal_logpdf(posterior_approx, 1, test_value)
```

## Advanced Usage

### Custom Optimization Settings

```julia
using Optim

# Use different optimization method
θ_star = find_hyperparameter_mode(
    model, y;
    method = LBFGS(),
    collect_points = false
)[1]
```

### Integration Tolerances

```julia
# Control exploration resolution and accuracy
exploration = explore_hyperparameter_posterior(
    model, y, θ_star,
    GaussianMarginal(), 1:length(y);
    integration_step_z = 0.5,        # Step size for exploration
    max_log_drop = 2.5,              # Log-density tolerance
    interpolation_subdivisions = 2   # Interpolation refinement
)

# Control marginal integration accuracy
marginal_logpdf = hyperparameter_marginal_logpdf(
    posterior_approx, 1, test_value;
    rtol = 1e-6, atol = 1e-10  # Tighter integration tolerances
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

The exploration builds grids for integration and interpolation by:

1. **Reparameterization**: Transform θ → z using eigendecomposition of the Hessian
2. **Dimensional exploration**: Explore along each transformed dimension until log-density drops by δ_π
3. **Grid construction**: Build multidimensional grid from dimensional explorations
4. **Normalization**: Normalize using integration points for proper posterior approximation

### Marginal Integration

Marginal posteriors are computed by numerical integration:

```math
\tilde{\pi}(\theta_j | y) = \int \tilde{\pi}(\theta | y) d\theta_{-j}
```

Using adaptive cubature with the bounds determined during exploration.

## API Reference

### Types

```@docs
HyperparameterExploration
HyperparameterPosteriorApproximation
```

### Mode Finding

```@docs
hyperparameter_logpdf
find_hyperparameter_mode
```

### Exploration and Interpolation

```@docs
explore_hyperparameter_posterior
build_posterior_interpolant
```

### Marginals

```@docs
hyperparameter_marginal_logpdf
```
