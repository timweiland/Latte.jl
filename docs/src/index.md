```@meta
CurrentModule = IntegratedNestedLaplace
```

# IntegratedNestedLaplace.jl

A Julia package for Integrated Nested Laplace Approximation (INLA), providing fast Bayesian inference for latent Gaussian models.

## Overview

IntegratedNestedLaplace.jl implements the INLA methodology for approximate Bayesian inference in models with Gaussian latent fields. The package is designed to work seamlessly with [GaussianMarkovRandomFields.jl](https://github.com/JuliaGaussianMarkovRandomFields/GaussianMarkovRandomFields.jl) for efficient handling of structured priors.

### Key Features

- **Observation Models**: Flexible interface supporting exponential family distributions with link functions
- **Gaussian Approximation**: Fast Newton-Raphson optimization with Fisher scoring for finding posterior modes  
- **Automatic Differentiation**: Efficient gradient and Hessian computation with sparsity detection
- **GMRF Integration**: Native support for Gaussian Markov Random Fields as priors

## Quick Start

```julia
using IntegratedNestedLaplace
using GaussianMarkovRandomFields
using Distributions

# Set up observation model
obs_model = ExponentialFamily(Binomial)  # Binomial with logit link

# Set up prior GMRF
μ_prior = zeros(10)
Q_prior = spdiagm(0 => ones(10))
prior_gmrf = GMRF(μ_prior, Q_prior, CholeskySolverBlueprint())

# Generate some data
x_true = rand(prior_gmrf)
θ_named = (n = 20,)  # Number of trials per observation
y_obs = rand(likelihood(obs_model, x_true, θ_named))

# Find Gaussian approximation to posterior
result = gaussian_approximation(prior_gmrf, obs_model, θ_named, y_obs)

# Extract posterior
posterior_gmrf = to_gmrf(result)
posterior_mean = mean(posterior_gmrf)
```

## Package Structure

The package is organized into several key components:

### Observation Models

The observation model interface connects observations to latent fields through probability distributions and link functions. See [Observation Models](@ref observation-models) for detailed documentation.

### Gaussian Approximation

Efficient Newton-Raphson optimization for finding posterior modes in INLA. See [Gaussian Approximation](@ref gaussian-approximation) for detailed documentation.

## Examples

The `examples/` directory contains complete working examples:

- **`autoregressive_bernoulli_inla.jl`**: INLA vs MCMC comparison for AR(1) latent field with Bernoulli observations
- **`gaussian_approximation_demo.jl`**: Basic usage of the Gaussian approximation functionality

## Installation

```julia
using Pkg
Pkg.add("https://github.com/timweiland/IntegratedNestedLaplace.jl")
```

## Related Packages

- [GaussianMarkovRandomFields.jl](https://github.com/timweiland/GaussianMarkovRandomFields.jl): Efficient GMRF operations
- [Distributions.jl](https://github.com/JuliaStats/Distributions.jl): Probability distributions
- [Turing.jl](https://github.com/TuringLang/Turing.jl): Probabilistic programming for comparison

