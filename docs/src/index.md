```@raw html
---
layout: home

hero:
  name: "IntegratedNestedLaplace.jl"
  text: "Fast Bayesian Inference with INLA"
  tagline: A Julia package for Integrated Nested Laplace Approximation, providing fast Bayesian inference for latent Gaussian models.
  image:
    src: /logo.svg
    alt: IntegratedNestedLaplace.jl
  actions:
    - theme: brand
      text: Get Started
      link: /main_interface
    - theme: alt
      text: View on GitHub
      link: https://github.com/timweiland/IntegratedNestedLaplace.jl

features:
  - icon: 🔗
    title: Observation Models
    details: Flexible interface supporting exponential family distributions with link functions for connecting observations to latent fields.
  
  - icon: 🧮
    title: Gaussian Approximation
    details: Fast Newton-Raphson optimization with Fisher scoring for finding posterior modes and automatic differentiation.
  
  - icon: 📊
    title: GMRF Integration
    details: Native support for Gaussian Markov Random Fields as priors with efficient sparse matrix operations.
  
  - icon: ⚡
    title: High Performance
    details: Efficient gradient and Hessian computation with automatic sparsity detection and optimized linear algebra.
  
  - icon: 🔄
    title: Automatic Differentiation
    details: Seamless AD integration with ForwardDiff.jl and SparseDiffTools.jl for fast and accurate derivatives.
  
  - icon: 📈
    title: Progress Tracking
    details: Built-in progress monitoring with ProgressMeter.jl integration for long-running inference tasks.
---
```

```@meta
CurrentModule = IntegratedNestedLaplace
```

## Overview

IntegratedNestedLaplace.jl implements the INLA methodology for approximate Bayesian inference in models with Gaussian latent fields. The package is designed to work seamlessly with [GaussianMarkovRandomFields.jl](https://github.com/JuliaGaussianMarkovRandomFields/GaussianMarkovRandomFields.jl) for efficient handling of structured priors.

## Quick Start

```julia
using IntegratedNestedLaplace
using GaussianMarkovRandomFields
using Distributions
using SparseArrays

# AR-1 precision matrix for time series
function ar1_precision(ρ, k)
    return spdiagm(-1 => -ρ * ones(k - 1), 0 => ones(k) .+ ρ^2, 1 => -ρ * ones(k - 1))
end

# Define hyperparameters with the @hyperparams macro
k = 100
spec = @hyperparams begin
    (τ ~ Exponential(1.0), transform = log, space = natural)  # Precision
    (ρ ~ Beta(5, 1), transform = logit, space = natural)      # Autocorrelation
end

# Latent GMRF with AR-1 structure
# Uses keyword arguments matching hyperparameter names
function latent_gmrf(; τ, ρ, kwargs...)
    Q = ar1_precision(ρ, k) .* τ
    μ = log(1000.0) .* [ρ^i for i in 1:k]  # Exponential decay
    return GMRF(μ, Q)
end

# Poisson observations with log-link
obs_model = ExponentialFamily(Poisson)
model = INLAModel(spec, latent_gmrf, obs_model)

# Run INLA inference
result = inla(model, y_observed)

# Access posterior results
hyperparameter_marginals = result.hyperparameter_marginals
latent_marginals = result.latent_marginals
posterior_mode = result.hyperparameter_mode
```

## Package Structure

The package provides both high-level and low-level interfaces:

### [Main Interface](@id main-interface-overview)

The [`inla`](@ref) function provides a unified interface for INLA inference with automatic hyperparameter marginalization and progress tracking. See [Main Interface](@ref main-interface) for complete documentation and examples.

### Low-Level Components

For advanced users, the package exposes individual components:

- **[Observation Models](@ref observation-models)**: Connects observations to latent fields through probability distributions and link functions
- **[Gaussian Approximation](@ref gaussian-approximation)**: Newton-Raphson optimization for finding posterior modes
- **[Hyperparameter Posterior](@ref hyperparameter-posterior)**: Exploration and marginalization over hyperparameters
- **[Marginalization](@ref)**: Computation of latent field marginals

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

