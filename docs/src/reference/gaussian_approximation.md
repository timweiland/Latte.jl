# [Gaussian Approximation](@id gaussian-approximation)

The Gaussian approximation functionality provides efficient Newton-Raphson optimization for finding posterior modes in INLA. This is the core computational engine for constructing Gaussian approximations to non-Gaussian posteriors.

## Overview

The Gaussian approximation process finds the mode of the posterior distribution p(x|y) and constructs a Gaussian approximation around it using Fisher scoring (Newton-Raphson with Fisher information matrix). This approximation forms the foundation of the INLA methodology.

## Main Function

```@docs
gaussian_approximation
```

## Implementation Details

The gaussian_approximation function uses NonlinearSolve.jl with Newton-Raphson optimization and CHOLMOD factorization for efficient sparse linear algebra.

## Basic Usage

```julia
using IntegratedNestedLaplace
using GaussianMarkovRandomFields
using Distributions

# Set up prior GMRF
μ_prior = zeros(10)
Q_prior = spdiagm(0 => ones(10))
prior_gmrf = GMRF(μ_prior, Q_prior, CholeskySolverBlueprint())

# Set up observation model
obs_model = ExponentialFamily(Poisson)
θ_named = NamedTuple()  # No hyperparameters for Poisson

# Generate synthetic data
x_true = rand(prior_gmrf)
data_dist = data_distribution(obs_model, x_true, θ_named)
y_obs = rand(data_dist)

# Find Gaussian approximation
obs_lik = obs_model(y_obs; θ_named...)
posterior_gmrf = gaussian_approximation(prior_gmrf, obs_lik)

# Extract posterior statistics
posterior_mean = mean(posterior_gmrf)
posterior_precision = precision_matrix(posterior_gmrf)
println("Gaussian approximation computed successfully")
```

## Mathematical Properties

The Gaussian approximation finds the mode of the posterior distribution and constructs a Gaussian around it. For observation models with Gaussian likelihood, this approximation is exact:

```julia
# For Gaussian observations, the approximation is exact
obs_model = ExponentialFamily(Normal)
θ_named = (σ = 0.5,)

# The posterior precision combines prior and observation precision
# Q_posterior = Q_prior + Q_obs
# μ_posterior = Q_posterior^(-1) * (Q_prior * μ_prior + Q_obs * y)
```

## Performance Considerations

### Fisher Information vs Hessian

The implementation uses Fisher scoring, which approximates the Hessian with the Fisher information matrix. This has several advantages:

1. **Stability**: Fisher information is always positive semi-definite
2. **Efficiency**: Often faster convergence than pure Newton-Raphson
3. **Numerical robustness**: Less sensitive to poor conditioning

### Sparse Linear Algebra

The implementation leverages sparse linear algebra throughout:

- Sparse precision matrices are preserved during optimization
- Cholesky factorizations use sparse solvers when available
- Memory usage scales with the sparsity pattern of the prior

### Optimization Details

The implementation uses NonlinearSolve.jl with Newton-Raphson method:

- **Jacobian computation**: Uses automatic differentiation for precise gradients and Hessians
- **Linear solver**: CHOLMOD factorization for sparse precision matrices
- **Convergence**: Built-in tolerances for `abstol=1e-6` and `reltol=1e-6`
- **Permutation handling**: Preserves sparsity patterns from the prior GMRF

## Mathematical Background

### Fisher Scoring Update

At each iteration, the algorithm performs the update:

```
Q_new = Q_prior - Hessian_obs(μ_current)
gradient = Q_prior * (μ_current - μ_prior) - grad_obs(μ_current)
μ_new = μ_current - Q_new⁻¹ * gradient
```

Where:
- `Q_prior` is the prior precision matrix
- `μ_prior` is the prior mean
- `Hessian_obs` is the Hessian of the observation log-likelihood
- `grad_obs` is the gradient of the observation log-likelihood

### Gaussian Approximation

The final Gaussian approximation takes the form:

```
p(x|y) ≈ N(μ_mode, Q_mode⁻¹)
```

Where `μ_mode` is the posterior mode and `Q_mode` is the posterior precision matrix.

## Error Handling

The function handles various edge cases:

- **Non-convergence**: NonlinearSolve.jl provides convergence diagnostics
- **Numerical issues**: CHOLMOD factorization handles ill-conditioned matrices robustly
- **Matrix conditioning**: Sparse factorization with automatic pivoting

For problematic cases, consider:
- Checking observation model implementations (`loglik`, `loggrad`, `loghessian`)
- Verifying prior GMRF is well-conditioned
- Using more informative priors to improve conditioning