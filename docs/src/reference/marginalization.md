# Marginalization

```@meta
CurrentModule = IntegratedNestedLaplace
```

After computing a Gaussian approximation to the posterior, INLA provides several methods to marginalize over the latent field variables to obtain univariate marginal distributions. These marginals are essential for inference about individual components of the latent field.

## Overview

The [`marginalize`](@ref) function computes marginal approximations for specified latent variables using different approximation methods:

- **Gaussian Marginalization**: Direct marginalization of the Gaussian approximation (fastest, but ignores non-Gaussian structure)
- **Laplace Marginalization**: Spline-corrected Gaussian approximation (more accurate for non-Gaussian likelihoods)

## Basic Usage

```julia
using IntegratedNestedLaplace
using GaussianMarkovRandomFields
using Distributions

# Set up INLA model components
prior_gmrf = GMRF(μ_prior, Q_prior, CholeskySolverBlueprint())
obs_model = ExponentialFamily(Bernoulli)  # Non-Gaussian likelihood
θ = NamedTuple()  # Hyperparameters
y = [1, 0, 1, 0]  # Observed data

# Compute Gaussian approximation
ga_result = gaussian_approximation(prior_gmrf, obs_model, θ, y)
ga = to_gmrf(ga_result)

# Marginalize selected variables
result = marginalize(ga, obs_model, θ, y, 0.0, LaplaceMarginal(), [1, 3]; 
                    prior_gmrf=prior_gmrf)

# Access marginal distributions
marginal_1 = result.marginals[1]  # Marginal for variable 1
marginal_3 = result.marginals[2]  # Marginal for variable 3

# Use standard Distributions.jl interface
μ_1 = mean(marginal_1)
σ_1 = std(marginal_1)
p_1 = pdf(marginal_1, 0.5)
samples = rand(marginal_1, 1000)
```

## Marginalization Methods

### Gaussian Marginalization

```@docs
GaussianMarginal
```

**When to use:**
- Gaussian or nearly-Gaussian likelihoods
- When speed is critical
- As a baseline comparison

**Example:**
```julia
# Fast Gaussian marginalization
gaussian_result = marginalize(ga, obs_model, θ, y, log_prior_θ, GaussianMarginal())
```

### Laplace Marginalization

```@docs
LaplaceMarginal
```

**When to use:**
- Non-Gaussian likelihoods (Bernoulli, Poisson, etc.)
- When accuracy is important
- For skewed or heavy-tailed marginals

**Example:**
```julia
# Accurate Laplace marginalization
laplace_result = marginalize(ga, obs_model, θ, y, log_prior_θ, 
                           LaplaceMarginal(true), [1, 2, 5]; 
                           prior_gmrf=prior_gmrf)
```

**Normalization Options:**
- `LaplaceMarginal(true)`: Exact numerical integration (slower, more accurate)
- `LaplaceMarginal(false)`: Gauss-Hermite approximation (faster, default)

## Main Function

```@docs
marginalize
```

## Result Structure

```@docs
MarginalResult
```

The marginal distributions returned are standard Julia `Distribution` objects that support the full `Distributions.jl` interface:

- **Gaussian marginals**: `Normal{Float64}` distributions
- **Laplace marginals**: `SplineAugmentedGaussian{Float64}` distributions with lazy computation

## SplineAugmentedGaussian Distribution

For Laplace marginalization, the package provides a specialized distribution type:

```@docs
SplineAugmentedGaussian
```

This distribution implements the full `Distributions.jl` interface with high-performance lazy computation:

```julia
# All standard operations are supported
d = result.marginals[1]  # SplineAugmentedGaussian

# Statistical properties (computed efficiently via Gauss-Hermite quadrature)
μ = mean(d)      # Cached after first computation
σ = std(d)       # Cached after first computation
γ = skewness(d)  # Higher-order moments

# Density evaluation
p = pdf(d, x)
ℓ = logpdf(d, x)

# Cumulative distribution (computed via cached interpolation)
F = cdf(d, x)        # Cached spline after first computation
x_p = quantile(d, p) # Inverse CDF via cached spline

# Random sampling (efficient inverse transform method)
samples = rand(d, 1000)  # Uses cached quantile function
```

## Performance Notes

- **First call overhead**: Laplace methods compute expensive corrections on first use
- **Subsequent calls**: Very fast due to caching (splines, moments, etc.)
- **Memory efficient**: Only computes what's requested (moments OR quantiles)
- **Batch processing**: Marginalize multiple variables in a single call for efficiency

## Mathematical Details

### Gaussian Marginalization
Directly extracts marginals from the multivariate Gaussian approximation:
```
π̃_G(x_i | θ, y) = N(μ_i, Σ_ii)
```

### Laplace Marginalization  
Uses spline-corrected Gaussian approximation:
```
π̃_LA(x_i | θ, y) ≈ π̃_G(x_i | θ, y) × exp(spline(x_i))
```

where the spline correction accounts for non-Gaussian structure in the likelihood.

## Common Patterns

### Comparing Methods
```julia
# Compare Gaussian vs Laplace for validation
gauss_result = marginalize(ga, obs_model, θ, y, log_prior_θ, GaussianMarginal())
laplace_result = marginalize(ga, obs_model, θ, y, log_prior_θ, LaplaceMarginal(); 
                           prior_gmrf=prior_gmrf)

# For Gaussian likelihoods, these should be nearly identical
mean_diff = abs(mean(gauss_result.marginals[1]) - mean(laplace_result.marginals[1]))
```

### Posterior Inference
```julia
# Extract credible intervals
marginal = result.marginals[1]
ci_lower = quantile(marginal, 0.025)
ci_upper = quantile(marginal, 0.975)

# Posterior probability of positive effect
prob_positive = 1 - cdf(marginal, 0.0)
```

### Custom Variable Selection
```julia
# Marginalize specific variables of interest
n = length(mean(ga))
indices = [1, div(n,2), n]  # First, middle, last variables
result = marginalize(ga, obs_model, θ, y, log_prior_θ, LaplaceMarginal(), indices; 
                    prior_gmrf=prior_gmrf)
```

## See Also

- [`gaussian_approximation`](@ref): Computing the Gaussian approximation
- [`INLAModel`](@ref): Setting up complete INLA models
- [Observation Models](@ref observation-models): Different likelihood types