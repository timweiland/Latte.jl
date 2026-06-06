# [Main Interface](@id main-interface)

Latte's unified interface: **define a latent Gaussian model once**, run it through
any inference engine — [`inla`](@ref), [`tmb`](@ref), or [`hmc_laplace`](@ref) — and
work with the results through a single API.

For the methods themselves and when to reach for each, see the engine pages,
starting with [INLA](engines/inla.md). This page covers **defining models**,
**calling an engine**, and **working with results**.

## Quick Start

```julia
using Latte
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

# Latent field: AR-1 GMRF with exponential decay mean
# Uses keyword arguments matching hyperparameter names
function latent_gmrf(; τ, ρ, kwargs...)
    Q = ar1_precision(ρ, k) .* τ
    μ₀ = log(1000.0)             # Log-scale for Poisson rates
    μ = μ₀ .* [ρ^i for i in 1:k] # Exponential decay
    return (μ, Q)                # FunctionLatentModel expects (mean, precision)
end

# Observation model: Poisson with log-link
obs_model = ExponentialFamily(Poisson)
# Wrap the latent function with its output dimension
model = LatentGaussianModel(spec, FunctionLatentModel(latent_gmrf, k), obs_model)

# Run INLA on your data
result = inla(model, y_observed)

# Access results
hyperparameter_marginals = result.hyperparameter_marginals
latent_marginals = result.latent_marginals
posterior_mode = result.hyperparameter_mode
```

## Running an engine

The full `inla` reference and the method itself live on the [INLA](engines/inla.md)
page (and the other engine pages). The short version is just:

```julia
result = inla(model, y)        # or tmb(model, y), or hmc_laplace(model, y)
```

## Defining models

Models are written with the `@latte` macro, or — for a full DynamicPPL model —
converted with `latte_from_dppl`. Both produce the `LatentGaussianModel` that
`inla` consumes.

```@docs
@latte
latte_from_dppl
```

## Alternative inference engines

The same model can be run through other engines for comparison and validation:

```@docs
tmb
hmc_laplace
```

## Diagnostics

```@docs
diagnose
```

## Understanding the Results

The `inla` function returns an [`INLAResult`](@ref) object containing all posterior information:

### Hyperparameter Marginals
Posterior marginal distributions for model hyperparameters (in natural space):

```julia
result = inla(model, y)

# Access marginals by index (already in natural space)
τ_marginal = result.hyperparameter_marginals[1]  # Precision (natural space)
ρ_marginal = result.hyperparameter_marginals[2]  # Correlation (natural space)

# Compute posterior summaries directly (no transformation needed)
precision_mean = mean(τ_marginal)
correlation_mean = mean(ρ_marginal)

# Credible intervals
precision_ci = quantile(τ_marginal, [0.025, 0.975])
correlation_ci = quantile(ρ_marginal, [0.025, 0.975])
```

### Latent Field Marginals  
Posterior marginal distributions for each latent field component:

```julia
# Extract posterior means and uncertainties
posterior_means = [mean(m) for m in result.latent_marginals]
posterior_stds = [std(m) for m in result.latent_marginals]

# Credible intervals for each component
credible_intervals = [quantile(m, [0.025, 0.975]) for m in result.latent_marginals]

# Transform to original scale (for Poisson: exp for rates)
poisson_rates_mean = exp.(posterior_means)
```

### Convergence Diagnostics
Always check that the optimization converged:

```julia
if result.convergence.mode_converged
    println("✓ Mode finding converged successfully")
    println("Final log-likelihood: $(result.convergence.final_loglik)")
else
    @warn "Mode finding did not converge - check model specification"
end
```

## Complete Example: AR-1 Time Series with Count Data

This example demonstrates INLA on a realistic time series model with count observations:

```julia
using Latte
using GaussianMarkovRandomFields
using Distributions
using SparseArrays
using Random

# Set reproducible seed
Random.seed!(123)

# AR-1 precision matrix function
function ar1_precision(ρ, k)
    return spdiagm(-1 => -ρ * ones(k - 1), 0 => ones(k) .+ ρ^2, 1 => -ρ * ones(k - 1))
end

# Model setup
k = 200  # Time series length

# Define hyperparameters with @hyperparams macro
spec = @hyperparams begin
    (τ ~ Exponential(1.0), transform = log, space = natural)  # Precision
    (ρ ~ Beta(5, 1), transform = logit, space = natural)      # Autocorrelation
end

# Latent GMRF definition (uses keyword arguments)
function latent_gmrf(; τ, ρ, kwargs...)
    # AR-1 precision matrix
    Q = ar1_precision(ρ, k) .* τ

    # Mean structure: exponentially decaying (typical for AR models)
    μ₀ = log(1000.0)             # Base log-rate
    μ = μ₀ .* [ρ^i for i in 1:k] # Exponential decay

    return (μ, Q)                # FunctionLatentModel expects (mean, precision)
end

# Poisson observations with log-link (canonical)
obs_model = ExponentialFamily(Poisson)

# Complete model specification — wrap the latent function with its dimension
model = LatentGaussianModel(spec, FunctionLatentModel(latent_gmrf, k), obs_model)

# Generate synthetic data for demonstration
true_params = (τ = 10.0, ρ = 0.98)
x_true = rand(GMRF(latent_gmrf(; true_params...)...))
y_observed = rand.(Poisson.(exp.(x_true)))

println("Generated $(length(y_observed)) observations")
println("Observation range: $(minimum(y_observed)) to $(maximum(y_observed))")

# Run INLA inference with progress tracking
println("Running INLA inference...")
result = inla(model, y_observed, progress=true)

# Check convergence
if result.convergence.mode_converged
    println("✓ INLA converged successfully!")
else
    @warn "INLA did not converge - check model specification"
end

# Extract posterior summaries (marginals are in natural space)
τ_posterior = result.hyperparameter_marginals[1]  # Precision
ρ_posterior = result.hyperparameter_marginals[2]  # Correlation

# Compute posterior means
precision_mean = mean(τ_posterior)
correlation_mean = mean(ρ_posterior)

println("\\nPosterior Results:")
println("Precision (τ):")
println("  Mean: $(round(precision_mean, digits=2))")
println("  True: $(round(true_params.τ, digits=2))")

println("Correlation (ρ):")
println("  Mean: $(round(correlation_mean, digits=3))")
println("  True: $(round(true_params.ρ, digits=3))")

# Analyze latent field
latent_means = [mean(m) for m in result.latent_marginals]
latent_stds = [std(m) for m in result.latent_marginals]

println("\\nLatent Field Summary:")
println("Mean absolute error: $(round(mean(abs.(latent_means - x_true)), digits=3))")
println("Average posterior std: $(round(mean(latent_stds), digits=3))")

# Compare with true values (if available)
correlation = cor(latent_means, x_true)
println("Correlation with truth: $(round(correlation, digits=3))")
```

## Advanced Usage

### Different Marginalization Methods

INLA offers different methods for computing latent field marginals:

```julia
# Gaussian marginalization (faster, good for most cases)
result_gaussian = inla(model, y_observed, marginalization_method=GaussianMarginal())

# Laplace marginalization (more accurate, especially for non-Gaussian posteriors)
result_laplace = inla(model, y_observed, marginalization_method=LaplaceMarginal())

# Compare the methods
i = 50  # Example latent component
gauss_mean = mean(result_gaussian.latent_marginals[i])
laplace_mean = mean(result_laplace.latent_marginals[i])
println("Gaussian mean: $(round(gauss_mean, digits=3))")
println("Laplace mean: $(round(laplace_mean, digits=3))")
```

### Subset Marginalization for Large Models

For computational efficiency, compute marginals only where needed:

```julia
# Only compute marginals for every 10th component
indices_of_interest = 1:10:k
result = inla(model, y_observed, latent_indices=indices_of_interest)

# result.latent_marginals now has length 20, not 200
subset_means = [mean(m) for m in result.latent_marginals]
```

### Progress Control

```julia
# Run with progress bar (default)
result = inla(model, y_observed, progress=true)

# Run silently (useful for batch processing)
result = inla(model, y_observed, progress=false)
```

## Best Practices

### Parameterization
- **Use appropriate transforms**: The `@hyperparams` macro handles transformations automatically
  - `transform = log` for positive parameters (e.g., precision, variance)
  - `transform = logit` for bounded parameters in (0,1) (e.g., correlations, proportions)
- **Specify prior space**: Use `space = natural` to specify priors in the natural parameter space
- **Informative priors**: Weakly informative priors often work better than flat priors

### Model Specification
- **Keyword arguments**: Latent field functions should use keyword arguments matching hyperparameter names
- **Check scales**: Ensure data and parameters are on reasonable scales
- **Sparse precision**: Use sparse matrices for large problems

### Diagnostics
- **Always check convergence**: `result.convergence.mode_converged`
- **Examine posterior modes**: Should be in reasonable ranges
- **Compare methods**: Try both Gaussian and Laplace marginalization
- **Cross-validate**: Compare predictions with held-out data

### Performance Tips
- **Problem size**: INLA scales well to thousands of latent variables
- **Subset marginalization**: Use `latent_indices` for large models
- **Sparse operations**: Ensure GMRF precision matrices are sparse
- **Batch processing**: Process multiple datasets efficiently

## Common Pitfalls

### Convergence Issues
```julia
# Check if optimization failed
if !result.convergence.mode_converged
    # Try different priors (wider, less informative)
    spec_relaxed = @hyperparams begin
        (τ ~ Exponential(0.1), transform = log, space = natural)  # Wider prior
        (ρ ~ Beta(2, 2), transform = logit, space = natural)      # Less informative
    end
    result = inla(LatentGaussianModel(spec_relaxed, FunctionLatentModel(latent_gmrf, k), obs_model), y_observed)
end
```

### Numerical Instability
```julia
# Check for extreme parameter values
mode_values = result.hyperparameter_mode
if any(abs.(mode_values) .> 10)
    @warn "Extreme parameter values detected - consider reparameterization"
end
```

### Poor Approximation Quality
```julia
# Compare Gaussian vs Laplace marginalization
result_g = inla(model, y, marginalization_method=GaussianMarginal())
result_l = inla(model, y, marginalization_method=LaplaceMarginal())

# Large differences suggest Gaussian approximation may be poor
mean_diff = mean(abs.([mean(m) for m in result_g.latent_marginals] - 
                     [mean(m) for m in result_l.latent_marginals]))
if mean_diff > 0.1
    @warn "Large difference between marginalization methods"
    println("Consider using LaplaceMarginal() for better accuracy")
end
```

## API Reference

```@docs
INLAResult
```