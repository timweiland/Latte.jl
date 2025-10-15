# [Main Interface](@id main-interface)

The [`inla`](@ref) function is the primary entry point for performing Integrated Nested Laplace Approximation. It provides a unified, user-friendly interface that handles all aspects of INLA inference automatically.

## What is INLA?

Integrated Nested Laplace Approximation (INLA) is a method for fast approximate Bayesian inference in latent Gaussian models. Instead of using expensive MCMC sampling, INLA uses clever analytical approximations and numerical integration to compute posterior marginals directly.

**Key advantages:**
- **Speed**: Orders of magnitude faster than MCMC
- **Accuracy**: High-quality approximations for most models  
- **Deterministic**: Same results every run (no MCMC uncertainty)
- **Automatic**: Minimal tuning required

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

# Model parameters (use proper INLA parameterization)
k = 100
θ_prior = HyperparameterPrior((
    τ_gmrf_log = Normal(0, 1),                               # Log precision
    η = Normal(atanh(0.95), 0.5 * (atanh(0.98) - atanh(0.95)))  # atanh(correlation)
))

# Latent field: AR-1 GMRF with exponential decay mean
function latent_gmrf(θ)
    τ = exp(θ.τ_gmrf_log)        # Transform to precision
    ρ = tanh(θ.η)                # Transform to correlation
    
    Q = ar1_precision(ρ, k) .* τ
    μ₀ = log(1000.0)             # Log-scale for Poisson rates
    μ = μ₀ .* [ρ^i for i in 1:k] # Exponential decay
    
    return GMRF(μ, Q)
end

# Observation model: Poisson with log-link
obs_model = ExponentialFamily(Poisson)
model = INLAModel(θ_prior, latent_gmrf, obs_model)

# Run INLA on your data
result = inla(model, y_observed)

# Access results
hyperparameter_marginals = result.hyperparameter_marginals  
latent_marginals = result.latent_marginals
posterior_mode = result.hyperparameter_mode
```

## The `inla` Function

```@docs
inla
```

## Understanding the Results

The `inla` function returns an [`INLAResult`](@ref) object containing all posterior information:

### Hyperparameter Marginals
Posterior marginal distributions for model hyperparameters:

```julia
result = inla(model, y)

# Access marginals (in transformed space)
τ_log_marginal = result.hyperparameter_marginals[1]  # Log precision
η_marginal = result.hyperparameter_marginals[2]      # atanh(correlation)

# Compute posterior summaries
log_precision_mean = mean(τ_log_marginal)
correlation_mode = tanh(result.hyperparameter_mode[2])  # Transform back

# Credible intervals
log_prec_ci = quantile(τ_log_marginal, [0.025, 0.975])
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
using IntegratedNestedLaplace
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

# Model setup with proper INLA parameterization
k = 200  # Time series length

# Hyperparameter priors (in transformed space for numerical stability)
θ_prior = HyperparameterPrior((
    τ_gmrf_log = Normal(0, 1),  # Log precision: exp(τ_gmrf_log) = 1/σ²
    η = Normal(atanh(0.95), 0.5 * (atanh(0.98) - atanh(0.95)))  # atanh(ρ)
))

# Latent GMRF definition
function latent_gmrf(θ)
    # Transform parameters back to natural scale
    τ = exp(θ.τ_gmrf_log)        # Precision
    ρ = tanh(θ.η)                # Correlation in (-1, 1)
    
    # AR-1 precision matrix
    Q = ar1_precision(ρ, k) .* τ
    
    # Mean structure: exponentially decaying (typical for AR models)
    μ₀ = log(1000.0)             # Base log-rate  
    μ = μ₀ .* [ρ^i for i in 1:k] # Exponential decay
    
    return GMRF(μ, Q)
end

# Poisson observations with log-link (canonical)
obs_model = ExponentialFamily(Poisson)

# Complete model specification
model = INLAModel(θ_prior, latent_gmrf, obs_model)

# Generate synthetic data for demonstration
true_params = (τ_gmrf_log = log(1/0.3^2), η = atanh(0.98))
x_true = rand(latent_gmrf(true_params))
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

# Extract posterior summaries
τ_log_posterior = result.hyperparameter_marginals[1]
η_posterior = result.hyperparameter_marginals[2]

# Transform back to interpretable scale
precision_mode = exp(result.hyperparameter_mode[1])
correlation_mode = tanh(result.hyperparameter_mode[2])
precision_mean = exp(mean(τ_log_posterior))
correlation_mean = tanh(mean(η_posterior))

println("\\nPosterior Results:")
println("Precision (1/σ²):")
println("  Mode: $(round(precision_mode, digits=2))")  
println("  Mean: $(round(precision_mean, digits=2))")
println("  True: $(round(exp(true_params.τ_gmrf_log), digits=2))")

println("Correlation (ρ):")
println("  Mode: $(round(correlation_mode, digits=3))")
println("  Mean: $(round(correlation_mean, digits=3))")  
println("  True: $(round(tanh(true_params.η), digits=3))")

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
- **Use log-scale for positive parameters**: `τ_log` instead of `τ` for precision
- **Use transforms for bounded parameters**: `atanh(ρ)` for correlations in (-1,1)
- **Center parameters**: Use `Normal(0, σ)` priors for transformed parameters

### Model Specification
- **Informative priors**: Weakly informative priors often work better than flat priors
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
    # Try different hyperparameter bounds
    θ_prior_relaxed = HyperparameterPrior((
        τ_gmrf_log = Normal(0, 2),  # Wider prior
        η = Normal(0, 1)            # Less informative
    ))
    result = inla(INLAModel(θ_prior_relaxed, latent_gmrf, obs_model), y_observed)
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