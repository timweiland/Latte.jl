# AR-1 Poisson Model Test Case

This test case validates the INLA implementation on an AR-1 Poisson model with challenging high autocorrelation.

## Model Description

**Latent Field**: AR-1 Gaussian Markov Random Field
- Precision matrix: `Q = ar_precision(ρ, k) * τ`
- Mean structure: `μ = μ₀ * [ρⁱ for i in 1:k]` (exponential decay)
- Hyperparameters: `τ_gmrf_log = log(1/σ²)`, `η = atanh(ρ)`

**Observations**: Poisson with log-link
- `y[i] ~ Poisson(exp(x[i]))`
- Problem size: k=200 latent variables

**Hyperparameter Priors**:
- `τ_gmrf_log ~ Normal(0, 1)`
- `η ~ Normal(atanh(0.95), 0.5*(atanh(0.98) - atanh(0.95)))`

## Test Structure

### `test_fast.jl`
- **Runtime**: ~30 seconds
- **Data**: Pre-computed MCMC reference from `reference_data.jld2`
- **Usage**: Automatic in CI via `test/end_to_end/runtests.jl`

### `test_full.jl` 
- **Runtime**: ~3-5 minutes
- **Data**: Runs live MCMC for comparison (800 samples)
- **Usage**: Development testing and debugging

### `generate_reference.jl`
- **Runtime**: ~10-15 minutes
- **Purpose**: Creates high-quality reference data (2000×4 MCMC samples)
- **Usage**: Run manually when updating reference data

## Files

- `test_fast.jl`: Fast CI test using pre-computed reference
- `test_full.jl`: Full test with live MCMC comparison
- `generate_reference.jl`: MCMC reference data generator
- `reference_data.jld2`: Pre-computed MCMC results (generated)
- `README.md`: This documentation

## Usage

### Running Tests
```bash
# Fast test (CI)
julia --project test/end_to_end/ar1_poisson/test_fast.jl

# Full test (development)
julia --project test/end_to_end/ar1_poisson/test_full.jl
```

### Generating Reference Data
```bash
cd test/end_to_end/ar1_poisson
julia --project generate_reference.jl
```

## Validation Approach

### Statistical Comparisons
- **Hyperparameters**: Compare means and 95% credible intervals
- **Latent Field**: Test 6 representative indices across the field
- **Metrics**: Posterior means, standard deviations, quantiles

### Quality Assurance
- **Convergence**: Mode finding success
- **Performance**: INLA speed advantage over MCMC
- **Robustness**: Error handling and edge cases
- **Approximation**: LaplaceMarginal vs GaussianMarginal methods

## Test Challenge

This model provides a rigorous test because:
- **Nonlinear likelihood**: Poisson observations test approximation quality
- **Large dimension**: 200 latent variables test scalability
