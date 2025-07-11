# End-to-End Integration Tests

This directory contains comprehensive integration tests for the INLA interface that validate the implementation against MCMC reference results.

## Directory Structure

```
test/end_to_end/
├── runtests.jl           # Test runner (includes all test cases)
├── ar1_poisson/          # AR-1 Poisson model test case
│   ├── test_fast.jl      # Fast CI test (uses reference data)
│   ├── test_full.jl      # Full test (runs live MCMC)
│   ├── generate_reference.jl  # MCMC reference data generator
│   ├── reference_data.jld2     # Pre-computed MCMC results
│   └── README.md         # Test case documentation
└── README.md             # This file
```

## Test Architecture

### Fast CI Tests (`*/test_fast.jl`)
- **Purpose**: Validates INLA implementation in CI environment
- **Speed**: ~30 seconds per test case
- **Data**: Uses pre-computed MCMC reference data
- **Coverage**: Tests hyperparameter and latent field marginals

### Reference Data Generators (`*/generate_reference.jl`)
- **Purpose**: Creates high-quality MCMC reference data
- **Speed**: ~10-15 minutes per test case
- **Output**: `reference_data.jld2` in each test case directory
- **Usage**: Run manually when model changes

### Full Tests (`*/test_full.jl`)
- **Purpose**: Complete test with live MCMC comparison
- **Speed**: ~3-5 minutes per test case
- **Usage**: Development testing and verification

## Workflow

### For Development
1. Generate reference data: `julia --project test/end_to_end/ar1_poisson/generate_reference.jl`
2. Run tests normally - they will use the fast version automatically

### For CI
- Tests automatically use pre-computed reference data
- No MCMC computation in CI environment
- Fast execution while maintaining validation quality

## Current Test Cases

### AR-1 Poisson Model (`ar1_poisson/`)
- **Challenge**: High autocorrelation (ρ ≈ 0.98) with nonlinear Poisson observations
- **Size**: 200 latent variables
- **Parameterization**: τ_gmrf_log (log-precision), η (atanh correlation)
- **Validation**: Hyperparameter and latent field marginals vs MCMC

## Adding New Test Cases

To add a new test case:

1. **Create directory**: `test/end_to_end/new_model/`
2. **Implement files**:
   - `test_fast.jl`: Fast CI test
   - `test_full.jl`: Full test with live MCMC
   - `generate_reference.jl`: Reference data generator
   - `README.md`: Test case documentation
3. **Update**: Add `include("new_model/test_fast.jl")` to `runtests.jl`
4. **Generate**: Run `generate_reference.jl` to create reference data

## Validation Approach

### Statistical Comparisons
- **Hyperparameters**: Compare posterior means and 95% credible intervals
- **Latent field**: Compare marginal means, standard deviations, and quantiles
- **Tolerances**: Conservative rtol values (10-25%) account for approximation differences

### Quality Checks
- **Convergence**: Verify mode finding and optimization success
- **Structure**: Validate result object types and dimensions
- **Performance**: Ensure INLA speed advantage over MCMC
- **Error handling**: Test edge cases and invalid inputs

## Maintenance

### Updating Reference Data
Regenerate reference data when:
- Model parameterization changes
- Bug fixes affect numerical results
- Test coverage needs expansion

```bash
cd test/end_to_end/ar1_poisson
julia --project generate_reference.jl
```

### Quality Assurance
- Reference data includes validation checks
- Tests verify data quality before use
- Comprehensive error handling for missing data

This architecture ensures thorough validation while keeping CI fast and supporting multiple test cases.