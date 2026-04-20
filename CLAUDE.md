# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Latte.jl is a Julia package for Integrated Nested Laplace Approximation (INLA), a Bayesian inference method. The package is currently in early development (v1.0.0-DEV) and uses the Julia package ecosystem structure.

## Common Commands

### Testing
```bash
# Run all tests
make test
```

### Development Workflow
```bash
# Set up development environment
make setup

# Generate MCMC reference data for tests
make reference-data

# Build documentation
make docs

# Format code
make format
```

**Automated Formatting**: This project includes a Claude Code hook (`.claude/hooks.json`) that automatically runs the formatter after editing Julia files. The hook triggers on `Edit`, `Write`, and `MultiEdit` operations for `.jl` files.

## Testing Tips

- To run a specific test, do `julia --project`, followed by `using TestEnv; TestEnv.activate(); include("...")`, where "..." contains the test file with the tests to run

## Architecture

- **Main Interface**: `src/main_interface/` - Unified INLA inference interface
  - `inla_inference.jl`: Main `inla()` function with progress tracking
  - `types.jl`: INLAResult and related types
  - `progress.jl`: Progress tracking system with ProgressMeter.jl integration
  - `validation.jl`: Input validation and error handling
- **Latent Marginalization**: `src/latent_marginalization/` - Marginalization over latent field variables
  - `gaussian_marginal.jl`: Gaussian marginal computations
  - `laplace/`: Laplace approximation methods for marginals with spline augmentation
  - `simplified_laplace.jl`: Simplified Laplace approximation
- **Hyperparameter Posterior**: `src/hyperparameter_posterior/` - Hyperparameter posterior exploration and marginalization
  - `mode_finding.jl`: Mode finding with progress callbacks and initial guess computation
  - `exploration/`: Grid-based exploration with reparameterization
  - `interpolation.jl`: Posterior interpolation with adaptive method selection
  - `hyperparameter_marginals.jl`: Marginal computation over hyperparameters
  - `hyperparameter_marginal_distribution.jl`: Distribution interface for hyperparameter marginals
- **Hyperparameters**: `src/hyperparameters/` - Hyperparameter prior specification
- **Distributions**: `src/distributions/` - Custom distributions including WeightedMixture
- **Tests**: Comprehensive test suite with end-to-end validation
  - `test/end_to_end/`: Integration tests with MCMC reference data via Git LFS
  - `test/hyperparameter_posterior/`: Unit tests for hyperparameter posterior methods
- **Documentation**: `docs/` with Documenter.jl setup, includes main interface documentation
- **Examples**: `examples/` contains clean, documented examples for various model types

**Note**: Observation models, Gaussian approximation, and link functions are provided by GaussianMarkovRandomFields.jl v0.4+ and re-exported by this package for user convenience.

## Key Dependencies

The package relies on several specialized Julia packages:
- **GaussianMarkovRandomFields.jl** (v0.4+): Core GMRF functionality including observation models, Gaussian approximation, link functions, and automatic differentiation support
- **Distributions.jl**, **StatsFuns.jl**: Statistical distributions and functions
- **FiniteDiff.jl**: Finite difference methods for numerical differentiation
- **LinearSolve.jl**: Linear system solvers for sparse and dense systems
- **ProgressMeter.jl**: Progress tracking with rich callbacks
- **Optim.jl**: Optimization algorithms for mode finding
- **DataInterpolations.jl**: 1D interpolation (CubicSpline, LinearInterpolation, ConstantInterpolation)
- **ScatteredInterpolation.jl**: Multidimensional RBF interpolation
- **HCubature.jl**: Numerical integration for marginal computation
- **Turing.jl**: Probabilistic programming framework for validation and comparisons (test dependency)
- **JLD2.jl**: Serialization for reference data storage in tests (test dependency)

## Development Notes

- The package follows Julia package conventions with Project.toml/Manifest.toml
- CI runs on Julia 1.0, 1.10, and nightly versions
- Uses Aqua.jl for code quality testing
- Documentation automatically deploys via GitHub Actions
- Exports are organized per module rather than centralized
- All public functions have comprehensive docstrings following Julia conventions

## Implementation Insights

### GMRF v0.4 Integration
- **Observation models**: This package re-exports observation model types from GaussianMarkovRandomFields.jl v0.4+, including `ExponentialFamily`, `LinearlyTransformedObservationModel`, `CompositeObservationModel`, and link functions
- **Gaussian approximation**: The `gaussian_approximation` function is provided by GMRF and re-exported for convenience
- **API compatibility**: All GMRF types are re-exported with the same names to preserve user-facing API compatibility

### Testing Strategy
- **Modular**: Separate test files per component (INLA model, latent marginalization, hyperparameter methods, etc.)
- **End-to-end validation**: Integration tests against MCMC reference data using Turing.jl
- **Type stability**: Use `@inferred` with anonymous functions for broadcasting type checks
- **Deterministic tests**: Use `Random.seed!` to ensure reproducibility of randomized tests
- **GMRF integration**: Tests verify correct usage of GMRF v0.4 observation models and gaussian approximation

### Documentation Structure
- **Reference organization**: Separate pages per major component with custom IDs
- **API coverage**: All public-facing functions documented with `@docs` blocks
- **Examples**: Both usage examples and mathematical background in documentation
- **Cross-references**: Proper linking between related functions and types

### Task Management
This project uses an org-mode task system for tracking development work. Tasks are organized in the `tasks/` directory.

## Task Organization

```
tasks/
├── task-1-example-a.org
├── task-2-example-b.org
├── ...
```

## Task Structure

Each task file follows this org-mode structure:
- **Overview**: Description of the high-level goal
- **Subtasks**: Individual work items with TODO/DONE status
- **Acceptance Criteria**: Testable outcomes that define completion
- **Implementation Plan**: Steps to achieve the task
- **Implementation Notes**: Documentation of work completed

## Working with Tasks

### Viewing Tasks
```bash
# View all tasks in a file
cat tasks/task-1-testing.org

# Find TODO items across all tasks
grep -r "TODO" tasks/
```

### Updating Task Status
Edit the org files directly to:
- Change `TODO` to `DONE` when completing subtasks
- Add implementation notes and file changes
- Update acceptance criteria checkboxes `- [ ]` to `- [x]`
- Add CLOSED timestamps for completed tasks

### Definition of Done
A task is complete when:
1. All acceptance criteria are checked off `- [x]`
2. Implementation notes document the approach and changes
3. All tests pass and code is properly formatted
4. Relevant documentation is updated
