# Development Guide

Welcome to Latte.jl development! This guide will help you get started with the development workflow.

## Quick Start

```bash
# 1. Setup development environment
make setup

# 2. Generate reference data for tests
make generate-reference

# 3. Run tests
make test
```

## Development Workflow

### Initial Setup

```bash
# Install dependencies and TestEnv
make setup

# Generate MCMC reference data (takes 10-15 minutes)
make generate-reference
```

### Daily Development

```bash
# Run tests
make test

# Format code before committing
make format

# Run full tests occasionally (3-5 minutes)
make test-full
```

### Reference Data Management

The test suite uses pre-computed MCMC reference data for fast CI testing:

```bash
# Generate all reference data
make generate-reference

# Generate specific test case
make ar1-poisson-ref

# Clean and regenerate
make clean
make generate-reference
```

## Testing Architecture

### Fast Tests (CI)
- **Command**: `make test` or `make test-fast`
- **Runtime**: ~30 seconds
- **Data**: Pre-computed MCMC reference data
- **Usage**: Regular development and CI

### Full Tests (Development)
- **Command**: `make test-full`
- **Runtime**: ~3-5 minutes
- **Data**: Live MCMC computation
- **Usage**: Thorough validation during development

### Reference Generation
- **Command**: `make generate-reference`
- **Runtime**: ~10-15 minutes
- **Purpose**: Create high-quality MCMC reference data
- **Usage**: When updating models or test cases

## Make Commands

### Setup
- `make setup` - Install dependencies and TestEnv
- `make deps` - Install package dependencies only

### Testing
- `make test` - Run tests - defaults to `test-fast`
- `make test-fast` - Run fast tests with pre-computed MCMC reference
- `make test-full` - Run full tests with live MCMC

### Reference Data
- `make generate-reference` - Generate all MCMC reference data
- `make ar1-poisson-ref` - Generate AR-1 Poisson reference data

### Development
- `make clean` - Clean generated files
- `make format` - Format code with Runic
- `make docs` - Build documentation
- `make help` - Show all available commands

## Adding New Test Cases

To add a new end-to-end test case:

1. **Create directory**: `test/end_to_end/new_model/`
2. **Implement files**:
   ```
   test/end_to_end/new_model/
   ├── test_fast.jl              # Fast CI test
   ├── test_full.jl              # Full test with live MCMC
   ├── generate_reference.jl     # Reference data generator
   └── README.md                 # Test case documentation
   ```
3. **Add to Makefile**: Add target for new reference generation
4. **Update test runner**: Add `include("new_model/test_fast.jl")` to `test/end_to_end/runtests.jl`
5. **Generate reference**: `make new-model-ref`

## Dependencies

### Package Dependencies
- Core INLA implementation dependencies
- Managed via `Project.toml`

### Test Dependencies
- `JLD2` - Reference data storage
- `Turing` - MCMC reference computation
- `TestEnv` - Test environment management
- Managed via `Project.toml` extras

### Global Dependencies
- `TestEnv` - Automatically installed by `make setup`
- `Runic` - Code formatting

## Common Issues

### Missing TestEnv
```bash
# Error: TestEnv not found
make setup
```

### Missing Reference Data
```bash
# Error: reference_data.jld2 not found
make generate-reference
```

### Stale Reference Data
```bash
# After model changes
make clean
make generate-reference
```

## CI Integration

The CI system uses the fast test approach:
- No MCMC computation in CI
- Pre-computed reference data committed to repo
- Fast execution while maintaining validation quality

## Performance Expectations

- **Fast tests**: ~30 seconds total
- **Full tests**: ~3-5 minutes total
- **Reference generation**: ~10-15 minutes per test case
- **INLA inference**: ~1-2 seconds per test case

## File Organization

```
├── Makefile                  # Development workflow
├── DEVELOPMENT.md           # This file
├── src/                     # Package source code
├── test/
│   ├── end_to_end/         # Integration tests
│   │   ├── ar1_poisson/    # AR-1 Poisson test case
│   │   └── runtests.jl     # Test runner
│   └── */                  # Unit tests
└── docs/                   # Documentation
```

This workflow ensures efficient development while maintaining thorough validation of the INLA implementation.
