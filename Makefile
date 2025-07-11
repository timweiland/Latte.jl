# IntegratedNestedLaplace.jl Development Makefile

.PHONY: help setup test test-fast test-full generate-reference clean docs

# Default target
help:
	@echo "IntegratedNestedLaplace.jl Development Commands"
	@echo "=============================================="
	@echo ""
	@echo "Setup:"
	@echo "  make setup           - Install dependencies and TestEnv"
	@echo "  make deps            - Install package dependencies only"
	@echo ""
	@echo "Testing:"
	@echo "  make test            - Run all tests (fast CI version)"
	@echo "  make test-fast       - Run fast tests with pre-computed reference"
	@echo "  make test-full       - Run full tests with live MCMC"
	@echo ""
	@echo "Reference Data:"
	@echo "  make generate-reference - Generate all MCMC reference data"
	@echo "  make ar1-poisson-ref    - Generate AR-1 Poisson reference data"
	@echo ""
	@echo "Development:"
	@echo "  make clean           - Clean generated files"
	@echo "  make format          - Format code with Runic"
	@echo "  make docs            - Build documentation"

# Setup development environment
setup: deps
	@echo "Setting up development environment..."
	@julia -e 'using Pkg; Pkg.add("TestEnv")'
	@echo "Setup complete!"

# Install package dependencies
deps:
	@echo "Installing package dependencies..."
	@julia --project -e 'using Pkg; Pkg.instantiate()'
	@echo "Dependencies installed!"

# Run all tests (fast version)
test: test-fast

# Run fast tests with pre-computed reference data
test-fast:
	@echo "Running fast tests with pre-computed reference data..."
	@julia --project -e 'using Pkg; Pkg.test()'

# Run full tests with live MCMC
test-full:
	@echo "Running full tests with live MCMC..."
	@julia --project -e 'using Test; include("test/end_to_end/ar1_poisson/test_full.jl")'

# Generate all MCMC reference data
generate-reference: ar1-poisson-ref
	@echo "All reference data generated!"

# Generate AR-1 Poisson reference data
ar1-poisson-ref:
	@echo "Generating AR-1 Poisson MCMC reference data..."
	@echo "This will take 10-15 minutes..."
	@cd test/end_to_end/ar1_poisson && julia --project generate_reference.jl
	@echo "AR-1 Poisson reference data generated!"

# Format code
format:
	@echo "Formatting code with Runic..."
	@julia -e 'using Runic; Runic.format(".", verbose=true)'

# Build documentation
docs:
	@echo "Building documentation..."
	@julia --project=docs docs/make.jl

docs-deps:
	@echo "Instantiating docs environment..."
	@julia --project=docs -e 'using Pkg; Pkg.instantiate(); Pkg.update()'
	@echo "Done!"

docs-server:
	@echo "Starting docs server..."
	@julia -e 'using LiveServer; serve(dir="docs/build")'

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	@rm -f test/end_to_end/ar1_poisson/reference_data.jld2
	@rm -rf docs/build/
	@echo "Clean complete!"

# Check if TestEnv is installed (utility target)
check-testenv:
	@julia -e 'try; using TestEnv; catch; println("TestEnv not found - run: make setup"); exit(1); end'

# Verify reference data exists
check-reference:
	@if [ ! -f test/end_to_end/ar1_poisson/reference_data.jld2 ]; then \
		echo "Reference data not found. Run: make generate-reference"; \
		exit 1; \
	fi
