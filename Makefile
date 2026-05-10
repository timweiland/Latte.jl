# IntegratedNestedLaplace.jl Development Makefile

.PHONY: help setup test test-fast test-full generate-reference clean docs docs-rebuild docs-skip docs-serve docs-preview logo

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
	@echo "  make docs            - Build documentation (incremental tutorials)"
	@echo "  make docs-rebuild    - Build docs, force rebuild every tutorial"
	@echo "  make docs-skip       - Build docs, skip every tutorial"
	@echo "  make docs-serve      - Vitepress dev server with hot reload (after a build)"
	@echo "  make docs-preview    - Serve the built static site for preview"
	@echo "  make logo            - Generate logo"

# Setup development environment
setup: deps
	@echo "Setting up development environment..."
	@julia -e 'using Pkg; Pkg.add("TestEnv"); Pkg.add("Runic")'
	@echo "Setup complete!"
	@echo "Consider also adding the Runic shell script. Check https://github.com/fredrikekre/Runic.jl"

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
	@runic --inplace .

# Build documentation. Tutorials only rebuild when their .jl source is newer
# than the .md output (mtime cache); force a full rebuild after touching
# package internals via `make docs-rebuild`, or skip tutorials entirely with
# `make docs-skip` when iterating on prose / theme / layout only.
docs:
	@echo "Building documentation (incremental tutorials)..."
	@julia --project=docs docs/make.jl

docs-rebuild:
	@echo "Building documentation (full tutorial rebuild)..."
	@LATTE_REBUILD_TUTORIALS=1 julia --project=docs docs/make.jl

docs-skip:
	@echo "Building documentation (tutorials skipped)..."
	@LATTE_SKIP_TUTORIALS=1 julia --project=docs docs/make.jl

# Vitepress dev server with hot reload. Edits to .vue/.md/.css files in
# docs/src/ trigger an automatic page refresh. Requires a build to have
# run at least once (so build/.documenter/ exists).
docs-serve:
	@echo "Starting Vitepress dev server (hot reload). Ctrl-C to stop."
	@cd docs && npm run docs:dev

# Static preview of the most recent built site.
docs-preview:
	@echo "Serving built docs (no hot reload). Ctrl-C to stop."
	@cd docs && npm run docs:preview

docs-deps:
	@echo "Instantiating docs environment..."
	@julia --project=docs -e 'using Pkg; Pkg.instantiate(); Pkg.update()'
	@echo "Done!"

docs-server:
	@echo "Starting VitePress docs server..."
	@cd docs && julia --project -e 'using DocumenterVitepress; DocumenterVitepress.dev_docs("build")'

# Generate logo
logo:
	@echo "Generating logo..."
	@cd docs/logo-generation && julia --project -e 'include("draw_logo.jl")'
	@echo "Logo generated at docs/src/assets/logo.svg"

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
