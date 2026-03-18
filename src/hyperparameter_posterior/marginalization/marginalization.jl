"""
Hyperparameter marginalization module.

Provides abstractions and implementations for computing hyperparameter marginal
distributions from exploration results.

This is step 3 in the INLA workflow:
1. Mode finding
2. Exploration (creates coarse grid for latent integration)
3. Hyperparameter marginalization (THIS MODULE)
4. Latent marginalization (uses exploration from step 2)
"""

# Abstract interface
include("types.jl")

# Concrete implementations
include("spline_based/spline_based.jl")

# Re-export main interface and types
export HyperparameterMarginalizationMethod, marginalize_hyperparameters
export GridSumMarginal, CCDInterpolantMarginal, AutoHyperparameterMarginal
