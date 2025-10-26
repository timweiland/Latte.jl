"""
Grid-based hyperparameter marginalization with adaptive exploration.

This module implements the grid-based approach to hyperparameter marginalization,
which builds an interpolant from exploration points and adapts the region based on
tail coverage diagnostics.
"""

# Include all submodules in dependency order
include("types.jl")
include("diagnostics.jl")
include("utils.jl")
include("expansion.jl")
include("grid_based_marginal.jl")

# Re-export main types and functions
export GridBasedMarginal, AsymmetricLogDropLimits
