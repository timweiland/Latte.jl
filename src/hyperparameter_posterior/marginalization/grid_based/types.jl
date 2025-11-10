"""
Types and data structures for grid-based hyperparameter marginalization.
"""

export GridBasedMarginal, AsymmetricLogDropLimits

"""
    AsymmetricLogDropLimits

Tracks separate max_log_drop values for each hyperparameter dimension and direction.

This allows asymmetric expansion: if the right tail of a hyperparameter is heavy
but the left tail is light, we only expand to the right.

# Fields
- `limits::Matrix{Float64}`: [n_dim × 2] matrix where limits[i, 1] is the max log drop
  for dimension i in the negative direction, and limits[i, 2] is for the positive direction
"""
struct AsymmetricLogDropLimits
    limits::Matrix{Float64}  # [n_dim × 2]

    function AsymmetricLogDropLimits(n_dim::Int, initial_value::Float64)
        return new(fill(initial_value, n_dim, 2))
    end

    function AsymmetricLogDropLimits(limits::Matrix{Float64})
        size(limits, 2) == 2 || error("limits must have exactly 2 columns (neg/pos directions)")
        return new(limits)
    end
end

"""
    GridBasedMarginal <: HyperparameterMarginalizationMethod

Grid-based hyperparameter marginalization with adaptive exploration.

This method:
1. Builds an interpolant from the exploration grid points
2. Computes 1D marginals by numerically integrating the interpolant
3. Diagnoses tail coverage for each marginal
4. Adaptively expands the exploration region if needed (can be asymmetric)
5. Repeats until convergence or maximum iterations

# Fields
- `log_drop_increment::Float64`: How much to increase max_log_drop when expanding (default: 2.0)
- `max_log_drop_cap::Float64`: Hard limit on max_log_drop (default: 20.0)
- `allow_asymmetric::Bool`: Allow different limits per direction (default: true)
- `target_tail_mass::Float64`: Maximum allowed unexplored tail mass fraction (default: 1e-4)
- `stability_tolerance::Float64`: Convergence criterion for summary statistics (default: 0.005)
- `auto_adjust::Bool`: Auto-expand when issues detected vs emit warnings (default: true)
- `max_iterations::Int`: Maximum number of expansion iterations (default: 5)

# Diagnostic Tests
The method uses three tests from adaptive integration theory:
1. **Edge slope test**: Log-density at boundaries should be decreasing (negative slope)
2. **Edge mass test**: Outermost grid cells should carry negligible probability mass
3. **Exponential upper bound**: Estimate remaining tail mass from edge behavior

# Auto-Adjust Modes
- `auto_adjust=true` (default): Automatically expand region when diagnostics fail
- `auto_adjust=false` (manual): Emit warnings with suggestions, let user control

# Example
```julia
# Default: adaptive expansion enabled
method = GridBasedMarginal()

# Manual control: user sets parameters, gets warnings if insufficient
method = GridBasedMarginal(auto_adjust=false)

# Custom thresholds for stricter tail coverage
method = GridBasedMarginal(
    target_tail_mass = 1e-5,  # Stricter
    max_log_drop_cap = 15.0   # Lower cap
)
```
"""
struct GridBasedMarginal <: HyperparameterMarginalizationMethod
    log_drop_increment::Float64
    max_log_drop_cap::Float64
    allow_asymmetric::Bool
    target_tail_mass::Float64
    stability_tolerance::Float64
    auto_adjust::Bool
    max_iterations::Int
end

"""
    GridBasedMarginal(; kwargs...)

Construct GridBasedMarginal with sensible defaults from adaptive integration theory.

# Keyword Arguments
- `log_drop_increment=2.0`: Expand gradually to avoid overshooting
- `max_log_drop_cap=20.0`: Prevent infinite expansion
- `allow_asymmetric=true`: Efficient asymmetric expansion
- `target_tail_mass=1e-4`: Standard numerical integration tolerance
- `stability_tolerance=0.005`: 0.5% change in summary statistics
- `auto_adjust=true`: Automatic adaptation (disable for expert control)
- `max_iterations=5`: Usually converges in 2-3 iterations
"""
function GridBasedMarginal(;
        log_drop_increment::Float64 = 2.0,
        max_log_drop_cap::Float64 = 20.0,
        allow_asymmetric::Bool = true,
        target_tail_mass::Float64 = 0.001,
        stability_tolerance::Float64 = 0.005,
        auto_adjust::Bool = true,
        max_iterations::Int = 5
    )
    return GridBasedMarginal(
        log_drop_increment,
        max_log_drop_cap,
        allow_asymmetric,
        target_tail_mass,
        stability_tolerance,
        auto_adjust,
        max_iterations
    )
end
