"""
Tail coverage diagnostics for grid-based hyperparameter marginalization.

Implements diagnostic tests from adaptive integration theory:
1. Edge slope test: d(log π)/dθ should be negative at boundaries
2. Edge mass test: Outermost cells should carry negligible mass
3. Exponential upper bound: Estimate remaining tail mass from edge behavior
"""

using Printf
using Statistics

export MarginalDiagnostics, diagnose_marginal_coverage

"""
    MarginalDiagnostics

Diagnostic information for one hyperparameter dimension's marginal.

# Fields
- `dimension::Int`: Which hyperparameter (1 to n_dim)
- `left_tail_ok::Bool`: Whether left (negative direction) tail is adequately covered
- `right_tail_ok::Bool`: Whether right (positive direction) tail is adequately covered
- `left_tail_mass::Float64`: Estimated unexplored mass in left tail
- `right_tail_mass::Float64`: Estimated unexplored mass in right tail
- `left_edge_slope::Float64`: d(log π)/dθ at left boundary (should be < 0)
- `right_edge_slope::Float64`: d(log π)/dθ at right boundary (should be < 0)
- `heavy_tail_detected::Bool`: Whether distribution appears heavy-tailed
- `suggested_left_extension::Float64`: Additional log-drop needed (0 if ok)
- `suggested_right_extension::Float64`: Additional log-drop needed (0 if ok)
"""
struct MarginalDiagnostics
    dimension::Int
    left_tail_ok::Bool
    right_tail_ok::Bool
    left_tail_mass::Float64
    right_tail_mass::Float64
    left_edge_slope::Float64
    right_edge_slope::Float64
    heavy_tail_detected::Bool
    suggested_left_extension::Float64
    suggested_right_extension::Float64
end

"""
    diagnose_marginal_coverage(posterior_approx, log_drop_limits, n_dim, target_tail_mass)

Diagnose tail coverage for all hyperparameter marginals.

# Algorithm
For each dimension:
1. Sample the marginal from the interpolant
2. Compute edge slopes using finite differences
3. Estimate tail mass using exponential extrapolation
4. Detect heavy tails by comparing to Gaussian
5. Determine if tails are adequately covered
6. Compute suggested extensions if needed

# Arguments
- `posterior_approx`: Interpolated posterior (from build_posterior_interpolant)
- `log_drop_limits::AsymmetricLogDropLimits`: Current exploration limits
- `n_dim::Int`: Number of hyperparameters
- `target_tail_mass::Float64`: Maximum allowed unexplored tail mass

# Returns
Vector of `MarginalDiagnostics`, one per hyperparameter dimension.
"""
function diagnose_marginal_coverage(
        posterior_approx,
        log_drop_limits::AsymmetricLogDropLimits,
        n_dim::Int,
        target_tail_mass::Float64
    )
    diagnostics = MarginalDiagnostics[]

    for dim in 1:n_dim
        # Get current limits for this dimension
        left_limit = log_drop_limits.limits[dim, 1]
        right_limit = log_drop_limits.limits[dim, 2]

        # Sample marginal distribution from interpolant
        # (This is a simplified approach - in practice we'd integrate properly)
        θ_grid, log_π_grid = sample_marginal_from_interpolant(posterior_approx, dim)

        # Test 1: Edge slope test
        left_slope = compute_left_edge_slope(θ_grid, log_π_grid)
        right_slope = compute_right_edge_slope(θ_grid, log_π_grid)

        # Test 2 & 3: Edge mass and exponential upper bound
        left_mass = estimate_left_tail_mass(θ_grid, log_π_grid, left_slope)
        right_mass = estimate_right_tail_mass(θ_grid, log_π_grid, right_slope)

        # Heavy-tail detection
        heavy_tail = detect_heavy_tail(θ_grid, log_π_grid)

        # Determine if tails are ok
        # Tails are ok if: (1) slope implies decrease in log density AND (2) estimated mass is small
        left_ok = (left_slope > 1.0e-6) && (left_mass < target_tail_mass)
        right_ok = (right_slope < -1.0e-6) && (right_mass < target_tail_mass)

        # Compute suggested extensions
        # If tail mass is too high, suggest additional log-drop based on exponential model
        left_ext = left_ok ? 0.0 : estimate_needed_extension(left_mass, target_tail_mass)
        right_ext = right_ok ? 0.0 : estimate_needed_extension(right_mass, target_tail_mass)

        # If slope is positive or near-zero, this is critical - suggest larger extension
        if left_slope >= -1.0e-6 && !left_ok
            left_ext = max(left_ext, 3.0)  # At least 3 nats more
        end
        if right_slope >= -1.0e-6 && !right_ok
            right_ext = max(right_ext, 3.0)
        end

        push!(
            diagnostics, MarginalDiagnostics(
                dim, left_ok, right_ok,
                left_mass, right_mass,
                left_slope, right_slope,
                heavy_tail,
                left_ext, right_ext
            )
        )
    end

    return diagnostics
end

"""
    sample_marginal_from_interpolant(posterior_approx, dim)

Sample a 1D marginal from the posterior interpolant.

Returns (θ_grid, log_π_grid) for the specified dimension.
For now, this is a placeholder that extracts values from the existing grid.
A more sophisticated implementation would numerically integrate over other dimensions.
"""
function sample_marginal_from_interpolant(posterior_approx, dim)
    # TODO: This should be replaced with proper numerical integration
    # For now, we use a simplified approach based on the HyperparameterMarginalDistribution
    # which already does the integration internally

    # Create a grid of θ values for this dimension
    n_points = 50
    dist = HyperparameterMarginalDistribution(posterior_approx, dim)

    # Sample at quantiles to get good coverage
    quantiles = range(0.001, 0.999, length = n_points)
    θ_grid = quantile.(Ref(dist), quantiles)

    # Evaluate log-density at these points
    log_π_grid = logpdf.(Ref(dist), θ_grid)

    return θ_grid, log_π_grid
end

"""
    compute_left_edge_slope(θ_grid, log_π_grid)

Compute d(log π)/dθ at the left boundary using finite differences.

Negative slope indicates density is decreasing (good).
Positive or zero slope indicates boundary is cutting off significant mass (bad).
"""
function compute_left_edge_slope(θ_grid, log_π_grid)
    # Use forward difference at left boundary
    if length(θ_grid) < 2
        return 0.0
    end
    Δθ = θ_grid[2] - θ_grid[1]
    Δlog_π = log_π_grid[2] - log_π_grid[1]
    return Δlog_π / Δθ
end

"""
    compute_right_edge_slope(θ_grid, log_π_grid)

Compute d(log π)/dθ at the right boundary using finite differences.

Negative slope indicates density is decreasing (good).
"""
function compute_right_edge_slope(θ_grid, log_π_grid)
    # Use backward difference at right boundary
    n = length(θ_grid)
    if n < 2
        return 0.0
    end
    Δθ = θ_grid[n] - θ_grid[n - 1]
    Δlog_π = log_π_grid[n] - log_π_grid[n - 1]
    return Δlog_π / Δθ
end

"""
    estimate_left_tail_mass(θ_grid, log_π_grid, left_slope)

Estimate the unexplored probability mass in the left tail using exponential extrapolation.

# Theory
If log π(θ) ≈ log π(θ_boundary) + slope*(θ - θ_boundary) for θ < θ_boundary, then:
∫_{-∞}^{θ_boundary} exp(log π(θ)) dθ ≈ exp(log π(θ_boundary)) / |slope|

This assumes exponential decay, which is conservative for faster-than-exponential tails
and may underestimate for slower-than-exponential (heavy) tails.

Returns the estimated mass as a fraction of the total mass (0 to 1).
"""
function estimate_left_tail_mass(θ_grid, log_π_grid, left_slope)
    if isempty(θ_grid)
        return 1.0  # Conservative: assume all mass is unexplored
    end

    # Edge log-density
    log_π_edge = log_π_grid[1]

    # Exponential extrapolation: ∫_{-∞}^{edge} exp(L + s*(θ - θ_edge)) dθ = exp(L) / |s|
    # where L = log_π_edge and s = left_slope (negative)
    tail_integral = exp(log_π_edge) / abs(left_slope)

    # Estimate total integral by trapezoidal rule on the grid
    total_integral = estimate_total_integral(θ_grid, log_π_grid)

    # Fraction of unexplored mass
    if total_integral <= 0
        return 1.0  # Defensive
    end

    return min(tail_integral / total_integral, 1.0)
end

"""
    estimate_right_tail_mass(θ_grid, log_π_grid, right_slope)

Estimate the unexplored probability mass in the right tail.

Similar to left tail but for θ > θ_boundary.
"""
function estimate_right_tail_mass(θ_grid, log_π_grid, right_slope)
    if isempty(θ_grid)
        return 1.0
    end

    # Edge log-density
    log_π_edge = log_π_grid[end]

    # For right tail: ∫_{edge}^{∞} exp(L + s*(θ - θ_edge)) dθ
    # With s < 0, this becomes exp(L) / |s|
    tail_integral = exp(log_π_edge) / abs(right_slope)

    # Estimate total integral
    total_integral = estimate_total_integral(θ_grid, log_π_grid)

    if total_integral <= 0
        return 1.0
    end

    return min(tail_integral / total_integral, 1.0)
end

"""
    estimate_total_integral(θ_grid, log_π_grid)

Estimate ∫ exp(log π(θ)) dθ using trapezoidal rule.
"""
function estimate_total_integral(θ_grid, log_π_grid)
    if length(θ_grid) < 2
        return exp(log_π_grid[1])  # Single point fallback
    end

    # Trapezoidal rule on exp(log_π)
    total = 0.0
    for i in 1:(length(θ_grid) - 1)
        Δθ = θ_grid[i + 1] - θ_grid[i]
        # Average of exp values (trapezoidal)
        avg = 0.5 * (exp(log_π_grid[i]) + exp(log_π_grid[i + 1]))
        total += Δθ * avg
    end

    return total
end

"""
    detect_heavy_tail(θ_grid, log_π_grid)

Detect if the marginal appears to be heavy-tailed by comparing to Gaussian behavior.

# Method
Compare the observed log-drop at ±2σ from the mode with the theoretical Gaussian drop.
Gaussian: log π(μ ± kσ) = log π(μ) - 0.5*k²
For k=2: drop should be -2
For k=3: drop should be -4.5

If observed drop is significantly smaller, flag as heavy-tailed.
"""
function detect_heavy_tail(θ_grid, log_π_grid)
    if length(θ_grid) < 5
        return false  # Not enough data
    end

    # Find mode
    max_log_π = maximum(log_π_grid)
    mode_idx = argmax(log_π_grid)

    # Estimate σ from curvature at mode (rough approximation)
    # In practice, this could use the Hessian from the reparameterization
    # For now, just check if the tails drop much slower than Gaussian

    # Check points at roughly ±2σ (25th and 75th percentiles as rough proxy)
    n = length(θ_grid)
    left_check_idx = max(1, div(n, 4))
    right_check_idx = min(n, 3 * div(n, 4))

    left_drop = max_log_π - log_π_grid[left_check_idx]
    right_drop = max_log_π - log_π_grid[right_check_idx]

    # For Gaussian at 2σ, expect drop of ~2
    # If drop is < 1, this is suspiciously flat (heavy-tailed)
    gaussian_expected_drop = 2.0
    heavy_tail_threshold = 0.5 * gaussian_expected_drop

    return (left_drop < heavy_tail_threshold) || (right_drop < heavy_tail_threshold)
end

"""
    estimate_needed_extension(tail_mass, target_mass)

Estimate how much additional log-drop is needed to reduce tail mass to target.

Uses exponential decay model: mass ~ exp(-log_drop * rate)
Solving for additional log_drop needed.
"""
function estimate_needed_extension(tail_mass::Float64, target_mass::Float64)
    if tail_mass <= target_mass
        return 0.0
    end

    # Use logarithmic relationship: if mass ~ exp(-α * log_drop), then
    # log(tail_mass) = log(mass_0) - α * log_drop
    # To reduce by factor f: Δlog_drop = log(f) / α
    # We assume α ≈ 1 (conservative)

    ratio = tail_mass / target_mass
    return log(ratio)  # Additional log-drop needed
end
