"""
Utility functions for grid-based hyperparameter marginalization.
"""

using Statistics

export compute_summary_statistics, check_stability

"""
    SummaryStatistics

Summary statistics for hyperparameter marginals used in convergence checking.

Stores mean, variance, and quantiles for each hyperparameter dimension.
"""
struct SummaryStatistics
    means::Vector{Float64}
    variances::Vector{Float64}
    quantiles_025::Vector{Float64}
    quantiles_975::Vector{Float64}
end

"""
    compute_summary_statistics(posterior_approx, n_dim)

Compute summary statistics for all hyperparameter marginals.

# Arguments
- `posterior_approx`: Interpolated posterior
- `n_dim::Int`: Number of hyperparameters

# Returns
`SummaryStatistics` object containing means, variances, and quantiles.
"""
function compute_summary_statistics(posterior_approx, n_dim::Int)
    means = zeros(n_dim)
    variances = zeros(n_dim)
    quantiles_025 = zeros(n_dim)
    quantiles_975 = zeros(n_dim)

    for i in 1:n_dim
        dist = HyperparameterMarginalDistribution(posterior_approx, i)

        means[i] = mean(dist)
        variances[i] = var(dist)
        quantiles_025[i] = quantile(dist, 0.025)
        quantiles_975[i] = quantile(dist, 0.975)
    end

    return SummaryStatistics(means, variances, quantiles_025, quantiles_975)
end

"""
    check_stability(previous_stats, current_stats, tolerance)

Check if summary statistics have stabilized between iterations.

Compares relative changes in means, variances, and quantiles across all dimensions.
Returns true if all changes are below the tolerance threshold.

# Arguments
- `previous_stats::SummaryStatistics`: Stats from previous iteration
- `current_stats::SummaryStatistics`: Stats from current iteration
- `tolerance::Float64`: Maximum allowed relative change (e.g., 0.005 for 0.5%)

# Returns
`true` if all statistics have stabilized, `false` otherwise.
"""
function check_stability(
        previous_stats::SummaryStatistics,
        current_stats::SummaryStatistics,
        tolerance::Float64
    )
    n_dim = length(previous_stats.means)

    for i in 1:n_dim
        # Check mean stability
        if !is_stable(previous_stats.means[i], current_stats.means[i], tolerance)
            return false
        end

        # Check variance stability
        if !is_stable(previous_stats.variances[i], current_stats.variances[i], tolerance)
            return false
        end

        # Check quantile stability
        if !is_stable(previous_stats.quantiles_025[i], current_stats.quantiles_025[i], tolerance)
            return false
        end
        if !is_stable(previous_stats.quantiles_975[i], current_stats.quantiles_975[i], tolerance)
            return false
        end
    end

    return true
end

"""
    is_stable(previous_value, current_value, tolerance)

Check if a single statistic is stable (relative change < tolerance).

Handles edge cases where values are near zero.
"""
function is_stable(previous_value::Float64, current_value::Float64, tolerance::Float64)
    # If both values are very small, consider stable
    if abs(previous_value) < 1.0e-10 && abs(current_value) < 1.0e-10
        return true
    end

    # Compute relative change
    # Use max of abs values to handle sign changes
    denominator = max(abs(previous_value), abs(current_value))

    if denominator < 1.0e-10
        return true  # Both effectively zero
    end

    relative_change = abs(current_value - previous_value) / denominator

    return relative_change < tolerance
end
