export PosteriorAccumulator, accumulate!, finalize!, get_integration_weights

"""
    PosteriorAccumulator

Abstract type for accumulators that process hyperparameter posterior exploration.

Accumulators follow a two-phase pattern:
1. **Accumulate**: Called at each integration point (unweighted)
2. **Finalize**: Apply weights and compute final results

# Required Methods
- `accumulate!(acc; kwargs...)`: Process one grid point
- `finalize!(acc, exploration)`: Compute final weighted results

# Usage
Accumulators are called during `explore_hyperparameter_posterior`:
- After a point is determined to be an integration point
- Before weights are normalized
- With all available information from `evaluate_at_grid_point`

# Example
```julia
mutable struct MyAccumulator <: PosteriorAccumulator
    values::Vector{Float64}
    result::Float64
    MyAccumulator() = new(Float64[], 0.0)
end

function accumulate!(acc::MyAccumulator; total_loglikelihood, kwargs...)
    push!(acc.values, total_loglikelihood)
end

function finalize!(acc::MyAccumulator, exploration)
    weights = get_integration_weights(exploration)
    acc.result = sum(weights .* acc.values)
end
```
"""
abstract type PosteriorAccumulator end

"""
    accumulate!(acc::PosteriorAccumulator; kwargs...)

Process information from one integration point (unweighted).

# Available kwargs (from evaluate_at_grid_point):
- `θ::WorkingHyperparameters`: Hyperparameters
- `log_density::Float64`: Log posterior π(θ|y)
- `obs_loglikelihoods::Vector{Float64}`: Per-observation log p(yᵢ|xᵢ,θ)
- `total_loglikelihood::Float64`: Sum of obs_loglikelihoods
- `ga::GMRF`: Gaussian approximation
- `x_star::Vector{Float64}`: Latent field mode
- `marginal_result`: Latent marginals at this point
- `obs_lik::ObservationLikelihood`: Materialized observation likelihood for current θ
- `y::Vector`: Observations (added by caller)
- `is_mode::Bool`: True if this is hyperparameter mode (added by caller)

Each accumulator uses only what it needs and ignores the rest via `kwargs...`.
"""
function accumulate! end

"""
    finalize!(acc::PosteriorAccumulator, exploration::AbstractHyperparameterExploration)

Finalize accumulator after all points processed. Apply integration weights.

# Arguments
- `acc`: The accumulator to finalize
- `exploration`: HyperparameterExploration containing weights and other info

Extract weights via helper: `weights = get_integration_weights(exploration)`
"""
function finalize! end

"""
    get_integration_weights(exploration::AbstractHyperparameterExploration) -> Vector{Float64}

Extract normalized integration weights from exploration result, ordered to match
the accumulator call order (i.e., the order in which `accumulate!` was called).

Returns vector of weights summing to 1.0.

The weights are computed from the normalized log densities stored in the grid points,
then reordered via `exploration.accumulator_reorder` so that `weights[k]` corresponds
to the k-th `accumulate!` call.
"""
function get_integration_weights(exploration::AbstractHyperparameterExploration)
    integration_points = exploration.grid_points[exploration.integration_indices]
    log_weights = [p.log_density for p in integration_points]
    weights = exp.(log_weights)
    weights ./= sum(weights)  # Normalize to sum to 1 (for numerical safety)
    # Reorder from grid order → accumulator call order
    return weights[exploration.accumulator_reorder]
end
