using Distributions

export MarginalApproximation, MarginalResult

"""
    MarginalApproximation

Abstract base type for different marginalization approximation methods in INLA.
"""
abstract type MarginalApproximation end

"""
    MarginalResult

Container for marginalization results.

# Fields
- `indices::Vector{Int}`: Indices of marginalized variables
- `marginals::Vector{<:ContinuousUnivariateDistribution}`: Marginal distributions
- `method::MarginalApproximation`: Approximation method used
- `computation_time::Float64`: Computation time in seconds
"""
struct MarginalResult
    indices::Vector{Int}
    marginals::Vector{<:ContinuousUnivariateDistribution}
    method::MarginalApproximation
    computation_time::Float64  # in seconds
end
