using Distributions
using Printf

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

# Custom show method for better user experience
function Base.show(io::IO, result::MarginalResult)
    n_vars = length(result.marginals)

    println(io, "MarginalResult:")
    println(io, "  Variables: ", n_vars, " (indices: ", result.indices, ")")
    println(io, "  Method: ", typeof(result.method).name.name)
    println(io, "  Computation time: ", @sprintf("%.4f", result.computation_time), " seconds")

    # Show first few marginals
    max_show = min(n_vars, 3)
    println(io, "  Marginal distributions:")
    for i in 1:max_show
        dist_name = typeof(result.marginals[i]).name.name
        println(
            io, "    Variable ", result.indices[i], ": ", dist_name,
            "(μ=", @sprintf("%.4f", mean(result.marginals[i])),
            ", σ=", @sprintf("%.4f", std(result.marginals[i])), ")"
        )
    end

    return if n_vars > 3
        print(io, "    ... and ", n_vars - 3, " more variables")
    end
end
