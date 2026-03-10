using Distributions
using Printf

export MarginalApproximation, MarginalResult, AdaptiveMarginal

"""
    MarginalApproximation

Abstract base type for different marginalization approximation methods in INLA.
"""
abstract type MarginalApproximation end

"""
    AdaptiveMarginal <: MarginalApproximation

Adaptive marginalization strategy following Rue et al. (2009) Section 4.2.

Starts with SimplifiedLaplace for all variables, computes SKLD against the Gaussian
baseline, and escalates to full LaplaceMarginal for variables where SKLD exceeds the
threshold.

# Fields
- `kld_threshold::Float64`: SKLD threshold for escalation (default: 0.1)

# Example
```julia
result = inla(model, y; latent_marginalization_method=AdaptiveMarginal())
result = inla(model, y; latent_marginalization_method=AdaptiveMarginal(0.05))
```
"""
struct AdaptiveMarginal <: MarginalApproximation
    kld_threshold::Float64

    function AdaptiveMarginal(kld_threshold::Float64)
        kld_threshold >= 0 || throw(ArgumentError("kld_threshold must be non-negative, got $kld_threshold"))
        isfinite(kld_threshold) || throw(ArgumentError("kld_threshold must be finite, got $kld_threshold"))
        return new(kld_threshold)
    end
end

AdaptiveMarginal() = AdaptiveMarginal(0.1)

"""
    MarginalResult

Container for marginalization results.

# Fields
- `indices::Vector{Int}`: Indices of marginalized variables
- `marginals::Vector{<:ContinuousUnivariateDistribution}`: Marginal distributions
- `method::MarginalApproximation`: Approximation method used
- `computation_time::Float64`: Computation time in seconds
- `kld_values::Vector{Float64}`: Symmetric KLD between Gaussian and corrected marginal per variable
"""
struct MarginalResult
    indices::Vector{Int}
    marginals::Vector{<:ContinuousUnivariateDistribution}
    method::MarginalApproximation
    computation_time::Float64  # in seconds
    kld_values::Vector{Float64}
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
