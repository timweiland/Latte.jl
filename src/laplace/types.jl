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

Adaptive marginalization: run SimplifiedLaplace for all variables, then escalate
to full LaplaceMarginal only where the skew-normal is inadequate. The escalation
gate is the magnitude of the leading term SimplifiedLaplace neglects — the
standardized 4th-order log-density coefficient `|a₄|` — so a variable is upgraded
only when the 3rd-order (skew-normal) approximation genuinely misses curvature,
not merely when its marginal is skewed.

The default `tol = 0.15` is calibrated so escalation fires roughly when the
simplified marginal's central-interval error exceeds ≈0.1: it upgrades
low-information Poisson-type latents while leaving the (already accurate)
Binomial / Bernoulli / Normal latents alone.

# Fields
- `tol::Float64`: escalation threshold on `|a₄|` (default: 0.15)

# Example
```julia
result = inla(model, y; latent_marginalization_method=AdaptiveMarginal())
result = inla(model, y; latent_marginalization_method=AdaptiveMarginal(0.05))
```
"""
struct AdaptiveMarginal <: MarginalApproximation
    tol::Float64

    function AdaptiveMarginal(tol::Float64)
        tol >= 0 || throw(ArgumentError("tol must be non-negative, got $tol"))
        isfinite(tol) || throw(ArgumentError("tol must be finite, got $tol"))
        return new(tol)
    end
end

AdaptiveMarginal() = AdaptiveMarginal(0.15)

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
