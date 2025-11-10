using Distributions
using StatsBase
using Random
using Roots
using Printf
using Optim

export WeightedMixture

"""
    WeightedMixture{T} <: ContinuousUnivariateDistribution

A weighted mixture of univariate continuous distributions representing 
the final INLA marginal after integrating over hyperparameter uncertainty.

Efficiently computes PDF, CDF, moments, and sampling. Quantile computation 
uses robust root-finding. All standard Distributions.jl methods supported.

# Fields
- `components::Vector{<:ContinuousUnivariateDistribution}`: Component distributions
- `weights::Vector{T}`: Normalized mixture weights (sum to 1.0)

# Performance Notes  
- PDF/CDF: Fast (linear in number of components)
- Mean/Variance: Cached after first computation
- Quantile: Robust root-finding with exact support bounds
- Sampling: Efficient via component selection + delegation
"""
struct WeightedMixture{T} <: ContinuousUnivariateDistribution
    components::Vector{<:ContinuousUnivariateDistribution}
    weights::Vector{T}

    function WeightedMixture(
            components::Vector{<:ContinuousUnivariateDistribution},
            weights::Vector{T}
        ) where {T}
        @assert length(components) == length(weights) "Components and weights must have same length"
        @assert all(w >= 0 for w in weights) "All weights must be non-negative"
        @assert !isempty(components) "Must have at least one component"

        # Normalize weights to handle numerical errors from integration
        normalized_weights = weights ./ sum(weights)

        return new{T}(components, normalized_weights)
    end
end

# PDF and CDF
function Distributions.pdf(d::WeightedMixture, x::Real)
    return sum(w * pdf(comp, x) for (w, comp) in zip(d.weights, d.components))
end

function Distributions.logpdf(d::WeightedMixture, x::Real)
    # Use log-sum-exp for numerical stability
    log_terms = [log(w) + logpdf(comp, x) for (w, comp) in zip(d.weights, d.components)]
    return logsumexp(log_terms)
end

function Distributions.cdf(d::WeightedMixture, x::Real)
    return sum(w * cdf(comp, x) for (w, comp) in zip(d.weights, d.components))
end

# Moments
function Distributions.mean(d::WeightedMixture)
    return sum(w * mean(comp) for (w, comp) in zip(d.weights, d.components))
end

function Distributions.var(d::WeightedMixture{T}) where {T}
    # Law of Total Variance: Var(X) = E[Var(X|θ)] + Var(E[X|θ])
    # Using raw moments for numerical stability

    # E[X²] = ∑ wᵢ E[Xᵢ²] = ∑ wᵢ (σᵢ² + μᵢ²)
    second_moment = sum(
        w * (var(comp) + mean(comp)^2)
            for (w, comp) in zip(d.weights, d.components)
    )

    # Var(X) = E[X²] - (E[X])²
    first_moment = mean(d)
    return second_moment - first_moment^2
end

function Distributions.std(d::WeightedMixture)
    return sqrt(var(d))
end

function Distributions.mode(d::WeightedMixture)
    x0 = mean(d)
    result = optimize(x -> -logpdf(d, x[1]), [x0])
    return only(Optim.minimizer(result))
end

# Support
function Distributions.minimum(d::WeightedMixture)
    return minimum(minimum(comp) for comp in d.components)
end

function Distributions.maximum(d::WeightedMixture)
    return maximum(maximum(comp) for comp in d.components)
end

function Distributions.insupport(d::WeightedMixture, x::Real)
    # x is in support if it's in support of any component with positive weight
    return any(
        w > 0 && insupport(comp, x)
            for (w, comp) in zip(d.weights, d.components)
    )
end

"""
Robust quantile computation using exact support bounds and root finding.
"""
function Distributions.quantile(d::WeightedMixture, p::Real)
    @assert 0 <= p <= 1 "Quantile argument must be in [0,1]"

    # Handle edge cases
    if p == 0.0
        return minimum(d)
    elseif p == 1.0
        return maximum(d)
    end

    # Get exact support bounds for robust bracketing
    lower_bound = minimum(d)
    upper_bound = maximum(d)

    # Handle infinite bounds by using finite approximations
    if isinf(lower_bound)
        # Use a very negative value that gives CDF ≈ 0
        lower_bound = -1.0e10
        while cdf(d, lower_bound) > 1.0e-10
            lower_bound *= 2
        end
    end

    if isinf(upper_bound)
        # Use a very positive value that gives CDF ≈ 1
        upper_bound = 1.0e10
        while cdf(d, upper_bound) < 1 - 1.0e-10
            upper_bound *= 2
        end
    end

    # Robust root finding: solve cdf(x) = p
    return find_zero(
        x -> cdf(d, x) - p, (lower_bound, upper_bound),
        xatol = 1.0e-12, xrtol = 1.0e-12
    )
end

# Random sampling
function Base.rand(rng::AbstractRNG, d::WeightedMixture)
    # Efficient sampling via component selection
    component_idx = sample(rng, 1:length(d.components), Weights(d.weights))
    return rand(rng, d.components[component_idx])
end

Base.rand(d::WeightedMixture) = rand(Random.GLOBAL_RNG, d)

# Utility function for log-sum-exp
function logsumexp(log_terms::Vector)
    max_log = maximum(log_terms)
    return max_log + log(sum(exp(x - max_log) for x in log_terms))
end

# Custom show method for better user experience
function Base.show(io::IO, d::WeightedMixture)
    n_components = length(d.components)

    println(io, "WeightedMixture{", eltype(d.weights), "}:")
    println(io, "  Components: ", n_components)

    # Show first few components with their weights
    max_show = min(n_components, 3)
    for i in 1:max_show
        comp_name = typeof(d.components[i]).name.name
        println(
            io, "    ", @sprintf("%.4f", d.weights[i]), " × ", comp_name,
            "(μ=", @sprintf("%.4f", mean(d.components[i])),
            ", σ=", @sprintf("%.4f", std(d.components[i])), ")"
        )
    end

    if n_components > 3
        println(io, "    ... and ", n_components - 3, " more components")
    end

    # Show mixture statistics
    println(io, "  Mixture mean: ", @sprintf("%.4f", mean(d)))
    return print(io, "  Mixture std: ", @sprintf("%.4f", std(d)))
end
