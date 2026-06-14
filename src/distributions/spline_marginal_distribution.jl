using DataInterpolations
using Distributions
using Bijectors

export SplineMarginalDistribution

"""
    SplineMarginalDistribution{T, S1, S2} <: ContinuousUnivariateDistribution

A pre-computed 1D marginal distribution backed by cubic splines for O(1) queries.

All downstream operations (logpdf, cdf, quantile, mean, var, mode) are O(1) spline
lookups — no numerical integration at query time.
"""
struct SplineMarginalDistribution{T, S1, S2} <: ContinuousUnivariateDistribution
    η_grid::Vector{T}
    logpdf_spline::S1
    cdf_spline::S2

    transform::Any           # natural → working bijector
    inv_transform::Any       # working → natural bijector
    transform_increasing::Bool

    bounds::NTuple{2, T}     # natural-space bounds (lower, upper)

    mean_val::T
    var_val::T
    mode_val::T
end

# ==================== Distributions.jl Interface ====================

function Distributions.logpdf(d::SplineMarginalDistribution, x::Real)
    η = d.transform(x)
    return d.logpdf_spline(η) + Bijectors.logabsdetjac(d.transform, x)
end

function Distributions.pdf(d::SplineMarginalDistribution, x::Real)
    return exp(logpdf(d, x))
end

function Distributions.cdf(d::SplineMarginalDistribution, x::Real)
    lo, hi = d.bounds
    x <= lo && return 0.0
    x >= hi && return 1.0

    η = d.transform(x)
    cdf_working = clamp(d.cdf_spline(η), 0.0, 1.0)
    return d.transform_increasing ? cdf_working : 1.0 - cdf_working
end

function Distributions.quantile(d::SplineMarginalDistribution, q::Real)
    0 <= q <= 1 || throw(ArgumentError("quantile argument must be in [0,1]"))
    q == 0.0 && return minimum(d)
    q == 1.0 && return maximum(d)

    # Map to working-space CDF target
    q_working = d.transform_increasing ? q : 1.0 - q

    # Binary search on the η grid for CDF(η) = q_working
    η_lo, η_hi = d.η_grid[1], d.η_grid[end]

    # Bisection: find η such that cdf_spline(η) ≈ q_working
    for _ in 1:60
        η_mid = (η_lo + η_hi) / 2
        cdf_mid = clamp(d.cdf_spline(η_mid), 0.0, 1.0)
        if cdf_mid < q_working
            η_lo = η_mid
        else
            η_hi = η_mid
        end
    end

    η_result = (η_lo + η_hi) / 2
    return d.inv_transform(η_result)
end

function Distributions.mean(d::SplineMarginalDistribution)
    return d.mean_val
end

function Distributions.var(d::SplineMarginalDistribution)
    return d.var_val
end

function Distributions.mode(d::SplineMarginalDistribution)
    return d.mode_val
end

function Distributions.minimum(d::SplineMarginalDistribution)
    return d.bounds[1]
end

function Distributions.maximum(d::SplineMarginalDistribution)
    return d.bounds[2]
end

function Distributions.insupport(d::SplineMarginalDistribution, x::Real)
    lo, hi = d.bounds
    return lo <= x <= hi && isfinite(x)
end

function Base.rand(rng::AbstractRNG, d::SplineMarginalDistribution)
    return quantile(d, rand(rng))
end

Base.rand(d::SplineMarginalDistribution) = rand(Random.GLOBAL_RNG, d)

# Compact summary instead of dumping the spline internals (η_grid, splines, …).
function Base.show(io::IO, d::SplineMarginalDistribution)
    lo, hi = d.bounds
    print(
        io, "SplineMarginalDistribution(mean=", round(d.mean_val; sigdigits = 4),
        ", sd=", round(sqrt(d.var_val); sigdigits = 4),
        ", support=[", round(lo; sigdigits = 4), ", ", round(hi; sigdigits = 4), "])",
    )
    return nothing
end
