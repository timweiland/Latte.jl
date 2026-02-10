using Distributions
using HCubature
using FastGaussQuadrature
using DataInterpolations
using DataInterpolations: ExtrapolationType
using Random

export SplineAugmentedGaussian

"""
    SplineAugmentedGaussian{T, S} <: ContinuousUnivariateDistribution

A distribution representing a Gaussian base with a spline correction factor.
This implementation uses on-demand computation with caching for expensive operations
like moments (mean, var) and quantiles to ensure high performance in typical
use cases (e.g., repeated calls to `quantile` or `mean`).

# Fields (Internal)
- `base::Normal{T}`: The base Gaussian distribution π̃_G.
- `spline::S`: Interpolation object for the log-density correction.
- `normalization_constant::T`: The pre-computed log of the normalization constant.

# Cached Fields (Internal, Lazy-Loaded)
- `_moments`: A cached tuple of (mean, var), computed on first request via Gauss-Hermite quadrature.
- `_cdf_spline`: A cached interpolating spline for the CDF.
- `_quantile_spline`: A cached interpolating spline for the quantile function (inverse CDF).
"""
mutable struct SplineAugmentedGaussian{T, S} <: ContinuousUnivariateDistribution
    base::Normal{T}
    spline::S
    normalization_constant::T

    # --- Caches for expensive, derived quantities ---
    # Initialized to `nothing` and computed on first use.
    _moments::Union{Nothing, NTuple{2, T}}
    _cdf_spline::Union{Nothing, DataInterpolations.AbstractInterpolation}
    _quantile_spline::Union{Nothing, DataInterpolations.AbstractInterpolation}

    function SplineAugmentedGaussian(base::Normal{T}, spline::S, normalization_constant::T) where {T, S}
        # The constructor is cheap: it only stores inputs and initializes caches to `nothing`.
        return new{T, S}(base, spline, normalization_constant, nothing, nothing, nothing)
    end
end

# ==================== Core PDF/LOGPDF Methods ====================

function Distributions.logpdf(d::SplineAugmentedGaussian, x::Real)
    return logpdf(d.base, x) + d.spline(x) - d.normalization_constant
end

function Distributions.pdf(d::SplineAugmentedGaussian, x::Real)
    return exp(logpdf(d, x))
end

# ==================== Moment Calculations (mean, var) ====================

"""
    _compute_and_cache_moments!(d::SplineAugmentedGaussian; n_nodes=30)

(Private) Computes mean and variance using highly efficient Gauss-Hermite quadrature
and caches the result in `d._moments`. The integral to compute is of the form
∫ g(x) * N(x|μ,σ) dx, which is perfect for this method.
"""
function _compute_and_cache_moments!(d::SplineAugmentedGaussian{T}; n_nodes = 30) where {T}
    # If already computed, do nothing.
    isnothing(d._moments) || return

    μ_base, σ_base = params(d.base)

    # Get Gauss-Hermite nodes (t_i) and weights (w_i) for integrating ∫f(t)exp(-t²)dt
    nodes, weights = gausshermite(n_nodes)

    e_x = 0.0
    e_x2 = 0.0

    # The integral E[g(X)] = ∫g(x)N(x|μ,σ)dx becomes (1/√π)∫g(μ+t√2σ)exp(-t²)dt
    # Our g(x) is x^k * exp(spline(x) - norm_const)
    for i in 1:n_nodes
        t_i = nodes[i]
        w_i = weights[i]

        # Change of variables from GH node `t_i` back to the original scale `x`
        x_i = μ_base + t_i * sqrt(2) * σ_base

        # This is the non-Gaussian part of the PDF
        correction_factor = exp(d.spline(x_i) - d.normalization_constant)

        # Function part for E[X]
        f_val_1 = x_i * correction_factor
        # Function part for E[X^2]
        f_val_2 = x_i^2 * correction_factor

        e_x += w_i * f_val_1
        e_x2 += w_i * f_val_2
    end

    # Apply the (1/√π) scaling factor from the change of variables
    scaling_factor = 1 / sqrt(T(π))
    mean_val = e_x * scaling_factor
    var_val = (e_x2 * scaling_factor) - mean_val^2

    # Atomically store the result
    d._moments = (T(mean_val), T(var_val))
    return
end

function Distributions.mean(d::SplineAugmentedGaussian)
    _compute_and_cache_moments!(d)
    return d._moments[1]
end

function Distributions.var(d::SplineAugmentedGaussian)
    _compute_and_cache_moments!(d)
    return d._moments[2]
end

# ==================== CDF, Quantile, and Sampling ====================

"""
    _compute_and_cache_splines!(d::SplineAugmentedGaussian)

(Private) Computes the CDF on a grid via numerical integration and creates
interpolating splines for both the CDF and quantile functions. Caches them
in `d._cdf_spline` and `d._quantile_spline`.
"""
function _compute_and_cache_splines!(d::SplineAugmentedGaussian{T}) where {T}
    isnothing(d._cdf_spline) || return

    μ, σ = params(d.base)

    # 1. Create a grid for interpolation. More points near the mode.
    # A wide range (±10σ) ensures we capture the vast majority of the mass.
    lower_bound, upper_bound = μ - 10σ, μ + 10σ
    grid_center = range(μ - 4σ, μ + 4σ, length = 200)
    grid_full = range(lower_bound, upper_bound, length = 100)
    grid_points = sort(unique([collect(grid_center); collect(grid_full)]))

    # 2. Compute CDF values at grid points via numerical integration
    cdf_values = Vector{T}(undef, length(grid_points))
    cdf_values[1] = 0.0 # Approximation at the lower bound

    integrand(t_vec) = pdf(d, t_vec[1])

    for i in 2:length(grid_points)
        # Integrate from previous grid point to current one and accumulate
        integral, _ = hcubature(integrand, [grid_points[i - 1]], [grid_points[i]], rtol = 1.0e-9)
        cdf_values[i] = cdf_values[i - 1] + integral
    end
    # Clamp to handle potential minor numerical errors and ensure range is [0,1]
    cdf_values = min.(max.(cdf_values, 0.0), 1.0)

    # 3. Create interpolating splines
    # Linear interpolation is robust, fast, and guarantees monotonicity.
    # CDF: x -> cdf_value, so LinearInterpolation(cdf_values, grid_points)
    # Quantile: cdf_value -> x, so LinearInterpolation(grid_points, cdf_values)
    d._cdf_spline = LinearInterpolation(cdf_values, grid_points; extrapolation = ExtrapolationType.Constant)
    d._quantile_spline = LinearInterpolation(grid_points, cdf_values; extrapolation = ExtrapolationType.Constant)

    return
end

function Distributions.cdf(d::SplineAugmentedGaussian, x::Real)
    _compute_and_cache_splines!(d)
    return d._cdf_spline(x)
end

function Distributions.quantile(d::SplineAugmentedGaussian, q::Real)
    if !(0 ≤ q ≤ 1)
        throw(ArgumentError("quantile argument must be in [0,1]"))
    end
    _compute_and_cache_splines!(d)
    return d._quantile_spline(q)
end

# `rand` is now correct, simple, and fast via inverse transform sampling.
function Base.rand(rng::AbstractRNG, d::SplineAugmentedGaussian)
    # This leverages the fast, cached quantile function.
    # The first call to `rand` triggers the one-time spline construction cost.
    return quantile(d, rand(rng))
end

# Convenience method for default RNG
Base.rand(d::SplineAugmentedGaussian) = rand(Random.GLOBAL_RNG, d)

function Distributions.minimum(d::SplineAugmentedGaussian)
    return -Inf  # Support is all real numbers
end

function Distributions.maximum(d::SplineAugmentedGaussian)
    return Inf  # Support is all real numbers
end

function Distributions.insupport(d::SplineAugmentedGaussian, x::Real)
    return isfinite(x)  # Support is all finite real numbers
end

# ==================== Additional Distribution Methods ====================

function Distributions.skewness(d::SplineAugmentedGaussian)
    # Use cached moments and numerical integration for third central moment
    μ_d = mean(d)
    σ_d = std(d)
    μ, σ = params(d.base)

    integrand(x_vec) = ((x_vec[1] - μ_d) / σ_d)^3 * pdf(d, x_vec[1])
    result, _ = hcubature(integrand, [μ - 6 * σ], [μ + 6 * σ], rtol = 1.0e-6)
    return result
end

function Distributions.kurtosis(d::SplineAugmentedGaussian)
    # Use cached moments and numerical integration for fourth central moment minus 3 (excess kurtosis)
    μ_d = mean(d)
    σ_d = std(d)
    μ, σ = params(d.base)

    integrand(x_vec) = ((x_vec[1] - μ_d) / σ_d)^4 * pdf(d, x_vec[1])
    result, _ = hcubature(integrand, [μ - 6 * σ], [μ + 6 * σ], rtol = 1.0e-6)
    return result - 3.0  # Excess kurtosis
end

function Distributions.entropy(d::SplineAugmentedGaussian)
    # Numerical integration: -∫ pdf(x) * log(pdf(x)) dx
    μ, σ = params(d.base)

    integrand(x_vec) = begin
        p = pdf(d, x_vec[1])
        p > 0 ? -p * log(p) : 0.0  # Handle p=0 case
    end

    result, _ = hcubature(integrand, [μ - 6 * σ], [μ + 6 * σ], rtol = 1.0e-6)
    return result
end

# ==================== Parameter access ====================

"""
    params(d::SplineAugmentedGaussian)

Return the parameters of the distribution as a tuple (base, spline, normalization_constant).
"""
Distributions.params(d::SplineAugmentedGaussian) = (d.base, d.spline, d.normalization_constant)

"""
    partype(::Type{<:SplineAugmentedGaussian{T}}) where T

Return the parameter type.
"""
Distributions.partype(::Type{<:SplineAugmentedGaussian{T}}) where {T} = T
