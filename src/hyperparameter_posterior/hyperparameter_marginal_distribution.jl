using Distributions
using HCubature
using Random
using Roots
using Printf
using Optim

export HyperparameterMarginalDistribution

"""
    HyperparameterMarginalDistribution{T} <: ContinuousUnivariateDistribution

A distribution representing the marginal distribution of a single hyperparameter
from a `HyperparameterPosteriorApproximation`. This provides a user-friendly
Distributions.jl interface for hyperparameter marginals.

All operations are in **natural space** (constrained parameter space), as this is
the space where users interact with hyperparameters.

This implementation uses lazy computation with caching for expensive operations
like moments (mean, var) to ensure high performance. CDF and quantile are computed
directly when needed without pre-computing splines, as they are typically used
sparingly (e.g., for confidence intervals).

# Type Parameters
- `T`: Numeric type (typically Float64)

# Fields (Internal)
- `approx::HyperparameterPosteriorApproximation`: The posterior approximation object (operates in natural space)
- `marginal_dim::Int`: The dimension (1-indexed) for which to compute the marginal
- `rtol::T`: Relative tolerance for numerical integration
- `atol::T`: Absolute tolerance for numerical integration
- `bounds::NTuple{2, T}`: Natural space bounds for this dimension
- `log_normalization_constant::T`: Normalization constant to ensure PDF integrates to 1

# Cached Fields (Internal, Lazy-Loaded)
- `_moments::Union{Nothing, NTuple{2, T}}`: Cached (mean, var), computed on first request
- `_mode::Union{Nothing, T}`: Cached mode, computed on first request

# Constructor
```julia
HyperparameterMarginalDistribution(
    approx::HyperparameterPosteriorApproximation,
    marginal_dim::Int;
    rtol::Real = 1.0e-3,
    atol::Real = 1.0e-6
)
```

# Example
```julia
# Create marginal distribution for first hyperparameter
marginal_dist = HyperparameterMarginalDistribution(approx, 1)

# Use standard Distribution interface - all in natural space
logpdf(marginal_dist, 2.5)      # Evaluate at τ = 2.5 (natural space)
mean(marginal_dist)             # Mean in natural space
quantile(marginal_dist, 0.95)   # 95th quantile in natural space
rand(marginal_dist)             # Sample in natural space
mode(marginal_dist)             # Mode in natural space
```
"""
mutable struct HyperparameterMarginalDistribution{T} <: ContinuousUnivariateDistribution
    approx::HyperparameterPosteriorApproximation
    marginal_dim::Int
    rtol::T
    atol::T

    # Natural space bounds for this dimension
    bounds::NTuple{2, T}

    # Normalization constant to ensure PDF integrates to 1
    log_normalization_constant::T

    # --- Caches for expensive, derived quantities ---
    # Initialized to `nothing` and computed on first use
    _moments::Union{Nothing, NTuple{2, T}}
    _mode::Union{Nothing, T}

    function HyperparameterMarginalDistribution(
            approx::HyperparameterPosteriorApproximation,
            marginal_dim::Int;
            rtol::Real = 1.0e-3,
            atol::Real = 1.0e-6
        )
        T = Float64

        # Input validation - get dimensions from the first grid point
        n_dims = length(approx.exploration.grid_points[1].θ)
        if !(1 <= marginal_dim <= n_dims)
            throw(ArgumentError("marginal_dim must be between 1 and $n_dims"))
        end

        # Extract natural space bounds for this dimension
        integration_bounds = approx.exploration.integration_bounds
        lower_bound = T(integration_bounds[marginal_dim, 1])
        upper_bound = T(integration_bounds[marginal_dim, 2])
        bounds = (lower_bound, upper_bound)

        # Compute normalization constant via integration over full hyperparameter space
        lower_bounds_full = integration_bounds[:, 1]
        upper_bounds_full = integration_bounds[:, 2]

        # Integrand that just computes the density (for normalization)
        function normalization_integrand(θ_vec)
            θ = θ_vec[1:end]
            try
                log_density = approx(θ)
                return exp(log_density)
            catch
                return 0.0
            end
        end

        # Integrate over full hyperparameter space
        normalization_integral, _ = hcubature(
            normalization_integrand, lower_bounds_full, upper_bounds_full;
            rtol = T(rtol), atol = T(atol)
        )

        log_normalization_constant = T(log(normalization_integral))

        return new{T}(
            approx, marginal_dim, T(rtol), T(atol),
            bounds, log_normalization_constant, nothing, nothing
        )
    end
end

# ==================== Core PDF/LOGPDF Methods ====================

"""
    Distributions.logpdf(d::HyperparameterMarginalDistribution, x::Real)

Evaluate log-density at x (in natural space).

This integrates out all other hyperparameters:
log π(θⱼ = x | y) = log ∫ π(θ | y) dθ₋ⱼ - log Z

where Z is the normalization constant.
"""
function Distributions.logpdf(d::HyperparameterMarginalDistribution, x::Real)
    unnormalized_logpdf = hyperparameter_marginal_logpdf(
        d.approx, d.marginal_dim, x;
        rtol = d.rtol, atol = d.atol
    )
    return unnormalized_logpdf - d.log_normalization_constant
end

function Distributions.pdf(d::HyperparameterMarginalDistribution, x::Real)
    return exp(logpdf(d, x))
end

# ==================== Moment Calculations (mean, var) ====================

"""
    _integrate_over_joint(d::HyperparameterMarginalDistribution, fun, output_size)

Helper function to integrate a function over the full joint hyperparameter space.
Used for computing moments and other expectations.
"""
function _integrate_over_joint(d::HyperparameterMarginalDistribution, fun, output_size)
    # Get integration bounds from exploration
    integration_bounds = d.approx.exploration.integration_bounds
    lower_bounds = integration_bounds[:, 1]
    upper_bounds = integration_bounds[:, 2]

    function integrand(θ_vec)
        θ = θ_vec[1:end]
        # Evaluate the interpolated posterior
        try
            log_density = d.approx(θ)
            normalized_density = exp(log_density - d.log_normalization_constant)

            # Extract the marginal dimension value (already in natural space)
            θⱼ = θ[d.marginal_dim]
            return fun(θⱼ, log_density) * normalized_density
        catch
            # Return zeros if interpolation fails (outside domain)
            return zeros(output_size)
        end
    end

    result, _ = hcubature(
        integrand, lower_bounds, upper_bounds;
        rtol = d.rtol, atol = d.atol
    )
    return result
end

"""
    _compute_and_cache_moments!(d::HyperparameterMarginalDistribution)

(Private) Computes mean and variance using direct integration over the full
hyperparameter space (avoiding double integration) and caches the result.

For efficiency, this integrates E[θⱼ] and E[θⱼ²] simultaneously:
- E[θⱼ] = ∫...∫ θⱼ * π(θ|y) dθ₁...dθₙ
- E[θⱼ²] = ∫...∫ θⱼ² * π(θ|y) dθ₁...dθₙ
- Var(θⱼ) = E[θⱼ²] - (E[θⱼ])²

Uses the interpolated posterior directly to avoid nested integration.
"""
function _compute_and_cache_moments!(d::HyperparameterMarginalDistribution{T}) where {T}
    # If already computed, do nothing
    isnothing(d._moments) || return

    first_moment, second_moment = _integrate_over_joint(d, (θⱼ, _) -> [θⱼ, θⱼ^2], 2)

    # Compute variance: Var(θⱼ) = E[θⱼ²] - (E[θⱼ])²
    mean_val = T(first_moment)
    var_val = T(second_moment - first_moment^2)

    # Ensure variance is non-negative (handle numerical errors)
    var_val = max(var_val, zero(T))

    # Atomically store the result
    d._moments = (mean_val, var_val)
    return
end

function Distributions.mean(d::HyperparameterMarginalDistribution)
    _compute_and_cache_moments!(d)
    return d._moments[1]
end

function Distributions.var(d::HyperparameterMarginalDistribution)
    _compute_and_cache_moments!(d)
    return d._moments[2]
end

# ==================== Mode ====================

"""
    _compute_and_cache_mode!(d::HyperparameterMarginalDistribution)

(Private) Finds the mode by maximizing the marginal log-density and caches it.
"""
function _compute_and_cache_mode!(d::HyperparameterMarginalDistribution{T}) where {T}
    # If already computed, do nothing
    isnothing(d._mode) || return

    a, b = d.bounds

    # Maximize logpdf over the support
    result = Optim.optimize(
        x -> -logpdf(d, x),  # Minimize negative logpdf
        a, b,
        Optim.Brent()  # Univariate optimization
    )

    mode_val = T(Optim.minimizer(result))
    d._mode = mode_val
    return
end

function Distributions.mode(d::HyperparameterMarginalDistribution)
    _compute_and_cache_mode!(d)
    return d._mode
end

# ==================== CDF and Quantile ====================

"""
    Distributions.cdf(d::HyperparameterMarginalDistribution, x::Real)

Computes CDF at point x via direct numerical integration over
the full hyperparameter space. No caching - computed fresh each time.

CDF(x) = P(θⱼ ≤ x) = ∫...∫ π(θ|y) dθ₁...dθₙ where θⱼ ∈ [lower_bound, x]

This avoids double integration by using the interpolated posterior directly.
"""
function Distributions.cdf(d::HyperparameterMarginalDistribution, x::Real)
    # Get integration bounds
    integration_bounds = d.approx.exploration.integration_bounds
    lower_bounds = integration_bounds[:, 1]
    upper_bounds = integration_bounds[:, 2]

    d_min, d_max = d.bounds
    # Check if x is outside support
    if x <= d_min
        return 0.0
    elseif x >= d_max
        return 1.0
    end

    # Set up integration bounds with marginal dimension bounded by x
    cdf_lower = copy(lower_bounds)
    cdf_upper = copy(upper_bounds)
    cdf_upper[d.marginal_dim] = x

    # Integrand: normalized posterior density
    function cdf_integrand(θ_vec)
        θ = θ_vec[1:end]
        try
            log_density = d.approx(θ)
            return exp(log_density - d.log_normalization_constant)
        catch
            return 0.0
        end
    end

    # Integrate to get CDF value
    integral_result, _ = hcubature(
        cdf_integrand, cdf_lower, cdf_upper;
        rtol = d.rtol, atol = d.atol
    )

    return integral_result
end

function Distributions.quantile(d::HyperparameterMarginalDistribution, q::Real)
    if !(0 <= q <= 1)
        throw(ArgumentError("quantile argument must be in [0,1]"))
    end

    # Handle edge cases
    if q == 0.0
        return minimum(d)
    elseif q == 1.0
        return maximum(d)
    end

    # Use root finding to solve cdf(x) = q
    # Get support bounds for bracketing
    lower_bound = minimum(d)
    upper_bound = maximum(d)

    # Robust root finding
    return find_zero(
        x -> Distributions.cdf(d, x) - q,
        (lower_bound, upper_bound);
        xatol = 1.0e-10, xrtol = 1.0e-10
    )
end

# ==================== Sampling ====================

function Base.rand(rng::AbstractRNG, d::HyperparameterMarginalDistribution)
    # Inverse transform sampling
    return quantile(d, rand(rng))
end

Base.rand(d::HyperparameterMarginalDistribution) = rand(Random.GLOBAL_RNG, d)

# ==================== Support Methods ====================

function Distributions.minimum(d::HyperparameterMarginalDistribution)
    return d.bounds[1]
end

function Distributions.maximum(d::HyperparameterMarginalDistribution)
    return d.bounds[2]
end

function Distributions.insupport(d::HyperparameterMarginalDistribution, x::Real)
    lower, upper = d.bounds
    return lower <= x <= upper && isfinite(x)
end

# ==================== Higher-Order Statistics ====================

function Distributions.skewness(d::HyperparameterMarginalDistribution)
    # Use cached moments and numerical integration for third central moment
    μ = mean(d)
    σ = sqrt(var(d))

    return _integrate_over_joint(d, (θⱼ, _) -> ((θⱼ - μ) / σ)^3, 1)
end

function Distributions.kurtosis(d::HyperparameterMarginalDistribution)
    # Use cached moments and numerical integration for fourth central moment
    μ = mean(d)
    σ = sqrt(var(d))

    return _integrate_over_joint(d, (θⱼ, _) -> ((θⱼ - μ) / σ)^4, 1) - 3.0
end

function Distributions.entropy(d::HyperparameterMarginalDistribution)
    # Numerical integration: -∫ π(θⱼ|y) * log(π(θⱼ|y)) dθⱼ
    # Note: we integrate over the full joint space to avoid double integration
    return _integrate_over_joint(d, (θⱼ, log_density) -> -log_density, 1)
end

# ==================== Parameter Access ====================

"""
    params(d::HyperparameterMarginalDistribution)

Return the parameters as a tuple (approx, marginal_dim, rtol, atol).
"""
Distributions.params(d::HyperparameterMarginalDistribution) = (
    d.approx, d.marginal_dim, d.rtol, d.atol,
)

"""
    partype(::Type{HyperparameterMarginalDistribution{T}}) where T

Return the parameter type.
"""
Distributions.partype(::Type{HyperparameterMarginalDistribution{T}}) where {T} = T

# ==================== Custom Show Method ====================

function Base.show(io::IO, d::HyperparameterMarginalDistribution)
    n_dims = length(d.approx.exploration.grid_points[1].θ)
    lower, upper = d.bounds

    println(io, "HyperparameterMarginalDistribution{", eltype(d.rtol), "}:")
    println(io, "  Marginal dimension: ", d.marginal_dim, " (of ", n_dims, ")")
    println(io, "  Support (natural space): [", @sprintf("%.4f", lower), ", ", @sprintf("%.4f", upper), "]")
    println(io, "  Integration tolerance: rtol=", d.rtol, ", atol=", d.atol)

    # Show computed statistics if available
    if !isnothing(d._mode)
        println(io, "  Mode: ", @sprintf("%.4f", d._mode))
    end
    if !isnothing(d._moments)
        println(io, "  Mean: ", @sprintf("%.4f", d._moments[1]))
        println(io, "  Std: ", @sprintf("%.4f", sqrt(d._moments[2])))
    end

    return print(io, "  Use logpdf(), mean(), quantile(), rand() for analysis")
end
