using Distributions
using HCubature
using Random
using Roots
using Printf

export HyperparameterMarginalDistribution

"""
    HyperparameterMarginalDistribution{T} <: ContinuousUnivariateDistribution

A distribution representing the marginal distribution of a single hyperparameter
from a `HyperparameterPosteriorApproximation`. This provides a user-friendly
Distributions.jl interface for hyperparameter marginals.

This implementation uses lazy computation with caching for expensive operations
like moments (mean, var) to ensure high performance. CDF and quantile are computed
directly when needed without pre-computing splines, as they are typically used
sparingly (e.g., for confidence intervals).

# Fields (Internal)
- `approx::HyperparameterPosteriorApproximation`: The posterior approximation object
- `marginal_dim::Int`: The dimension (1-indexed) for which to compute the marginal
- `rtol::T`: Relative tolerance for numerical integration
- `atol::T`: Absolute tolerance for numerical integration

# Cached Fields (Internal, Lazy-Loaded)
- `_moments`: A cached tuple of (mean, var), computed on first request

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

# Use standard Distribution interface
logpdf(marginal_dist, 0.5)      # Fast - wraps existing function
mean(marginal_dist)             # Computed via numerical integration, cached
quantile(marginal_dist, 0.95)   # Computed via root finding when needed
rand(marginal_dist)             # Inverse transform sampling
```
"""
mutable struct HyperparameterMarginalDistribution{T} <: ContinuousUnivariateDistribution
    approx::HyperparameterPosteriorApproximation
    marginal_dim::Int
    rtol::T
    atol::T

    # Normalization constant to ensure PDF integrates to 1
    log_normalization_constant::T

    # --- Caches for expensive, derived quantities ---
    # Initialized to `nothing` and computed on first use
    _moments::Union{Nothing, NTuple{2, T}}

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

        # Compute normalization constant via single integration over full space
        bounds = approx.exploration.integration_bounds
        lower_bounds = bounds[:, 1]
        upper_bounds = bounds[:, 2]

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
            normalization_integrand, lower_bounds, upper_bounds;
            rtol = T(rtol), atol = T(atol)
        )

        log_normalization_constant = T(log(normalization_integral))

        return new{T}(
            approx, marginal_dim, T(rtol), T(atol),
            log_normalization_constant, nothing
        )
    end
end

# ==================== Core PDF/LOGPDF Methods ====================

function Distributions.logpdf(d::HyperparameterMarginalDistribution, x::Real)
    unnormalized_logpdf = hyperparameter_marginal_logpdf(
        d.approx, d.marginal_dim, Float64(x);
        rtol = d.rtol, atol = d.atol
    )
    return unnormalized_logpdf - d.log_normalization_constant
end

function Distributions.pdf(d::HyperparameterMarginalDistribution, x::Real)
    return exp(logpdf(d, x))
end

# ==================== Moment Calculations (mean, var) ====================

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

    # Get integration bounds from exploration
    bounds = d.approx.exploration.integration_bounds
    lower_bounds = bounds[:, 1]
    upper_bounds = bounds[:, 2]

    # Integrand function that computes both E[θⱼ] and E[θⱼ²] simultaneously
    function integrand(θ_vec)
        # Convert to proper format for interpolant
        θ = θ_vec[1:end]  # HCubature passes Vector{Float64}

        # Evaluate the interpolated posterior
        try
            log_density = d.approx(θ)
            # Apply normalization to get proper density
            normalized_density = exp(log_density - d.log_normalization_constant)

            # Extract the marginal parameter value
            θⱼ = θ[d.marginal_dim]

            # Return [θⱼ * π(θ|y), θⱼ² * π(θ|y)]
            return [θⱼ * normalized_density, θⱼ^2 * normalized_density]
        catch
            # Return zeros if interpolation fails (outside domain)
            return [0.0, 0.0]
        end
    end

    # Perform integration
    result, _ = hcubature(
        integrand, lower_bounds, upper_bounds;
        rtol = d.rtol, atol = d.atol
    )

    # Extract moments
    first_moment = result[1]  # E[θⱼ]
    second_moment = result[2]  # E[θⱼ²]

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

# ==================== CDF and Quantile (Computed Directly) ====================

"""
    _compute_cdf(d::HyperparameterMarginalDistribution, x::Real)

(Private) Computes CDF at point x via direct numerical integration over
the full hyperparameter space. No caching - computed fresh each time.

CDF(x) = P(θⱼ ≤ x) = ∫...∫ π(θ|y) dθ₁...dθₙ where θⱼ ∈ [lower_bound, x]

This avoids double integration by using the interpolated posterior directly.
"""
function _compute_cdf(d::HyperparameterMarginalDistribution, x::Real)
    # Get integration bounds
    bounds = d.approx.exploration.integration_bounds
    lower_bounds = bounds[:, 1]
    upper_bounds = bounds[:, 2]

    # Check if x is outside support
    marginal_lower = lower_bounds[d.marginal_dim]
    marginal_upper = upper_bounds[d.marginal_dim]

    if x <= marginal_lower
        return 0.0
    elseif x >= marginal_upper
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

function Distributions.cdf(d::HyperparameterMarginalDistribution, x::Real)
    return _compute_cdf(d, x)
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
        x -> _compute_cdf(d, x) - q,
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
    return d.approx.exploration.integration_bounds[d.marginal_dim, 1]
end

function Distributions.maximum(d::HyperparameterMarginalDistribution)
    return d.approx.exploration.integration_bounds[d.marginal_dim, 2]
end

function Distributions.insupport(d::HyperparameterMarginalDistribution, x::Real)
    bounds = d.approx.exploration.integration_bounds
    lower = bounds[d.marginal_dim, 1]
    upper = bounds[d.marginal_dim, 2]
    return lower <= x <= upper && isfinite(x)
end

# ==================== Higher-Order Statistics ====================

function Distributions.skewness(d::HyperparameterMarginalDistribution)
    # Use cached moments and numerical integration for third central moment
    μ = mean(d)
    σ = sqrt(var(d))
    bounds = d.approx.exploration.integration_bounds
    lower_bounds = bounds[:, 1]
    upper_bounds = bounds[:, 2]

    function skewness_integrand(θ_vec)
        θ = θ_vec[1:end]
        try
            log_density = d.approx(θ)
            normalized_density = exp(log_density - d.log_normalization_constant)
            θⱼ = θ[d.marginal_dim]
            return ((θⱼ - μ) / σ)^3 * normalized_density
        catch
            return 0.0
        end
    end

    result, _ = hcubature(
        skewness_integrand, lower_bounds, upper_bounds;
        rtol = d.rtol, atol = d.atol
    )

    return result
end

function Distributions.kurtosis(d::HyperparameterMarginalDistribution)
    # Use cached moments and numerical integration for fourth central moment minus 3
    μ = mean(d)
    σ = sqrt(var(d))
    bounds = d.approx.exploration.integration_bounds
    lower_bounds = bounds[:, 1]
    upper_bounds = bounds[:, 2]

    function kurtosis_integrand(θ_vec)
        θ = θ_vec[1:end]
        try
            log_density = d.approx(θ)
            normalized_density = exp(log_density - d.log_normalization_constant)
            θⱼ = θ[d.marginal_dim]
            return ((θⱼ - μ) / σ)^4 * normalized_density
        catch
            return 0.0
        end
    end

    result, _ = hcubature(
        kurtosis_integrand, lower_bounds, upper_bounds;
        rtol = d.rtol, atol = d.atol
    )

    return result - 3.0  # Excess kurtosis
end

function Distributions.entropy(d::HyperparameterMarginalDistribution)
    # Numerical integration: -∫ π(θⱼ|y) * log(π(θⱼ|y)) dθⱼ
    # We need to integrate over the marginal, not the full space
    bounds = d.approx.exploration.integration_bounds
    lower = bounds[d.marginal_dim, 1]
    upper = bounds[d.marginal_dim, 2]

    function entropy_integrand(x_vec)
        x = x_vec[1]
        try
            log_p = logpdf(d, x)
            p = exp(log_p)
            return p > 0 ? -p * log_p : 0.0
        catch
            return 0.0
        end
    end

    result, _ = hcubature(
        entropy_integrand, [lower], [upper];
        rtol = d.rtol, atol = d.atol
    )

    return result
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
    bounds = d.approx.exploration.integration_bounds
    lower = bounds[d.marginal_dim, 1]
    upper = bounds[d.marginal_dim, 2]

    println(io, "HyperparameterMarginalDistribution{", eltype(d.rtol), "}:")
    println(io, "  Marginal dimension: ", d.marginal_dim, " (of ", n_dims, ")")
    println(io, "  Support: [", @sprintf("%.4f", lower), ", ", @sprintf("%.4f", upper), "]")
    println(io, "  Integration tolerance: rtol=", d.rtol, ", atol=", d.atol)

    # Show computed statistics if available
    if !isnothing(d._moments)
        println(io, "  Mean: ", @sprintf("%.4f", d._moments[1]))
        println(io, "  Std: ", @sprintf("%.4f", sqrt(d._moments[2])))
    end

    return print(io, "  Use logpdf(), mean(), quantile(), rand() for analysis")
end
