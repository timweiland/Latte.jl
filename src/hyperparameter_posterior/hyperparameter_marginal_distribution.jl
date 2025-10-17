using Distributions
using HCubature
using Random
using Roots
using Printf

export HyperparameterMarginalDistribution

"""
    HyperparameterMarginalDistribution{T, Space} <: ContinuousUnivariateDistribution

A distribution representing the marginal distribution of a single hyperparameter
from a `HyperparameterPosteriorApproximation`. This provides a user-friendly
Distributions.jl interface for hyperparameter marginals.

By default, all operations use **natural space** (constrained parameter space).
Set `space=:working` to use working space (unconstrained optimization space).

This implementation uses lazy computation with caching for expensive operations
like moments (mean, var) to ensure high performance. CDF and quantile are computed
directly when needed without pre-computing splines, as they are typically used
sparingly (e.g., for confidence intervals).

# Type Parameters
- `T`: Numeric type (typically Float64)
- `Space`: `Val{:natural}` (default) or `Val{:working}` for compile-time dispatch

# Fields (Internal)
- `approx::HyperparameterPosteriorApproximation`: The posterior approximation object
- `marginal_dim::Int`: The dimension (1-indexed) for which to compute the marginal
- `spec::HyperparameterSpec`: Hyperparameter specification for conversions
- `marginal_transform`: Bijector for this dimension (natural ↔ working conversion)
- `rtol::T`: Relative tolerance for numerical integration
- `atol::T`: Absolute tolerance for numerical integration

# Cached Fields (Internal, Lazy-Loaded)
- `_moments`: A cached tuple of (mean, var), computed on first request

# Constructor
```julia
HyperparameterMarginalDistribution(
    approx::HyperparameterPosteriorApproximation,
    marginal_dim::Int;
    spec::HyperparameterSpec,
    rtol::Real = 1.0e-3,
    atol::Real = 1.0e-6,
    space::Symbol = :natural
)
```

# Example
```julia
# Create marginal distribution for first hyperparameter (natural space by default)
marginal_dist = HyperparameterMarginalDistribution(approx, 1, spec=model.hyperparameter_spec)

# Use standard Distribution interface - all in natural space
logpdf(marginal_dist, 2.5)      # Evaluate at σ = 2.5 (natural space)
mean(marginal_dist)             # Mean in natural space
quantile(marginal_dist, 0.95)   # 95th quantile in natural space
rand(marginal_dist)             # Sample in natural space

# For advanced users: working space
marginal_dist_working = HyperparameterMarginalDistribution(approx, 1, spec=model.hyperparameter_spec, space=:working)
logpdf(marginal_dist_working, 0.9)  # Evaluate at log(σ) = 0.9 (working space)
```
"""
mutable struct HyperparameterMarginalDistribution{T, Space} <: ContinuousUnivariateDistribution
    approx::HyperparameterPosteriorApproximation
    marginal_dim::Int
    spec::HyperparameterSpec
    marginal_transform  # Bijector for this dimension
    transform_to_target_space # Transform that takes working space to `Space`
    rtol::T
    atol::T

    # Working space bounds (internal representation)
    working_bounds::NTuple{2, T}

    # Normalization constant to ensure PDF integrates to 1 (in working space)
    log_normalization_constant::T

    # --- Caches for expensive, derived quantities ---
    # Initialized to `nothing` and computed on first use
    _moments::Union{Nothing, NTuple{2, T}}

    function HyperparameterMarginalDistribution(
            approx::HyperparameterPosteriorApproximation,
            marginal_dim::Int;
            spec::HyperparameterSpec,
            rtol::Real = 1.0e-3,
            atol::Real = 1.0e-6,
            space::Symbol = :natural
        )
        T = Float64

        if space ∉ (:natural, :working)
            throw(ArgumentError("space must be :natural or :working, got :$space"))
        end

        # Input validation - get dimensions from the first grid point
        n_dims = length(approx.exploration.grid_points[1].θ)
        if !(1 <= marginal_dim <= n_dims)
            throw(ArgumentError("marginal_dim must be between 1 and $n_dims"))
        end

        # Extract bijector for this dimension from spec
        free_param_names = collect(keys(spec.free))
        param_name = free_param_names[marginal_dim]
        marginal_transform = spec.free[param_name].transform
        if space == :natural
            transform_to_target_space = inverse(marginal_transform)
        else
            # Target == working
            transform_to_target_space = identity
        end

        # Extract working space bounds for this dimension
        bounds = approx.exploration.integration_bounds
        lower_bound_working = T(bounds[marginal_dim, 1])
        upper_bound_working = T(bounds[marginal_dim, 2])
        working_bounds = (lower_bound_working, upper_bound_working)

        # Compute normalization constant via single integration over full space (in working space)
        lower_bounds_full = bounds[:, 1]
        upper_bounds_full = bounds[:, 2]

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

        return new{T, Val{space}}(
            approx, marginal_dim, spec, marginal_transform, transform_to_target_space, T(rtol), T(atol),
            working_bounds, log_normalization_constant, nothing
        )
    end
end

# ==================== Core PDF/LOGPDF Methods ====================
function _working_logpdf(d::HyperparameterMarginalDistribution, x::Real)
    unnormalized_logpdf = hyperparameter_marginal_logpdf(
        d.approx, d.marginal_dim, x;
        rtol = d.rtol, atol = d.atol
    )
    return unnormalized_logpdf - d.log_normalization_constant
end

"""
    Distributions.logpdf(d::HyperparameterMarginalDistribution{T, Val{:working}}, x::Real)

Evaluate log-density in working space (no transformation needed).
"""
function Distributions.logpdf(d::HyperparameterMarginalDistribution{T, Val{:working}}, x::Real) where {T}
    return _working_logpdf(d, x)
end

"""
    Distributions.logpdf(d::HyperparameterMarginalDistribution{T, Val{:natural}}, x::Real)

Evaluate log-density in natural space (includes Jacobian correction).

For x in natural space, converts to working space and adds logabsdetjac:
log π(x) = log π(T(x)) + log|dT/dx|
"""
function Distributions.logpdf(d::HyperparameterMarginalDistribution{T, Val{:natural}}, x::Real) where {T}
    # Convert to working space
    x_working = d.marginal_transform(x)
    working_logpdf = _working_logpdf(d, x_working)
    # Add Jacobian correction: log|dT/dx|
    jacobian_correction = logabsdetjac(d.marginal_transform, x)

    return working_logpdf + jacobian_correction
end

function Distributions.pdf(d::HyperparameterMarginalDistribution, x::Real)
    return exp(logpdf(d, x))
end

# ==================== Moment Calculations (mean, var) ====================

function _integrate_in_working_space(d::HyperparameterMarginalDistribution, fun, output_size)
    # Get integration bounds from exploration
    bounds = d.approx.exploration.integration_bounds
    lower_bounds = bounds[:, 1]
    upper_bounds = bounds[:, 2]

    function integrand(θ_vec)
        θ = θ_vec[1:end]
        # Evaluate the interpolated posterior
        try
            log_density = d.approx(θ)
            normalized_density = exp(log_density - d.log_normalization_constant)

            θⱼ = d.transform_to_target_space(θ[d.marginal_dim])
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

    first_moment, second_moment = _integrate_in_working_space(d, (θ, _) -> [θ, θ^2], 2)

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

function _compute_cdf_working(d::HyperparameterMarginalDistribution, x::Real)
    # Get integration bounds
    bounds = d.approx.exploration.integration_bounds
    lower_bounds = bounds[:, 1]
    upper_bounds = bounds[:, 2]

    d_min, d_max = lower_bounds[d.marginal_dim], upper_bounds[d.marginal_dim]
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

    ## Integrand: normalized posterior density
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

"""
    Distributions.cdf(d::HyperparameterMarginalDistribution, x::Real)

Computes CDF at point x via direct numerical integration over
the full hyperparameter space. No caching - computed fresh each time.

CDF(x) = P(θⱼ ≤ x) = ∫...∫ π(θ|y) dθ₁...dθₙ where θⱼ ∈ [lower_bound, x]

This avoids double integration by using the interpolated posterior directly.
"""
function Distributions.cdf(d::HyperparameterMarginalDistribution{T, Val{:working}}, x::Real) where {T}
    return _compute_cdf_working(d, x)
end

function Distributions.cdf(d::HyperparameterMarginalDistribution{T, Val{:natural}}, x::Real) where {T}
    return _compute_cdf_working(d, d.marginal_transform(x))
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
    return d.transform_to_target_space(d.approx.exploration.integration_bounds[d.marginal_dim, 1])
end

function Distributions.maximum(d::HyperparameterMarginalDistribution)
    return d.transform_to_target_space(d.approx.exploration.integration_bounds[d.marginal_dim, 2])
end

function Distributions.insupport(d::HyperparameterMarginalDistribution, x::Real)
    lower = Distributions.minimum(d)
    upper = Distributions.maximum(d)
    return lower <= x <= upper && isfinite(x)
end

# ==================== Higher-Order Statistics ====================

function Distributions.skewness(d::HyperparameterMarginalDistribution)
    # Use cached moments and numerical integration for third central moment
    μ = mean(d)
    σ = sqrt(var(d))

    return _integrate_in_working_space(d, (θ, _) -> ((θ - μ) / σ)^3, 1)
end

function Distributions.kurtosis(d::HyperparameterMarginalDistribution)
    # Use cached moments and numerical integration for fourth central moment minus 3
    μ = mean(d)
    σ = sqrt(var(d))

    return _integrate_in_working_space(d, (θ, _) -> ((θ - μ) / σ)^4, 1)
end

function Distributions.entropy(d::HyperparameterMarginalDistribution{T, Val{:working}}) where {T}
    # Numerical integration: -∫ π(θⱼ|y) * log(π(θⱼ|y)) dθⱼ
    return _integrate_in_working_space(d, (θ, lpdf) -> -lpdf, 1)
end

function Distributions.entropy(d::HyperparameterMarginalDistribution{T, Val{:natural}}) where {T}
    # Numerical integration: -∫ π(θⱼ|y) * log(π(θⱼ|y)) dθⱼ
    return _integrate_in_working_space(d, (θ, lpdf) -> -lpdf - logabsdetjac(d.marginal_transform, θ), 1)
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
    lower, upper = Distributions.minimum(d), Distributions.maximum(d)

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
