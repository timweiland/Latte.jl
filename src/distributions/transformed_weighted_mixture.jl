using Distributions
using Bijectors
using HCubature
using Optim
using Random
using Roots
using Printf

export TransformedWeightedMixture, pushforward

"""
    TransformedWeightedMixture{T, B} <: ContinuousUnivariateDistribution

A weighted mixture distribution transformed through a bijector, representing
observation marginals (fitted values) in INLA.

This distribution transforms linear predictor marginals through the inverse link
function to obtain marginals for observations. For a linear predictor η ~ WeightedMixture
and link function g, this represents Y = g⁻¹(η).

# Mathematical Background

For a link function g and its inverse g⁻¹:
- Linear predictor space: η (unbounded, Gaussian-like)
- Observation space: Y = g⁻¹(η) (constrained to observation support)

The PDF transformation uses change of variables:
```
p_Y(y) = p_η(g(y)) × |dg/dy|
```

where g is the link function (stored as the `bijector` field).

# Type Parameters
- `T`: Numeric type (typically Float64)
- `B`: Bijector type

# Fields
- `base_distribution::WeightedMixture{T}`: Distribution in linear predictor space (η)
- `bijector::B`: The **link function** g where η = g(μ). Use `inverse(bijector)` to get g⁻¹.
- `rtol::T`: Relative tolerance for numerical integration (default: 1e-3)
- `atol::T`: Absolute tolerance for numerical integration (default: 1e-6)

# Cached Fields (Internal, Lazy-Loaded)
- `_moments::Union{Nothing, NTuple{2, T}}`: Cached (mean, var)
- `_support::Union{Nothing, NTuple{2, T}}`: Cached (min, max) bounds

# Constructor
```julia
TransformedWeightedMixture(
    base_distribution::WeightedMixture,
    bijector::Bijector;
    rtol::Real = 1.0e-3,
    atol::Real = 1.0e-6
)
```

# Examples
```julia
# Linear predictor marginal from INLA
η_marginal = WeightedMixture([Normal(2.0, 0.5), Normal(2.5, 0.3)], [0.6, 0.4])

# Transform through inverse log (exp) for Poisson observations
link_bijector = elementwise(log)  # Link function
obs_marginal = TransformedWeightedMixture(η_marginal, link_bijector)

# Now obs_marginal represents the Poisson rate parameter λ
mean(obs_marginal)  # Expected value of λ
quantile(obs_marginal, 0.95)  # 95th percentile of λ
```

# Implementation Notes
- **Bijector convention**: Stores the link function g (forward: observation → predictor).
  Use `inverse(bijector)` internally to transform predictor → observation.
- **Numerical integration**: Mean and variance computed via 1D quadrature over base distribution.
- **Lazy caching**: Moments computed once on first request.
- **Change of variables**: PDF correctly accounts for Jacobian using `logabsdetjac`.
"""
mutable struct TransformedWeightedMixture{T, B, D <: ContinuousUnivariateDistribution} <: ContinuousUnivariateDistribution
    base_distribution::D
    bijector::B
    rtol::T
    atol::T

    # --- Caches for expensive, derived quantities ---
    _moments::Union{Nothing, NTuple{2, T}}
    _support::Union{Nothing, NTuple{2, T}}

    # The base may be any 1-D distribution — the linear-predictor `WeightedMixture`
    # it was written for (observation marginals), or e.g. a hyperparameter
    # `SplineMarginalDistribution` (see `pushforward`). The moment/support code
    # only needs `mean`/`std`/`pdf` of the base.
    function TransformedWeightedMixture(
            base_distribution::D,
            bijector::B;
            rtol::Real = 1.0e-6,
            atol::Real = 1.0e-6
        ) where {D <: ContinuousUnivariateDistribution, B}
        T = float(eltype(base_distribution))
        return new{T, B, D}(
            base_distribution, bijector, T(rtol), T(atol),
            nothing, nothing
        )
    end
end

"""
    pushforward(marginal, g)

The distribution of `g(X)` where `X ~ marginal`. Moments are computed by
integration, so `mean(pushforward(m, exp))` is the true `E[exp(X)]` — not
`exp(E[X])` (the Jensen-inequality trap).

`g` may be `exp`, `log`, `identity`, or any `Bijectors` bijector. This is the
way to recover a *derived* hyperparameter's posterior: e.g. when a model
declares `log_α ~ Normal(...)` and uses `α = exp(log_α)`, the posterior of `α`
is `pushforward(result.hyperparameter_marginals.log_α, exp)`. (A *declared*
hyperparameter's marginal is already in natural space, so this isn't needed
there.)
"""
pushforward(m::ContinuousUnivariateDistribution, b::Bijectors.Bijector) =
    TransformedWeightedMixture(m, inverse(b))
pushforward(m::ContinuousUnivariateDistribution, ::typeof(exp)) =
    TransformedWeightedMixture(m, elementwise(log))
pushforward(m::ContinuousUnivariateDistribution, ::typeof(log)) =
    TransformedWeightedMixture(m, elementwise(exp))
pushforward(m::ContinuousUnivariateDistribution, ::typeof(identity)) = m

# ==================== Core PDF/LOGPDF Methods ====================

"""
    Distributions.logpdf(d::TransformedWeightedMixture, y::Real)

Evaluate log-density at observation value y.

Uses change of variables formula:
log p_Y(y) = log p_η(g(y)) + log|dg/dy|

where g is the link function (d.bijector).
"""
function Distributions.logpdf(d::TransformedWeightedMixture, y::Real)
    # Check support - return -Inf if outside
    insupport(d, y) || return -Inf

    # Transform y from observation space to linear predictor space
    # g(y) where g is the link function
    η = d.bijector(y)

    # Base distribution log-density at η
    log_base_density = logpdf(d.base_distribution, η)

    # Jacobian correction: log|dg/dy|
    log_jac = logabsdetjac(d.bijector, y)

    return log_base_density + log_jac
end

function Distributions.pdf(d::TransformedWeightedMixture, y::Real)
    return exp(logpdf(d, y))
end

# ==================== Support Methods ====================

"""
    _compute_and_cache_support!(d::TransformedWeightedMixture)

(Private) Computes and caches the support bounds by transforming base distribution support.

Assumes the bijector is monotonically increasing. Maps bounds through the inverse link.
"""
function _compute_and_cache_support!(d::TransformedWeightedMixture{T}) where {T}
    isnothing(d._support) || return

    # Get base distribution support in linear predictor space
    η_min = minimum(d.base_distribution)
    η_max = maximum(d.base_distribution)

    # Transform to observation space using inverse link
    # Assume monotonically increasing bijector
    inv_link = inverse(d.bijector)

    # Map bounds through inverse link (handles Inf/-Inf correctly)
    y_min = T(inv_link(η_min))
    y_max = T(inv_link(η_max))

    d._support = (y_min, y_max)
    return
end

function Distributions.minimum(d::TransformedWeightedMixture)
    _compute_and_cache_support!(d)
    return d._support[1]
end

function Distributions.maximum(d::TransformedWeightedMixture)
    _compute_and_cache_support!(d)
    return d._support[2]
end

function Distributions.insupport(d::TransformedWeightedMixture, y::Real)
    _compute_and_cache_support!(d)
    lower, upper = d._support

    # Check finiteness first
    isfinite(y) || return false

    # For bounded support (like logistic), use strict inequalities at boundaries
    if isfinite(lower) && isfinite(upper)
        return lower < y < upper
    elseif isfinite(lower)
        return lower < y
    elseif isfinite(upper)
        return y < upper
    else
        return true  # Infinite support both ways
    end
end

# ==================== Moment Calculations ====================

"""
    _integrate_over_base(d::TransformedWeightedMixture, fun)

Helper function to integrate a function over the base distribution (in η space).

Computes: ∫ fun(g⁻¹(η)) * p_η(η) dη
"""
function _integrate_over_base(d::TransformedWeightedMixture, fun)
    # Integration bounds in base (η) space. Clamp the ±10σ window to the base's
    # actual support so a bounded base (e.g. a Uniform, or a hyperparameter
    # SplineMarginal with finite bounds) is never sampled where its density is
    # zero — and, crucially, where the inverse map may be undefined (e.g. logit
    # outside [0,1] → NaN).
    η_mean, η_std = mean(d.base_distribution), std(d.base_distribution)
    η_min = max(η_mean - 10.0 * η_std, minimum(d.base_distribution))
    η_max = min(η_mean + 10.0 * η_std, maximum(d.base_distribution))

    # Handle infinite bounds (unbounded base) with finite approximations.
    if isinf(η_min)
        η_min = -1.0e10
        while pdf(d.base_distribution, η_min) > 1.0e-10
            η_min *= 2
        end
    end

    if isinf(η_max)
        η_max = 1.0e10
        while pdf(d.base_distribution, η_max) > 1.0e-10
            η_max *= 2
        end
    end

    inv_link = inverse(d.bijector)

    function integrand(η_scalar)
        η = η_scalar[1]  # hcubature expects vector input
        # Transform to observation space
        y = inv_link(η)
        # Weight by base density
        weight = pdf(d.base_distribution, η)
        return [fun(y) * weight]  # hcubature expects vector output
    end

    result, err = hcubature(
        integrand, [η_min], [η_max];
        rtol = d.rtol, atol = d.atol
    )

    return result[1]
end

"""
    _compute_and_cache_moments!(d::TransformedWeightedMixture)

(Private) Computes mean and variance via numerical integration and caches the result.

E[Y] = ∫ g⁻¹(η) * p_η(η) dη
E[Y²] = ∫ (g⁻¹(η))² * p_η(η) dη
Var(Y) = E[Y²] - (E[Y])²
"""
function _compute_and_cache_moments!(d::TransformedWeightedMixture{T}) where {T}
    isnothing(d._moments) || return

    # Compute first and second moments via integration
    first_moment = _integrate_over_base(d, y -> y)
    second_moment = _integrate_over_base(d, y -> y^2)

    mean_val = T(first_moment)
    var_val = T(second_moment - first_moment^2)

    # Ensure variance is non-negative
    var_val = max(var_val, zero(T))

    d._moments = (mean_val, var_val)
    return
end

function Distributions.mean(d::TransformedWeightedMixture)
    _compute_and_cache_moments!(d)
    return d._moments[1]
end

function Distributions.var(d::TransformedWeightedMixture)
    _compute_and_cache_moments!(d)
    return d._moments[2]
end

function Distributions.std(d::TransformedWeightedMixture)
    return sqrt(var(d))
end

function Distributions.mode(d::TransformedWeightedMixture)
    x0 = mean(d)
    result = optimize(x -> -logpdf(d, x[1]), [x0])
    return only(Optim.minimizer(result))
end

# ==================== CDF and Quantile ====================

"""
    Distributions.cdf(d::TransformedWeightedMixture, y::Real)

Compute CDF at observation value y.

Uses the transformation: P(Y ≤ y) = P(g⁻¹(η) ≤ y) = P(η ≤ g(y))
(assuming monotone increasing inverse link; handles decreasing case too).
"""
function Distributions.cdf(d::TransformedWeightedMixture, y::Real)
    d_min, d_max = minimum(d), maximum(d)

    # Check bounds
    if y <= d_min
        return 0.0
    elseif y >= d_max
        return 1.0
    end

    # Transform to linear predictor space
    η = d.bijector(y)

    # CDF in base distribution
    # Note: Need to check if bijector is monotone increasing or decreasing
    # For most common cases (exp, logistic), the inverse link is monotone increasing
    # We can detect this by checking a point
    inv_link = inverse(d.bijector)
    test_η1, test_η2 = 0.0, 1.0
    test_y1, test_y2 = inv_link(test_η1), inv_link(test_η2)

    if test_y2 > test_y1
        # Monotone increasing inverse link
        return cdf(d.base_distribution, η)
    else
        # Monotone decreasing inverse link
        return 1.0 - cdf(d.base_distribution, η)
    end
end

function Distributions.quantile(d::TransformedWeightedMixture, p::Real)
    if !(0 <= p <= 1)
        throw(ArgumentError("Quantile argument must be in [0,1]"))
    end

    # Handle edge cases
    if p == 0.0
        return minimum(d)
    elseif p == 1.0
        return maximum(d)
    end

    # Use root finding to solve cdf(y) = p
    lower_bound = minimum(d)
    upper_bound = maximum(d)

    # Handle infinite bounds
    if isinf(lower_bound)
        lower_bound = -1.0e10
        while cdf(d, lower_bound) > 1.0e-10
            lower_bound *= 2
        end
    end

    if isinf(upper_bound)
        upper_bound = 1.0e10
        while cdf(d, upper_bound) < 1 - 1.0e-10
            upper_bound *= 2
        end
    end

    return find_zero(
        y -> cdf(d, y) - p,
        (lower_bound, upper_bound);
        xatol = 1.0e-10, xrtol = 1.0e-10
    )
end

# ==================== Sampling ====================

function Base.rand(rng::AbstractRNG, d::TransformedWeightedMixture)
    # Sample from base distribution in linear predictor space
    η = rand(rng, d.base_distribution)

    # Transform to observation space using inverse link
    inv_link = inverse(d.bijector)
    return inv_link(η)
end

Base.rand(d::TransformedWeightedMixture) = rand(Random.GLOBAL_RNG, d)

# ==================== Custom Show Method ====================

function Base.show(io::IO, d::TransformedWeightedMixture)
    n_components = length(d.base_distribution.components)

    println(io, "TransformedWeightedMixture{", eltype(d.rtol), "}:")
    println(io, "  Base distribution: WeightedMixture with ", n_components, " components")
    println(io, "  Transformation: ", typeof(d.bijector).name.name)

    # Show support if computed
    if !isnothing(d._support)
        lower, upper = d._support
        println(io, "  Support: [", @sprintf("%.4f", lower), ", ", @sprintf("%.4f", upper), "]")
    end

    # Show statistics if computed
    if !isnothing(d._moments)
        println(io, "  Mean: ", @sprintf("%.4f", d._moments[1]))
        println(io, "  Std: ", @sprintf("%.4f", sqrt(d._moments[2])))
    end

    return print(io, "  Integration tolerance: rtol=", d.rtol, ", atol=", d.atol)
end
