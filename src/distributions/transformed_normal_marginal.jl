# Natural-space marginal whose working-space density is a Gaussian
# pushed through a monotone bijector. Quantiles by push-through, pdf
# by change of variables, moments by Gauss-Hermite quadrature.

using Distributions
using FastGaussQuadrature: gausshermite
using ForwardDiff

export TransformedNormalMarginal

"""
    TransformedNormalMarginal{T, F, Finv} <: ContinuousUnivariateDistribution

Distribution of `Y = inv_transform(X)` where `X ~ Normal(μ_working, σ_working)`
and `inv_transform` is monotonically increasing.

Used for hyperparameter marginals from TMB-style inference: the Laplace
approximation gives a working-space Gaussian, and the user-facing
marginal on the natural coordinate is its image under the inverse
transform.

# Arithmetic
- `quantile(d, p)` is exact: `inv_transform(quantile(Normal(μ, σ), p))`.
- `cdf` and `logcdf` are exact: pushed through the monotone transform.
- `pdf` uses change of variables; the Jacobian is computed by
  `ForwardDiff.derivative` on the transform.
- `mean`, `var`, `std` use 32-node Gauss-Hermite quadrature on the
  working-space Gaussian — accurate to machine precision for smooth
  inverse transforms (`exp`, `sigmoid`, `identity`).
"""
struct TransformedNormalMarginal{T <: Real, F, Finv} <: ContinuousUnivariateDistribution
    μ_working::T
    σ_working::T
    transform::F          # natural → working (monotone increasing)
    inv_transform::Finv   # working → natural
end

# 32 Gauss-Hermite nodes — converged to ~machine epsilon for the
# smooth transforms we care about (exp, sigmoid, identity).
const _GH32_NODES, _GH32_WEIGHTS = gausshermite(32)
const _INV_SQRTPI = 1 / sqrt(π)

Distributions.minimum(::TransformedNormalMarginal) = -Inf
Distributions.maximum(::TransformedNormalMarginal) = Inf

function Distributions.quantile(d::TransformedNormalMarginal, p::Real)
    q_wh = quantile(Normal(d.μ_working, d.σ_working), p)
    return d.inv_transform(q_wh)
end

Distributions.median(d::TransformedNormalMarginal) = d.inv_transform(d.μ_working)

# Standard `Distribution` semantics outside the natural support: cdf
# saturates to 0/1, pdf is 0. Support boundaries are detected via
# `_support_side` (`d.transform` throwing or returning non-finite).
function Distributions.cdf(d::TransformedNormalMarginal, x::Real)
    side = _support_side(d, x)
    side === :below && return 0.0
    side === :above && return 1.0
    return cdf(Normal(d.μ_working, d.σ_working), side)
end

function Distributions.logcdf(d::TransformedNormalMarginal, x::Real)
    side = _support_side(d, x)
    side === :below && return -Inf
    side === :above && return 0.0
    return logcdf(Normal(d.μ_working, d.σ_working), side)
end

function Distributions.pdf(d::TransformedNormalMarginal, x::Real)
    # Change of variables: f_Y(x) = f_X(transform(x)) · |d transform / dx|
    side = _support_side(d, x)
    side isa Symbol && return 0.0
    deriv = ForwardDiff.derivative(d.transform, x)
    return pdf(Normal(d.μ_working, d.σ_working), side) * abs(deriv)
end

function Distributions.logpdf(d::TransformedNormalMarginal, x::Real)
    side = _support_side(d, x)
    side isa Symbol && return -Inf
    deriv = ForwardDiff.derivative(d.transform, x)
    return logpdf(Normal(d.μ_working, d.σ_working), side) + log(abs(deriv))
end

# Returns `transform(x)` when in support; `:below` / `:above` otherwise.
# Side is decided by comparison with the median (which is always in
# support since it's `inv_transform(μ_working)`).
function _support_side(d::TransformedNormalMarginal, x::Real)
    η = try
        d.transform(x)
    catch err
        err isa DomainError || rethrow()
        return x < median(d) ? :below : :above
    end
    isfinite(η) || return x < median(d) ? :below : :above
    return η
end

function Random.rand(rng::AbstractRNG, d::TransformedNormalMarginal)
    return d.inv_transform(d.μ_working + d.σ_working * randn(rng))
end

# Gauss-Hermite quadrature of E[inv_transform(X)] with X ~ Normal(μ, σ).
function Distributions.mean(d::TransformedNormalMarginal)
    s = zero(d.μ_working)
    sqrt2σ = sqrt(2) * d.σ_working
    @inbounds for i in eachindex(_GH32_NODES)
        η = d.μ_working + sqrt2σ * _GH32_NODES[i]
        s += _GH32_WEIGHTS[i] * d.inv_transform(η)
    end
    return s * _INV_SQRTPI
end

function Distributions.var(d::TransformedNormalMarginal)
    # E[Y²] - E[Y]²; compute both in one pass.
    s1 = zero(d.μ_working)
    s2 = zero(d.μ_working)
    sqrt2σ = sqrt(2) * d.σ_working
    @inbounds for i in eachindex(_GH32_NODES)
        η = d.μ_working + sqrt2σ * _GH32_NODES[i]
        y = d.inv_transform(η)
        w = _GH32_WEIGHTS[i]
        s1 += w * y
        s2 += w * y * y
    end
    m = s1 * _INV_SQRTPI
    m2 = s2 * _INV_SQRTPI
    return m2 - m * m
end

Distributions.std(d::TransformedNormalMarginal) = sqrt(var(d))
