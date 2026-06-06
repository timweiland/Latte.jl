"""
    AR1Correlation <: ContinuousUnivariateDistribution

PC prior for the lag-1 correlation parameter ρ of an AR(1) process.

The distance measure is d(ρ) = √(-log(1-ρ²)), which equals the KLD^{1/2}
from the AR(1) model to the base model (iid, ρ=0).

Calibrated via P(|ρ| > U) = α (or P(ρ > U) = α when `positive_only=true`).

# Constructors
    AR1Correlation(U; α=0.05, positive_only=false) # calibrated
    AR1Correlation(λ, positive_only::Bool)          # direct λ parameterization

# Reference
Sørbye & Rue (2017). "Penalised Complexity priors for stationary autoregressive processes."
"""
struct AR1Correlation <: ContinuousUnivariateDistribution
    λ::Float64
    positive_only::Bool
end

# Calibrated constructor
function AR1Correlation(U::Real; α::Real = 0.05, positive_only::Bool = false)
    0 < U < 1 || throw(ArgumentError("U must be in (0,1), got $U"))
    0 < α < 1 || throw(ArgumentError("α must be in (0,1), got $α"))
    d_U = sqrt(-log(1 - U^2))
    λ = lambda_from_tail(d_U, α)
    return AR1Correlation(λ, positive_only)
end

function Distributions.support(d::AR1Correlation)
    if d.positive_only
        return RealInterval(0.0, 1.0)
    else
        return RealInterval(-1.0, 1.0)
    end
end

Distributions.minimum(d::AR1Correlation) = d.positive_only ? 0.0 : -1.0
Distributions.maximum(::AR1Correlation) = 1.0

# The PC prior shrinks toward the base AR(1) model (ρ=0), so the mode is at 0.
Distributions.mode(::AR1Correlation) = 0.0

function Distributions.logpdf(d::AR1Correlation, ρ::Real)
    # Support check — strict inequalities exclude ρ=0 in both branches
    if d.positive_only
        (0 < ρ < 1) || return -Inf
    else
        (-1 < ρ < 1) || return -Inf
        ρ == 0 && return -Inf
    end

    ρ² = ρ^2
    one_minus_ρ² = 1 - ρ²
    # Work with d² to avoid a sqrt just to take log of it:
    # d(ρ)² = -log(1-ρ²), log(d(ρ)) = 0.5*log(d²), d(ρ) = sqrt(d²)
    d²_ρ = -log(one_minus_ρ²)
    d_ρ = sqrt(d²_ρ)
    lp = log(d.λ) + log(abs(ρ)) - log(one_minus_ρ²) - 0.5 * log(d²_ρ) - d.λ * d_ρ
    if !d.positive_only
        lp -= log(2)
    end
    return lp
end

function Base.rand(rng::AbstractRNG, d::AR1Correlation)
    dist = rand(rng, Exponential(1 / d.λ))
    ρ = sqrt(1 - exp(-dist^2))
    if !d.positive_only && rand(rng, Bool)
        ρ = -ρ
    end
    return ρ
end

# --- CDF / quantile / median --------------------------------------------------
# d(ρ) = √(-log(1-ρ²)) is the distance; d ~ Exponential(rate λ), so the cdf /
# quantile compose the Exponential cdf/quantile with the distance transform.
# mean/var/std are intentionally undefined — no closed form for positive_only;
# two-sided mean is 0 by symmetry but var/std remain non-elementary.
_ar1_distance(ρ) = sqrt(-log1p(-ρ^2))
_ar1_rho_from_distance(dist) = sqrt(-expm1(-dist^2))

function Distributions.cdf(d::AR1Correlation, ρ::Real)
    if d.positive_only
        ρ <= 0 && return 0.0
        ρ >= 1 && return 1.0
        return -expm1(-d.λ * _ar1_distance(ρ))            # 1 - exp(-λ·d(ρ))
    else
        ρ <= -1 && return 0.0
        ρ >= 1 && return 1.0
        ρ == 0 && return 0.5
        half_tail = 0.5 * exp(-d.λ * _ar1_distance(abs(ρ)))
        return ρ > 0 ? 1.0 - half_tail : half_tail
    end
end

function Distributions.quantile(d::AR1Correlation, p::Real)
    0 <= p <= 1 || throw(DomainError(p, "quantile probability must be in [0, 1]"))
    p == 0 && return Distributions.minimum(d)
    p == 1 && return 1.0
    if d.positive_only
        dist = -log1p(-p) / d.λ                            # Exponential quantile, rate λ
        return _ar1_rho_from_distance(dist)
    else
        p == 0.5 && return 0.0
        if p > 0.5
            dist = -log(2 * (1 - p)) / d.λ
            return _ar1_rho_from_distance(dist)
        else
            dist = -log(2 * p) / d.λ
            return -_ar1_rho_from_distance(dist)
        end
    end
end

Distributions.median(d::AR1Correlation) = quantile(d, 0.5)
