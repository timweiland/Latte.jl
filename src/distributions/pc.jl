export PCPrecision, pc_prior_precision
export pc_prior_sigma

import Distributions: ContinuousUnivariateDistribution, RealInterval, support, logpdf, rand, mode

export PrecisionToStdBijector, StdToPrecisionBijector

import Bijectors

# --- PC prior helpers --------------------------------------------------------

"""
    pc_lambda_from_tail(U, α)

Calibrate the PC prior via a tail probability:
P(σ > U) = α  ⇒  λ = -log(α) / U

Arguments:
- U > 0: a "soft upper bound" for σ
- α ∈ (0,1): prior tail mass beyond U
"""
pc_lambda_from_tail(U, α) = -log(α) / U

"""
    pc_prior_sigma(U; α=0.05)

Return the Distributions.jl distribution for σ under the PC prior.
For PC priors on the standard deviation:  σ ~ Exp(λ), with λ as above.

NOTE: In Distributions.jl, `Exponential(θ)` uses θ as the *scale* (mean),
so we pass `1/λ`.
"""
function pc_prior_sigma(U; α = 0.05)
    λ = pc_lambda_from_tail(U, α)
    return Exponential(1 / λ)  # scale = 1/λ
end

# --- (Optional) Induced PC prior on precision τ = 1/σ² -----------------------
# If you prefer to place the prior on τ directly, the induced density is:
#   π(τ) = (λ/2) * τ^(-3/2) * exp(-λ / sqrt(τ)),  τ > 0.
# Below is a small distribution type implementing logpdf + sampling.


struct PCPrecision <: ContinuousUnivariateDistribution
    λ::Float64
end

support(::PCPrecision) = RealInterval(0.0, Inf)

function logpdf(d::PCPrecision, τ::Real)
    τ <= 0 && return -Inf
    λ = d.λ
    # log π(τ) = log(λ/2) - (3/2)log(τ) - λ / sqrt(τ)
    return log(λ) - log(2) - 1.5 * log(τ) - λ / sqrt(τ)
end

"""
    mode(d::PCPrecision)

Return the mode of the PC prior on precision.

The mode has a closed-form solution: τ_mode = λ²/9, derived by setting
d/dτ [log π(τ)] = 0 and solving for τ.
"""
function mode(d::PCPrecision)
    return d.λ^2 / 9
end

function rand(rng::AbstractRNG, d::PCPrecision)
    # Sample via σ ~ Exp(λ), then τ = 1/σ^2
    σ = rand(rng, Exponential(1 / d.λ))
    return 1 / (σ^2)
end

rand(d::PCPrecision) = rand(Random.GLOBAL_RNG, d)

"""
    pc_prior_precision(U; α=0.05)

Convenience constructor for the induced PC prior on τ from the same calibration:
P(σ > U) = α.
"""
function pc_prior_precision(U; α = 0.05)
    λ = pc_lambda_from_tail(U, α)
    return PCPrecision(λ)
end


struct PrecisionToStdBijector <: Bijectors.Bijector end
struct StdToPrecisionBijector <: Bijectors.Bijector end

# Inverses
Bijectors.inverse(::PrecisionToStdBijector) = StdToPrecisionBijector()
Bijectors.inverse(::StdToPrecisionBijector) = PrecisionToStdBijector()
Base.inv(b::PrecisionToStdBijector) = Bijectors.inverse(b)
Base.inv(b::StdToPrecisionBijector) = Bijectors.inverse(b)

# ---- forward maps -----------------------------------------------------------
# precision -> std: σ = 1 / sqrt(τ)
function Bijectors.transform(::PrecisionToStdBijector, τ::Real)
    τ > 0 || throw(ArgumentError("Precision must be > 0, got $τ"))
    return inv(sqrt(τ))
end
Bijectors.transform(b::PrecisionToStdBijector, τ::AbstractArray) = @. inv(sqrt(τ))

# std -> precision: τ = 1 / σ^2
function Bijectors.transform(::StdToPrecisionBijector, σ::Real)
    σ > 0 || throw(ArgumentError("Std must be > 0, got $σ"))
    return inv(σ^2)
end
Bijectors.transform(b::StdToPrecisionBijector, σ::AbstractArray) = @. inv(σ^2)

# ---- log|det J| -------------------------------------------------------------
# For σ = τ^{-1/2}, dσ/dτ = -(1/2) τ^{-3/2} ⇒ |dσ/dτ| = (1/2) τ^{-3/2}
function Bijectors.logabsdetjac(::PrecisionToStdBijector, τ::Real)
    τ > 0 || return -Inf
    return log(0.5) - 1.5 * log(τ)
end
function Bijectors.logabsdetjac(b::PrecisionToStdBijector, τ::AbstractArray)
    # sum over independent elementwise transforms
    s = zero(eltype(τ))
    @inbounds @simd for t in τ
        s += Bijectors.logabsdetjac(b, t)
    end
    return s
end

# For τ = σ^{-2}, dτ/dσ = -2 σ^{-3} ⇒ |dτ/dσ| = 2 σ^{-3}
function Bijectors.logabsdetjac(::StdToPrecisionBijector, σ::Real)
    σ > 0 || return -Inf
    return log(2) - 3 * log(σ)
end
function Bijectors.logabsdetjac(b::StdToPrecisionBijector, σ::AbstractArray)
    s = zero(eltype(σ))
    @inbounds @simd for v in σ
        s += Bijectors.logabsdetjac(b, v)
    end
    return s
end

function Bijectors.with_logabsdet_jacobian(b::PrecisionToStdBijector, τ)
    return Bijectors.transform(b, τ), Bijectors.logabsdetjac(b, τ)
end

function Bijectors.with_logabsdet_jacobian(b::StdToPrecisionBijector, σ)
    return Bijectors.transform(b, σ), Bijectors.logabsdetjac(b, σ)
end
