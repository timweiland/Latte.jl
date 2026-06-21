import Bijectors

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
