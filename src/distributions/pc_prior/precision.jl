"""
    Precision <: ContinuousUnivariateDistribution

PC prior on precision τ = 1/σ², induced by an exponential prior on σ.

The density is: π(τ) = (λ/2) τ^{-3/2} exp(-λ/√τ), τ > 0.

# Constructors
    Precision(λ)         # direct λ parameterization
    Precision(U; α=0.05) # calibrate via P(σ > U) = α
"""
struct Precision <: ContinuousUnivariateDistribution
    λ::Float64

    function Precision(λ_or_U::Real; α::Union{Real, Nothing} = nothing)
        if α === nothing
            λ_or_U > 0 || throw(ArgumentError("λ must be positive, got $λ_or_U"))
            return new(Float64(λ_or_U))
        else
            U = λ_or_U
            U > 0 || throw(ArgumentError("U must be positive, got $U"))
            0 < α < 1 || throw(ArgumentError("α must be in (0,1), got $α"))
            λ = lambda_from_tail(U, α)
            return new(Float64(λ))
        end
    end
end

Distributions.support(::Precision) = RealInterval(0.0, Inf)
Distributions.minimum(::Precision) = 0.0
Distributions.maximum(::Precision) = Inf

function Distributions.logpdf(d::Precision, τ::Real)
    τ <= 0 && return -Inf
    λ = d.λ
    return log(λ) - log(2) - 1.5 * log(τ) - λ / sqrt(τ)
end

"""
    mode(d::Precision)

Mode of the PC prior on precision: τ_mode = λ²/9.
"""
function Distributions.mode(d::Precision)
    return d.λ^2 / 9
end

"""
    cdf(d::Precision, τ)

CDF of the PC prior on precision. With σ = τ^{-1/2} ~ Exponential(rate λ) and
τ = 1/σ² a decreasing bijection, F(τ) = P(σ ≥ τ^{-1/2}) = exp(-λ/√τ).
"""
function Distributions.cdf(d::Precision, τ::Real)
    τ <= 0 && return 0.0
    return exp(-d.λ / sqrt(τ))
end

"""
    quantile(d::Precision, p)

Inverse CDF: solving exp(-λ/√τ) = p gives τ = (λ / log p)².
"""
function Distributions.quantile(d::Precision, p::Real)
    (0 <= p <= 1) || throw(DomainError(p, "quantile requires p ∈ [0, 1]"))
    p == 0 && return 0.0
    p == 1 && return Inf
    return (d.λ / log(p))^2
end

# Median = quantile(1/2). Preferred optimiser seed: the τ density is heavily
# right-skewed (median ≫ mode = λ²/9). mean/var/std are intentionally left
# undefined — E[τ] = E[1/σ²] diverges (density tail ∝ τ^{-3/2}), so there is no
# finite value to return.
Distributions.median(d::Precision) = (d.λ / log(0.5))^2

function Base.rand(rng::AbstractRNG, d::Precision)
    σ = rand(rng, Exponential(1 / d.λ))
    return 1 / (σ^2)
end
