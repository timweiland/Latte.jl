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

function Base.rand(rng::AbstractRNG, d::Precision)
    σ = rand(rng, Exponential(1 / d.λ))
    return 1 / (σ^2)
end
