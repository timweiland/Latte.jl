"""
    Range <: ContinuousUnivariateDistribution

PC prior on the Matérn **range** ρ (Fuglstad et al., 2019), for an SPDE field in
spatial dimension `dim`. The base model is ρ → ∞ (an infinitely smooth, perfectly
correlated field), so *small* ranges are penalised. In dimension `d` the density is

    π(ρ) = (d/2) λ ρ^{-d/2-1} exp(-λ ρ^{-d/2}),   ρ > 0,

with CDF `F(ρ) = exp(-λ ρ^{-d/2})`. For `d = 2` this is exactly `1/ρ ~ Exponential(λ)`.
This is the standard penalised-complexity prior on the Matérn range for SPDE spatial models.

# Constructors
    Range(λ; dim=2)            # direct λ parameterization
    Range(ρ0; p=0.5, dim=2)    # calibrate via P(ρ < ρ0) = p
"""
struct Range <: ContinuousUnivariateDistribution
    λ::Float64
    dim::Int

    function Range(λ_or_ρ0::Real; p::Union{Real, Nothing} = nothing, dim::Integer = 2)
        dim >= 1 || throw(ArgumentError("dim must be ≥ 1, got $dim"))
        if p === nothing
            λ_or_ρ0 > 0 || throw(ArgumentError("λ must be positive, got $λ_or_ρ0"))
            return new(Float64(λ_or_ρ0), Int(dim))
        else
            ρ0 = λ_or_ρ0
            ρ0 > 0 || throw(ArgumentError("ρ0 must be positive, got $ρ0"))
            0 < p < 1 || throw(ArgumentError("p must be in (0,1), got $p"))
            # P(ρ < ρ0) = exp(-λ ρ0^{-d/2}) = p  ⇒  λ = -log(p) · ρ0^{d/2}
            λ = -log(p) * ρ0^(dim / 2)
            return new(Float64(λ), Int(dim))
        end
    end
end

Distributions.support(::Range) = RealInterval(0.0, Inf)
Distributions.minimum(::Range) = 0.0
Distributions.maximum(::Range) = Inf

function Distributions.logpdf(d::Range, ρ::Real)
    ρ <= 0 && return -Inf
    h = d.dim / 2
    return log(h) + log(d.λ) - (h + 1) * log(ρ) - d.λ * ρ^(-h)
end

Distributions.cdf(d::Range, ρ::Real) = ρ <= 0 ? 0.0 : exp(-d.λ * ρ^(-d.dim / 2))

"""
    quantile(d::Range, p)

Inverse CDF: solving `exp(-λ ρ^{-d/2}) = p` gives `ρ = (λ / (-log p))^{2/d}`.
"""
function Distributions.quantile(d::Range, p::Real)
    (0 <= p <= 1) || throw(DomainError(p, "quantile requires p ∈ [0, 1]"))
    p == 0 && return 0.0
    p == 1 && return Inf
    return (d.λ / (-log(p)))^(2 / d.dim)
end

"""
    mode(d::Range)

Mode of the PC range prior: `ρ_mode = (λ · (d/2) / (d/2 + 1))^{2/d}`. The mode is
finite and used as the optimiser seed; the mean is intentionally left undefined
(the density has a heavy right tail, so `E[ρ]` diverges for `d ≤ 2`).
"""
function Distributions.mode(d::Range)
    h = d.dim / 2
    return (d.λ * h / (h + 1))^(1 / h)
end

Distributions.median(d::Range) = quantile(d, 0.5)

Base.rand(rng::AbstractRNG, d::Range) = quantile(d, rand(rng))
