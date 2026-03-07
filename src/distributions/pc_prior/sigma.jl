"""
    Sigma(U; α=0.05)

PC prior on standard deviation σ ~ Exp(λ), calibrated via P(σ > U) = α.

Returns a `Distributions.Exponential` distribution (not a custom type).
"""
function Sigma(U::Real; α::Real = 0.05)
    U > 0 || throw(ArgumentError("U must be positive, got $U"))
    0 < α < 1 || throw(ArgumentError("α must be in (0,1), got $α"))
    λ = lambda_from_tail(U, α)
    return Exponential(1 / λ)
end
