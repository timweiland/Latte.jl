using LinearAlgebra: eigen, Symmetric
using Roots: find_zero, Bisection

"""
    BYMProportion <: ContinuousUnivariateDistribution

PC prior for the mixing proportion φ ∈ (0,1) in a BYM2 model, following
Riebler et al. (2016).

The distance from the base model (φ=0, pure unstructured noise) is computed from
the KLD using the eigenvalues of the scaled ICAR precision matrix Q*:

    d(φ) = √( φ·∑(1/γₖ - 1) - ∑ log(1 + φ·(1/γₖ - 1)) )

where γₖ are the non-zero eigenvalues of Q*.

Calibrated via P(φ > U) = α.

# Constructor
    BYMProportion(Q_scaled, U; α=0.05)

# Reference
Riebler et al. (2016). "An intuitive Bayesian spatial model for disease mapping
that accounts for scaling." *Statistical Methods in Medical Research*, 25(4), 1145–1165.
"""
struct BYMProportion <: ContinuousUnivariateDistribution
    λ::Float64
    gamma_inv_m1::Vector{Float64}  # 1/γₖ - 1 for non-zero eigenvalues
    sum_gamma_inv_m1::Float64      # precomputed ∑(1/γₖ - 1)
end

# --- Distance computation ---------------------------------------------------

function _bym_distance(φ::Real, gamma_inv_m1::Vector{Float64}, s1::Float64)
    s2 = zero(Float64)
    @inbounds for g in gamma_inv_m1
        s2 += log1p(φ * g)
    end
    return sqrt(max(φ * s1 - s2, 0.0))
end

function _bym_distance_and_deriv(φ::Real, gamma_inv_m1::Vector{Float64}, s1::Float64)
    s2 = zero(Float64)
    ds = zero(Float64)
    @inbounds for g in gamma_inv_m1
        s2 += log1p(φ * g)
        ds += g / (1 + φ * g)
    end
    val = φ * s1 - s2
    d_φ = sqrt(max(val, 0.0))
    d_φ < 1.0e-30 && return d_φ, zero(Float64)
    dd_dφ = (s1 - ds) / (2 * d_φ)
    return d_φ, dd_dφ
end

# --- Constructor -------------------------------------------------------------

"""
    BYMProportion(Q_scaled::AbstractMatrix, U; α=0.05)

Construct the PC prior from the scaled ICAR precision matrix Q*.

The matrix should be the structure matrix scaled so that the geometric mean of
the marginal variances equals 1 (i.e., as produced by `BesagModel` with variance
normalization in GaussianMarkovRandomFields.jl).
"""
function BYMProportion(Q_scaled::AbstractMatrix, U::Real; α::Real = 0.05)
    0 < U < 1 || throw(ArgumentError("U must be in (0,1), got $U"))
    0 < α < 1 || throw(ArgumentError("α must be in (0,1), got $α"))

    eigs = eigen(Symmetric(Matrix{Float64}(Q_scaled))).values
    tol = length(eigs) * eps(maximum(abs, eigs))
    gamma_inv_m1 = [1 / γ - 1 for γ in eigs if γ > tol]
    isempty(gamma_inv_m1) && throw(ArgumentError("Q_scaled has no positive eigenvalues"))

    s1 = sum(gamma_inv_m1)
    d_U = _bym_distance(U, gamma_inv_m1, s1)
    d_U > 0 || throw(ArgumentError("Distance at U=$U is zero; check Q_scaled"))
    λ = lambda_from_tail(d_U, α)
    return BYMProportion(λ, gamma_inv_m1, s1)
end

# --- Distribution interface ---------------------------------------------------

Distributions.support(::BYMProportion) = RealInterval(0.0, 1.0)

function Distributions.logpdf(bym::BYMProportion, φ::Real)
    (0 < φ < 1) || return -Inf
    d_φ, dd_dφ = _bym_distance_and_deriv(φ, bym.gamma_inv_m1, bym.sum_gamma_inv_m1)
    d_φ > 0 || return -Inf
    # π(φ) = λ·exp(-λ·d(φ))·|dd/dφ|
    return log(bym.λ) - bym.λ * d_φ + log(dd_dφ)
end

function Base.rand(rng::AbstractRNG, bym::BYMProportion)
    dist = rand(rng, Exponential(1 / bym.λ))
    dist == 0 && return 0.0
    # d(φ) may be bounded as φ→1 for some graphs; clamp if needed
    upper = 1.0 - 1.0e-15
    d_max = _bym_distance(upper, bym.gamma_inv_m1, bym.sum_gamma_inv_m1)
    dist >= d_max && return upper
    φ = find_zero(
        φ -> _bym_distance(φ, bym.gamma_inv_m1, bym.sum_gamma_inv_m1) - dist,
        (0.0, upper),
        Bisection();
        atol = 1.0e-12,
    )
    return φ
end
