using Distributions: Normal
using LinearAlgebra
using SparseArrays
using GaussianMarkovRandomFields
using GaussianMarkovRandomFields:
    PoissonLikelihood, GammaLikelihood, NormalLikelihood, ObservationLikelihood,
    LinearlyTransformedLikelihood, linear_predictor_marginals, linsolve_cache,
    precision_matrix, pointwise_loglik
using FastGaussQuadrature: gausshermite

export VBCMarginal, AutoVBCIndexSet

"""
    AutoVBCIndexSet(; short_dim = 8)

Default policy for the VBC hub set `I`: every fixed-effect block (intercepts,
coefficient blocks — they enter every ηᵢ, so touch all data) plus every
random-effect block of dimension ≤ `short_dim`. Large structured blocks (SPDE
field, long RW splines) are *excluded from* `I` and corrected implicitly by
propagation through `M`. Resolved per model from `latent_groups`/
`latent_components` (model-level resolution lands with the per-θ hook).
"""
struct AutoVBCIndexSet
    short_dim::Int
end
AutoVBCIndexSet(; short_dim::Int = 8) = AutoVBCIndexSet(short_dim)

"""
    VBCMarginal(index_set = AutoVBCIndexSet(); n_gh = 7) <: MarginalApproximation

Compact-mode marginalization with a low-rank Variational Bayes mean Correction
(Van Niekerk & Rue 2021). The per-θ conditional marginals stay Gaussian with the
Gaussian approximation's selected-inverse variances; only the mean μ(θ) is
corrected to μ*(θ) = μ(θ) + M λ*(θ). Requires a compact LGM
(`augmentation_info === nothing`) whose likelihood exposes a design matrix `A`
(a `LinearlyTransformedLikelihood`).

`index_set` is an `AutoVBCIndexSet` or an explicit `AbstractVector{<:Integer}` of
latent indices to use as the correction hubs.
"""
struct VBCMarginal{I} <: MarginalApproximation
    index_set::I
    n_gh::Int
end
VBCMarginal(index_set = AutoVBCIndexSet(); n_gh::Int = 7) =
    VBCMarginal{typeof(index_set)}(index_set, n_gh)

# Whether the likelihood family has a non-Gaussian skew the mean correction can
# capture. The Gaussian likelihood's GA mode is already the exact conditional
# mean, so VBC is identically a no-op there.
_is_vbc_correctable(lik::LinearlyTransformedLikelihood) = _is_vbc_correctable(lik.base_likelihood)
_is_vbc_correctable(::NormalLikelihood) = false
_is_vbc_correctable(::ObservationLikelihood) = true

# Resolve a method's `index_set` to a concrete vector of latent indices.
# Explicit vectors pass through; `AutoVBCIndexSet` needs the model layout and is
# resolved by the per-θ hook (model-level), not here.
resolve_vbc_indices(I::AbstractVector{<:Integer}, prior_gmrf) = collect(Int, I)
resolve_vbc_indices(::AutoVBCIndexSet, prior_gmrf) = throw(
    ArgumentError(
        "AutoVBCIndexSet must be resolved against the model layout; pass an explicit " *
            "index vector to vbc_correction/VBCMarginal until the model-level hook lands."
    )
)

"""
    _vbc_predictor_moments(ga, obs_lik) -> (η0, S, eta_lik)

Predictor mode `η0 = A·μ0 + offset` and marginal std `Sᵢ = √((A Q_X⁻¹ Aᵀ)ᵢᵢ)`,
plus the base η-likelihood (carries the response `eta_lik.y`). Routes through
`linear_predictor_marginals`, which applies the Woodbury constraint correction —
the same path the accumulators trust — so `S` is constraint-correct and no dense
`A Q_X⁻¹ Aᵀ` is ever formed.
"""
function _vbc_predictor_moments(ga, obs_lik)
    μ_η, v_η, eta_lik = linear_predictor_marginals(ga, obs_lik)
    return (μ_η, sqrt.(v_η), eta_lik)
end

# Per-obs linear/quadratic coefficients (B, C) of the expected −loglik about the
# predictor mean, evaluated at the GA mode. (B, C) are the 1st/2nd derivatives
# w.r.t. each ηᵢ's mean; the kernel assembles them into the p×p Newton system.

# Poisson, log link: closed form via the log-normal MGF E[e^η] = exp(m + ½S²).
# −loglik(η) = e^η − yη ⇒ B = E[e^η] − y, C = E[e^η].
function _vbc_coefficients(lik::PoissonLikelihood, η0, S; kwargs...)
    Eexp = @. exp(η0 + S^2 / 2)
    return (Eexp .- lik.y, Eexp)
end

# Gaussian: no-op (GA mode exact). Reached only defensively — vbc_correction
# short-circuits on `_is_vbc_correctable` before assembling coefficients.
_vbc_coefficients(::NormalLikelihood, η0, S; kwargs...) =
    (zeros(length(η0)), zeros(length(η0)))

# General family (Gamma/Bernoulli/Binomial/…): n_gh-node Gauss–Hermite over each
# ηᵢ's Gaussian marginal, differentiated to 2nd order via the Stein identities.
# Vectorized over quadrature nodes: one `pointwise_loglik` call per node, O(n_gh)
# total (not O(n·n_gh) scalar evaluations).
function _vbc_coefficients(eta_lik::ObservationLikelihood, η0, S; n_gh::Int = 7)
    ξ, ω = gausshermite(n_gh)                 # weights for ∫ e^{-x²} f(x) dx
    z = sqrt(2) .* ξ                          # standard-normal nodes
    w = ω ./ sqrt(π)                          # standard-normal weights
    B = zeros(length(η0))
    C = zeros(length(η0))
    for r in eachindex(z)
        ℓ = pointwise_loglik(η0 .+ z[r] .* S, eta_lik)   # length-n loglik at the node
        @. B += (w[r] * z[r] / S) * ℓ
        @. C += (w[r] * (z[r]^2 - 1) / S^2) * ℓ
    end
    return (-B, max.(-C, 0.0))                # derivatives of −loglik; clamp C ≥ 0
end

"""
    vbc_correction(ga, obs_lik, prior_gmrf, I; n_gh = 7)
        -> (μ_star::Vector{Float64}, λ::Vector{Float64})

One per-θ Variational Bayes mean correction. `I` is the p-vector of hub indices.
Returns `μ* = mean(ga) + M λ` where `M = Q_X⁻¹[:, I]` (the p columns of the GA
covariance) and `λ` solves a single p×p Newton system; the variance is untouched.

The correction is the Newton step at λ=0 of the variational objective
`g(λ) = E_{ψ~N(μ0+Mλ, Q_X⁻¹)}[−log π(y|ψ)] + ½(μ0+Mλ)ᵀ Q_π (μ0+Mλ)`. Because μ0
is the GA mode, the pointwise mode condition `Aᵀb0 + Q_π μ0 = 0` makes the data
gradient collapse to the skew-induced part, so a Gaussian likelihood (or any
symmetric one) returns λ = 0.

Cost O(m p²): p back-solves against the *existing* GA factor + an O(n·n_gh) (or
O(n) closed-form) coefficient pass + one p×p solve. Never densifies Q_X⁻¹.
"""
function vbc_correction(ga, obs_lik, prior_gmrf, I::AbstractVector{<:Integer}; n_gh::Int = 7)
    μ0 = collect(mean(ga))
    _is_vbc_correctable(obs_lik) || return (μ0, zeros(length(I)))

    A = obs_lik.design_matrix

    # (1) p propagation columns M = Q_X⁻¹[:, I] — reuse the GA factor (p back-
    #     solves). conditional_column carries the WorkspaceGMRF constraint
    #     (Woodbury) correction, so M's columns lie in the constraint tangent and
    #     μ* = μ0 + Mλ stays constraint-consistent (μ0 is already KKT-projected).
    lsc = linsolve_cache(ga)
    M = reduce(hcat, (conditional_column(ga, j, lsc) for j in I))   # m×p

    # (2) predictor mode η0 and std S — constraint-correct selected-inverse path.
    η0, S, eta_lik = _vbc_predictor_moments(ga, obs_lik)

    # (3) per-obs expected-loglik coefficients (closed form or Gauss–Hermite).
    Bc, Cc = _vbc_coefficients(eta_lik, η0, S; n_gh = n_gh)

    # (4) assemble & solve the p×p Newton system. The prior penalty contributes
    #     M'Q_πM (Hessian) and M'Q_πμ0 (gradient); the data term contributes
    #     (AM)'diag(C)(AM) and (AM)'B. No determinant is computed.
    M_A = A * M                                # n×p
    Qπ = precision_matrix(prior_gmrf)          # m×m sparse
    QπM = Qπ * M                               # m×p
    Hλ = M_A' * (Cc .* M_A) .+ M' * QπM        # p×p, SPD for log-concave lik
    gλ = M_A' * Bc .+ (QπM' * μ0)              # p
    λ = -(Symmetric(Matrix(Hλ)) \ Vector(gλ))
    return (μ0 .+ M * λ, λ)
end

function _marginalize_impl(
        ga, obs_lik, log_prior_θ::Real,
        method::VBCMarginal, indices::AbstractVector{<:Integer}, prior_gmrf;
        augmentation_info = nothing, mean_override = nothing,
    )
    augmentation_info === nothing || throw(
        ArgumentError(
            "VBCMarginal is a compact-mode method but received an augmented model. " *
                "Construct the model with augment=false, or use SimplifiedLaplace()."
        )
    )
    # μ* is computed once per θ in the exploration hook and threaded in as a plain
    # vector via `mean_override`. A bare-GA caller (tests, direct use) gets the
    # fallback: compute the correction here from the method's index set.
    μ = mean_override === nothing ?
        vbc_correction(
            ga, obs_lik, prior_gmrf,
            resolve_vbc_indices(method.index_set, prior_gmrf); n_gh = method.n_gh,
        )[1] : mean_override
    σ = std(ga)                                # real GA selected-inverse diagonal
    return [Normal(μ[i], σ[i]) for i in indices]
end

# VBC keeps the Gaussian shape and only shifts the mean, so the moment-KLD vs the
# GA baseline reduces to the standardized mean shift.
reported_moments(::VBCMarginal, μ_baseline::Real, σ_baseline::Real, marginal) =
    (mean(marginal), σ_baseline)
