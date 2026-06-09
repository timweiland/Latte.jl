using Distributions: Normal
using LinearAlgebra
using SparseArrays
using GaussianMarkovRandomFields
using GaussianMarkovRandomFields:
    PoissonLikelihood, GammaLikelihood, NormalLikelihood, ObservationLikelihood,
    LinearlyTransformedLikelihood, LinearlyTransformedObservationModel,
    linear_predictor_marginals, precision_matrix, pointwise_loglik
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

"""
    default_marginalization(model) -> MarginalApproximation

The latent-marginalization method `inla` uses when none is given. A compact
(`augmentation_info === nothing`) model with a linear-predictor likelihood
(`LinearlyTransformedObservationModel`) gets the low-rank Variational Bayes mean
correction (`VBCMarginal`). Augmented models keep the simplified Laplace skew
correction (`SimplifiedLaplace`), which the augmented path relies on. A compact
model with no resolvable hub set (no named layout — hand-built / DAG-path) falls
back to `GaussianMarginal`, *not* compact `SimplifiedLaplace`: the latter routes
onto the slower, less accurate non-diagonal AD path, and the GA mode is already
accurate for the smooth fields this case typically arises from.
"""
function default_marginalization(model)
    (
        model.augmentation_info === nothing &&
            model.observation_model isa LinearlyTransformedObservationModel
    ) || return SimplifiedLaplace()
    isempty(latent_index_set_for_vbc(model, AutoVBCIndexSet())) || return VBCMarginal()
    @info "VBC default: no resolvable latent hub set (no named layout) — using " *
        "GaussianMarginal. Pass VBCMarginal([indices...]) or SimplifiedLaplace() to override." maxlog = 1
    return GaussianMarginal()
end

# Whether the likelihood family has a non-Gaussian skew the mean correction can
# capture. The Gaussian likelihood's GA mode is already the exact conditional
# mean, so VBC is identically a no-op there.
_is_vbc_correctable(lik::LinearlyTransformedLikelihood) = _is_vbc_correctable(lik.base_likelihood)
_is_vbc_correctable(::NormalLikelihood) = false
_is_vbc_correctable(::ObservationLikelihood) = true

# Resolve a method's `index_set` to a concrete vector of latent indices in the
# bare-kernel path (no model in scope — only `prior_gmrf`). Explicit vectors pass
# through; `AutoVBCIndexSet` needs the model layout (see `latent_index_set_for_vbc`).
resolve_vbc_indices(I::AbstractVector{<:Integer}, prior_gmrf) = collect(Int, I)
resolve_vbc_indices(::AutoVBCIndexSet, prior_gmrf) = throw(
    ArgumentError(
        "AutoVBCIndexSet must be resolved against the model layout via " *
            "latent_index_set_for_vbc(model, …); the bare kernel only accepts an " *
            "explicit index vector."
    )
)

"""
    latent_index_set_for_vbc(model, index_set) -> Vector{Int}

Resolve a `VBCMarginal` `index_set` against the model's latent layout into the
concrete vector of hub indices used by `vbc_correction`. An explicit index vector
passes through unchanged. An `AutoVBCIndexSet` selects every named latent block
whose dimension is ≤ `short_dim` — intercepts, small coefficient blocks, and short
random effects — large structured blocks (SPDE fields, long spline/RW bases) are
otherwise corrected implicitly by propagation through `M`. If *no* block is small
(e.g. a pure Matérn-SPDE field with no fixed effects), it anchors on a spread-out
subset of the largest block instead — the correction still propagates to the whole
block via `M`, so VBC keeps running rather than degrading to a different method.

Returns an empty vector when the model has no named layout (hand-built / DAG-path);
the caller (`default_marginalization`) then picks a non-VBC method.

`model` is anything answering `latent_groups` (a `LatentGaussianModel` or an
`InferenceResult`).
"""
latent_index_set_for_vbc(model, I::AbstractVector{<:Integer}) = collect(Int, I)

function latent_index_set_for_vbc(model, policy::AutoVBCIndexSet)
    groups = latent_groups(model)
    isempty(groups) && return Int[]
    I = Int[]
    for r in values(groups)
        length(r) <= policy.short_dim && append!(I, r)
    end
    if isempty(I)
        # No small/fixed-effect block (pure field): anchor VBC on a spread-out
        # subset of the largest block. The mean correction propagates to the whole
        # block through M = Q_X⁻¹[:,I], so any reasonable hub set works; evenly
        # spaced nodes give well-separated anchors.
        biggest = argmax(length, values(groups))
        k = min(policy.short_dim, length(biggest))
        I = unique(round.(Int, range(first(biggest), last(biggest); length = k)))
    end
    return sort!(I)
end

"""
    _corrected_latent_mean(method, ga, obs_lik, prior_gmrf, model) -> Vector{Float64}

Posterior latent mean for a reconstructed grid point: the VBC-corrected μ* when
`method` is a `VBCMarginal`, otherwise the GA mode `mean(ga)`. Lets consumers that
rebuild the GA per integration point (linear combinations, sampling, predictor
marginals) apply the same mean correction the named-latent marginals use.
"""
_corrected_latent_mean(method, ga, obs_lik, prior_gmrf, model) = collect(mean(ga))
function _corrected_latent_mean(method::VBCMarginal, ga, obs_lik, prior_gmrf, model)
    I = latent_index_set_for_vbc(model, method.index_set)
    return first(vbc_correction(ga, obs_lik, prior_gmrf, I; n_gh = method.n_gh))
end

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

# Gamma, log link (shape/dispersion φ): −loglik(η) = φ(η + y·e^{−η}) + const, so the
# Gaussian expectation closes via the k=−1 log-normal MGF E[e^{−η}] = e^{−m+S²/2}.
# B = φ(1 − y·E[e^{−η}]), C = φ·y·E[e^{−η}] (B + C = φ).
function _vbc_coefficients(lik::GammaLikelihood, η0, S; kwargs...)
    Eneg = @. exp(-η0 + S^2 / 2)
    return (lik.phi .* (1 .- lik.y .* Eneg), lik.phi .* lik.y .* Eneg)
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

    # (1) p propagation columns M = Q_X⁻¹[:, I]. conditional_column dispatches on
    #     the GA's concrete type and applies the WorkspaceGMRF/ConstrainedGMRF
    #     Woodbury constraint correction, so M's columns lie in the constraint
    #     tangent and μ* = μ0 + Mλ stays constraint-consistent (μ0 is already
    #     KKT-projected). The no-lsc form is required for a constrained
    #     WorkspaceGMRF — the lsc form routes to the generic AbstractGMRF solve
    #     and drops the correction — and the workspace already reuses its factor.
    #     `stack` keeps M an m×p matrix even for a single hub (p=1), where
    #     `reduce(hcat, …)` would collapse to a vector and scalarize Hλ.
    M = stack(conditional_column(ga, j) for j in I)   # m×p

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
