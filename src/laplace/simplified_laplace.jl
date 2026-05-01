using Distributions
using GaussianMarkovRandomFields
using SparseArrays
using LinearAlgebra

export SimplifiedLaplace

"""
    SimplifiedLaplace <: MarginalApproximation

Simplified Laplace approximation (Rue/Martino/Chopin 2009 § 3.2.3):
third-order Taylor expansion of the Laplace marginal around the joint
mode, yielding skew-normal marginals.

For augmented LGMs (`[η; x_base]` with `η ≈ A·x_base` enforced by a
high-precision penalty), the augmentation coupling leaks into the
γ_3 skew direction for base latents. Pass `augmentation_info` through
`marginalize` to apply a base-coordinate-equivalent correction;
matches R-INLA compact mode for these cases.
"""
struct SimplifiedLaplace <: MarginalApproximation end

"""
    _compute_tr(Σ, v_i, σ_i, dMdxi)

Computes tr((Σ - (1/σ_i)^2 * v_i * v_i') * dMdxi)

This is used in the Simplified Laplace approximation to compute skewness corrections.
"""
function _compute_tr(Σ, v_i, σ_i, dMdxi::Diagonal)
    res = zero(eltype(dMdxi))
    τ_i = 1 / σ_i^2
    for (j, d) in enumerate(diag(dMdxi))
        res += d * (Σ[j, j] - τ_i * v_i[j]^2)
    end
    return res
end

"""
    _compute_tr(Σ, v_i, σ_i, dMdxi::SparseMatrixCSC)

Specialized implementation for sparse matrices.

For sparse dMdxi, we iterate over the sparsity pattern and compute:
    tr((Σ - τ_i * v_i * v_i') * dMdxi)
  = sum_{(i,j) in nz(dMdxi)} dMdxi[i,j] * (Σ[i,j] - τ_i * v_i[i] * v_i[j])

where τ_i = 1/σ_i^2 and Σ is assumed symmetric.
"""
function _compute_tr(Σ, v_i, σ_i, dMdxi::SparseMatrixCSC)
    res = zero(eltype(dMdxi))
    τ_i = 1 / σ_i^2

    rows = rowvals(dMdxi)
    vals = nonzeros(dMdxi)
    n, m = size(dMdxi)

    for col in 1:m
        for idx in nzrange(dMdxi, col)
            row = rows[idx]
            d_val = vals[idx]  # dMdxi[row, col]
            # Compute (Σ - τ_i * v_i * v_i')[row, col]
            # Σ is symmetric, so Σ[row, col] = Σ[col, row]
            sigma_term = Σ[row, col] - τ_i * v_i[row] * v_i[col]
            res += d_val * sigma_term
        end
    end

    return res
end

const R_SCALE = π^(3 / 2) / ((4 - π) * sqrt(2))
const C = 1 - 2 / π

function _get_skew_params(γ_1, γ_3, μ_i, σ_i)
    # Degenerate case: no skewness → Normal(μ + γ₁σ, σ)
    if abs(γ_3) < 1.0e-30
        return (μ_i + γ_1 * σ_i, σ_i, 0.0)
    end

    r = cbrt(R_SCALE * γ_3) # a / ω
    ρ = r^2
    t = ((ρ - 1) + sqrt((1 - ρ)^2 + 4 * C * ρ)) / (2C)
    a = sign(r) * sqrt(t)
    ω = sqrt(t / ρ)
    δ = a / sqrt(1 + a^2)
    ζ = γ_1 - ω * δ * sqrt(2 / π)

    return (μ_i + ζ * σ_i, ω * σ_i, a)
end

function _marginalize_impl(
        ga, obs_lik, log_prior_θ::Real,
        ::SimplifiedLaplace, indices::Vector{Int}, prior_gmrf;
        augmentation_info = nothing,
    )
    μ = mean(ga)
    Σ = selinv_mat(ga)
    σ = sqrt.(max.(diag(Σ), 0.0))

    marginals = SkewNormal{Float64}[]

    # Augmentation: when the caller passes `augmentation_info`, base
    # latents need a surgical correction on `dir` for γ_3 to drop the
    # artificial η-shadow induced by `η ≈ A·x_base`. With `prior_gmrf`
    # we extract A from `Q_aug` and subtract `σ_i · A[:, i]`; without
    # it, we fall back to zeroing the η-block (Gaussian-equivalent on
    # base latents).
    pred_idx, base_lookup, Q_aug, Q_η_scalar =
        _augmentation_inputs(augmentation_info, prior_gmrf)

    # Diagonal-third-derivative fast path: for exp-fam likelihoods with a
    # diagonal log-Hessian (Poisson/Bernoulli/Binomial/Normal), precompute
    # `λ_k = h'''(μ_k)` once per grid point. Then γ_1 and γ_3 reduce to
    # explicit weighted sums over `obs_indices`, with no per-index Diagonal
    # construction and no `M * dir` matvec. Lifts the per-index cost from
    # ~25 μs (matrix path) to ~5 μs (direct sum) on diagonal exp-fams.
    diag_h3 = third_derivative_diagonal(obs_lik, μ)
    σ² = σ .^ 2

    for i in indices
        μ_i = μ[i]
        σ_i = σ[i]

        if σ_i < 1.0e-10
            # Near-zero variance: dividing by σ_i in the skew formulas
            # would NaN. Return a degenerate SkewNormal instead.
            push!(marginals, SkewNormal(μ_i, max(σ_i, 1.0e-30), 0.0))
            continue
        end

        conditional_column = _compute_conditional_column(ga, i)
        a_coeffs = conditional_column ./ (σ * σ_i)

        # γ_1 needs the full conditional path including σ_i at slot i —
        # the σ_i perturbation of x_i itself enters the log|H_M| term
        # whenever the likelihood's curvature depends on x_i directly or
        # via a design matrix (e.g. additive intercept η_j = β + x_j).
        # γ_3 instead uses dir with [i] zeroed and the surgical
        # augmentation correction applied.
        curvature_dir = σ .* a_coeffs
        curvature_dir[i] = σ_i

        dir = σ .* a_coeffs
        dir[i] = 0.0
        if pred_idx !== nothing && base_lookup !== nothing && base_lookup[i]
            if Q_aug !== nothing
                _correct_augmentation_shadow!(dir, pred_idx, Q_aug, i, σ_i, Q_η_scalar)
            else
                dir[pred_idx] .= 0.0
            end
        end

        if diag_h3 !== nothing
            # Direct sum, no matrix allocation. Math:
            # γ_1 = 0.5 · Σ_k h'''(μ_k) · curvature_dir[k] · (Σ_kk − τ_i · cond_col[k]²)
            # γ_3 = Σ_k h'''(μ_k) · dir[k]³
            τ_i = 1 / σ_i^2
            γ_i_1 = 0.0
            γ_i_3 = 0.0
            @inbounds for (j, k) in enumerate(diag_h3.indices)
                λ_k = diag_h3.values[j]
                γ_i_1 += λ_k * curvature_dir[k] *
                    (σ²[k] - τ_i * conditional_column[k]^2)
                γ_i_3 += λ_k * dir[k]^3
            end
            γ_i_1 *= 0.5
        else
            # Fallback for non-diagonal obs_liks (LinearlyTransformedLikelihood,
            # generic AD-backed). Builds the directional-derivative matrix.
            loghess_curv_deriv = loghessian_directional_derivative(μ, curvature_dir, obs_lik)
            loghess_dir_deriv = loghessian_directional_derivative(μ, dir, obs_lik)
            γ_i_1 = 0.5 * _compute_tr(Σ, conditional_column, σ_i, loghess_curv_deriv)
            γ_i_3 = dot(dir, loghess_dir_deriv * dir)
        end

        marginal = SkewNormal(_get_skew_params(γ_i_1, γ_i_3, μ_i, σ_i)...)
        push!(marginals, marginal)
    end

    return marginals
end

# Returns (pred_idx, base_lookup, Q_aug, Q_η_scalar). All `nothing` /
# zero except `pred_idx` and `base_lookup` when only augmentation_info
# is supplied (then we fall back to the η-zero correction); fully
# populated when prior_gmrf is also available (surgical correction).
function _augmentation_inputs(augmentation_info, prior_gmrf)
    augmentation_info === nothing && return (nothing, nothing, nothing, 0.0)
    pred_idx = augmentation_info.linear_predictor_indices
    base_idx = augmentation_info.base_latent_indices
    n_total = isempty(base_idx) ? maximum(pred_idx; init = 0) : maximum(base_idx)
    n_total = max(n_total, isempty(pred_idx) ? 0 : maximum(pred_idx))
    base_lookup = falses(n_total)
    for j in base_idx
        base_lookup[j] = true
    end
    if prior_gmrf === nothing || isempty(pred_idx)
        return (pred_idx, base_lookup, nothing, 0.0)
    end
    Q_aug = GaussianMarkovRandomFields.precision_matrix(prior_gmrf)
    Q_η_scalar = Q_aug[first(pred_idx), first(pred_idx)]
    Q_η_scalar > 0 || return (pred_idx, base_lookup, nothing, 0.0)
    return (pred_idx, base_lookup, Q_aug, Q_η_scalar)
end

# Subtract `σ_i · A[:, i]` from the η-block of `dir`. Equivalent to
# the base-coordinate `(A · dir_base)` formula with `dir_base[i] = 0`.
function _correct_augmentation_shadow!(
        dir, augmented_idx, Q_aug, i::Int, σ_i::Real, Q_η_scalar::Real,
    )
    inv_λ = inv(Q_η_scalar)
    @inbounds for j in augmented_idx
        dir[j] += σ_i * Q_aug[j, i] * inv_λ
    end
    return dir
end
