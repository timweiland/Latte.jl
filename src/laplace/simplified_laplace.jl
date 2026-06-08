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

For augmented LGMs (`[η; x_base]` with `η ≈ A·x_base`) a base latent's
skew is carried by its conditional coupling to the linear predictors and
is captured directly by the γ_3 directional derivative `Σ[:,i]/σ_i` — no
separate augmented-coordinate correction is needed.
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
    Σ = selected_covariance(ga)
    σ = sqrt.(max.(diag(Σ), 0.0))

    marginals = SkewNormal{Float64}[]

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

        if augmentation_info !== nothing && i in augmentation_info.linear_predictor_indices
            # η linear-predictor node: emit the Gaussian marginal and skip the
            # per-node skew solve. The predictor aggregates many latent terms so
            # it is near-Gaussian (its skew correction is negligible), and these
            # marginals only feed observation_marginals/fitted values — never the
            # named latent components or the accumulators.
            push!(marginals, SkewNormal(μ_i, max(σ_i, 1.0e-30), 0.0))
            continue
        end

        if σ_i < 1.0e-10
            # Near-zero variance: dividing by σ_i in the skew formulas
            # would NaN. Return a degenerate SkewNormal instead.
            push!(marginals, SkewNormal(μ_i, max(σ_i, 1.0e-30), 0.0))
            continue
        end

        cond_col = conditional_column(ga, i)

        # The directional vectors are curvature_dir[k] = dir[k] = cond_col[k]/σ_i,
        # with curvature_dir[i] overridden to σ_i and dir[i] to 0. γ_1 needs the
        # σ_i perturbation of x_i itself (it enters log|H_M| whenever the
        # likelihood's curvature depends on x_i directly or via a design matrix,
        # e.g. additive intercept η_j = β + x_j); γ_3 uses dir with [i] zeroed.
        if diag_h3 !== nothing
            # Direct sum, no per-node vector allocation. The γ sums only read
            # curvature_dir/dir at the diagonal obs indices, so fold their
            # closed form (cond_col[k]/σ_i, with the [i] overrides) into the loop:
            # γ_1 = 0.5 · Σ_k h'''(μ_k) · curvature_dir[k] · (Σ_kk − τ_i · cond_col[k]²)
            # γ_3 = Σ_k h'''(μ_k) · dir[k]³
            τ_i = 1 / σ_i^2
            inv_σ_i = 1 / σ_i
            γ_i_1 = 0.0
            γ_i_3 = 0.0
            @inbounds for (j, k) in enumerate(diag_h3.indices)
                λ_k = diag_h3.values[j]
                c_k = cond_col[k]
                cdir = k == i ? σ_i : c_k * inv_σ_i
                d_k = k == i ? 0.0 : c_k * inv_σ_i
                γ_i_1 += λ_k * cdir * (σ²[k] - τ_i * c_k^2)
                γ_i_3 += λ_k * d_k^3
            end
            γ_i_1 *= 0.5
        else
            a_coeffs = cond_col ./ (σ * σ_i)
            curvature_dir = σ .* a_coeffs
            curvature_dir[i] = σ_i
            dir = σ .* a_coeffs
            dir[i] = 0.0
            # Fallback for non-diagonal obs_liks (LinearlyTransformedLikelihood,
            # generic AD-backed). Builds the directional-derivative matrix.
            #
            # Force dense Vectors before AD: `conditional_column` can
            # return a SparseVector (workspace solves over WorkspaceGMRFs go
            # through CHOLMOD which yields a dense vector, but constraint
            # paths and other GMRF backends can preserve sparsity). When the
            # direction is sparse, ForwardDiff materialises a SparseVector{Dual}
            # at execution. AutoDiffLikelihood's hessian prep cache keys only on
            # `eltype(x)`, so a Vector{Dual} prep gets reused for the SparseVector
            # call site and DI raises a PreparationMismatchError. The fix is
            # cheap — copy to a dense Vector before the AD pass, cost O(n).
            μ_dense = collect(μ)
            curvature_dir_dense = collect(curvature_dir)
            dir_dense = collect(dir)
            loghess_curv_deriv = loghessian_directional_derivative(μ_dense, curvature_dir_dense, obs_lik)
            loghess_dir_deriv = loghessian_directional_derivative(μ_dense, dir_dense, obs_lik)
            γ_i_1 = 0.5 * _compute_tr(Σ, cond_col, σ_i, loghess_curv_deriv)
            γ_i_3 = dot(dir_dense, loghess_dir_deriv * dir_dense)
        end

        marginal = SkewNormal(_get_skew_params(γ_i_1, γ_i_3, μ_i, σ_i)...)
        push!(marginals, marginal)
    end

    return marginals
end
