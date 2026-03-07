using Distributions
using GaussianMarkovRandomFields
using SparseArrays
using LinearAlgebra

export SimplifiedLaplace

"""
    SimplifiedLaplace <: MarginalApproximation

Simplified Laplace approximation: second-order Taylor expansion around the mode.
Results in skew-normal distributions.
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

"""
    _marginalize_impl(ga, obs_lik, log_prior_θ, ::SimplifiedLaplace, indices, prior_gmrf)

Implementation for Simplified Laplace approximation (second-order Taylor).
Results in skew-normal distributions.
"""
function _marginalize_impl(
        ga, obs_lik, log_prior_θ::Real,
        ::SimplifiedLaplace, indices::Vector{Int}, prior_gmrf
    )
    # obs_lik and prior_gmrf are not used in this simplified implementation
    μ = mean(ga)
    Σ = selinv_mat(ga)
    σ = sqrt.(diag(Σ))

    marginals = SkewNormal{Float64}[]

    n = length(μ)

    for i in indices
        # Get base Gaussian parameters
        μ_i = μ[i]
        σ_i = σ[i]

        conditional_column = _compute_conditional_column(ga, i)
        a_coeffs = conditional_column ./ (σ * σ_i)

        dir = σ .* a_coeffs
        dir[i] = 0.0
        loghess_dir_deriv = loghessian_directional_derivative(μ, dir, obs_lik)

        γ_i_1 = 0.5 * _compute_tr(Σ, conditional_column, σ_i, loghess_dir_deriv)
        γ_i_3 = dot(dir, loghess_dir_deriv * dir)

        marginal = SkewNormal(_get_skew_params(γ_i_1, γ_i_3, μ_i, σ_i)...)
        push!(marginals, marginal)
    end

    return marginals
end
