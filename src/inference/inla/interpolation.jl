using LinearAlgebra

export CCDInterpolant, profile_marginal

"""
    CCDInterpolant

A parametric interpolant for the hyperparameter log-posterior in z-space (the
reparameterized space from Hessian eigendecomposition).

The model is a skewness-corrected Gaussian:

    log p(z) = c - 0.5 * Σ_i z_i² / σ_corr_i(sign(z_i))²

where σ_corr captures asymmetry per dimension and per direction (skewness correction).
When σ_corr = 1 everywhere, this reduces to a standard
Gaussian `log p(z) = c - 0.5 * ||z||²`.

# Fields
- `mode_log_density`: log p(θ*|y) at the mode
- `sigma_corr_plus`: per-dimension σ corrections for positive z
- `sigma_corr_minus`: per-dimension σ corrections for negative z
- `transform`: ReparameterizationTransform (z ↔ θ conversion)
- `inv_hessian`: H⁻¹ for computing conditional modes during profiling
"""
struct CCDInterpolant{T <: ReparameterizationTransform}
    mode_log_density::Float64
    sigma_corr_plus::Vector{Float64}
    sigma_corr_minus::Vector{Float64}
    transform::T
    inv_hessian::Matrix{Float64}
end

"""
    (interp::CCDInterpolant)(z::AbstractVector)

Evaluate the CCD interpolant at z-space point z.

Returns log p(z) = c - 0.5 * Σ_i z_i² / σ_i(sign(z_i))².
"""
function (interp::CCDInterpolant)(z::AbstractVector)
    logp = interp.mode_log_density
    for i in eachindex(z)
        σ = z[i] >= 0 ? interp.sigma_corr_plus[i] : interp.sigma_corr_minus[i]
        logp -= 0.5 * z[i]^2 / σ^2
    end
    return logp
end

"""
    profile_marginal(interp::CCDInterpolant, dim::Int, n_grid::Int, bounds::AbstractMatrix)

Profile the CCD interpolant along θ dimension `dim`.

For each value of θ_dim on a regular grid, sets other dimensions to their
conditional modes (from the inverse Hessian) and evaluates the CCD interpolant.

Returns `(θ_grid, log_profile)` where θ_grid is in working space.

# Arguments
- `dim`: dimension to profile (1-indexed)
- `n_grid`: number of grid points
- `bounds`: [n_dim × 2] matrix of working-space bounds
"""
function profile_marginal(
        interp::CCDInterpolant, dim::Int, n_grid::Int,
        bounds::AbstractMatrix
    )
    transform = interp.transform
    θ_star = transform.θ_star.θ
    Σ = interp.inv_hessian
    n_dim = length(θ_star)

    # Precompute the inverse z-transform: z = Λ^{1/2} * V' * (θ - θ*)
    # From transform: θ = θ* + V * Λ^{-1/2} * z
    # So: z = Λ^{1/2} * V' * (θ - θ*)
    Λ_sqrt = Diagonal(1.0 ./ diag(transform.Λ_inv_sqrt))
    A = Λ_sqrt * transform.V'   # z = A * (θ - θ*)

    # Create grid in θ_dim (working space)
    θ_lo = bounds[dim, 1]
    θ_hi = bounds[dim, 2]
    θ_grid = collect(range(θ_lo, θ_hi; length = n_grid))

    log_profile = Vector{Float64}(undef, n_grid)

    for (idx, θ_k) in enumerate(θ_grid)
        # Compute conditional mode: θ_j = θ*_j + Σ_jk / Σ_kk * (θ_k - θ*_k)
        θ_full = copy(θ_star)
        Δθ_k = θ_k - θ_star[dim]
        for j in 1:n_dim
            if j == dim
                θ_full[j] = θ_k
            else
                θ_full[j] = θ_star[j] + Σ[j, dim] / Σ[dim, dim] * Δθ_k
            end
        end

        # Convert to z-space and evaluate
        z = A * (θ_full - θ_star)
        log_profile[idx] = interp(z)
    end

    return θ_grid, log_profile
end
