using Distributions
using LinearAlgebra
using SparseArrays
using GaussianMarkovRandomFields
using FastGaussQuadrature
using DataInterpolations
using HCubature
using LinearSolve

export LaplaceApproximationCache, fit_density_correction_spline, evaluate_corrected_density
export setup_conditional_computation, conditional_gmrf, evaluate_laplace_logpdf

"""
    submatrix(A, idcs)

Extract A[idcs, idcs] while preserving matrix structure (Diagonal, Sparse, etc.).

This function avoids accidental densification when indexing structured matrices.

# Arguments
- `A`: Input matrix (can be Diagonal, Sparse, or dense)
- `idcs`: Row and column indices to extract

# Returns
A matrix of the same structure type as `A`, containing A[idcs, idcs]

# Examples
```julia
# Diagonal stays diagonal
D = Diagonal([1.0, 2.0, 3.0, 4.0])
submatrix(D, [1, 3])  # Returns Diagonal([1.0, 3.0])

# Sparse stays sparse
S = spdiagm(0 => [1.0, 2.0, 3.0])
submatrix(S, [1, 2])  # Returns 2x2 SparseMatrixCSC
```
"""
function submatrix(A::Diagonal, idcs)
    return Diagonal(A.diag[idcs])
end

function submatrix(A::AbstractMatrix, idcs)
    # Generic fallback: keep sparse to avoid densification
    return sparse(A[idcs, idcs])
end

"""
    LaplaceApproximationCache

Cache for efficient evaluation of Laplace approximation marginals.

# Fields
- `base_gmrf`: The base GMRF from Gaussian approximation
- `obs_lik`: Materialized observation likelihood (contains data and hyperparameters)
- `conditioning_index`: Index of the conditioning variable
- `conditional_column`: Precomputed Q^{-1}[:,i]
- `active_set`: Precomputed active set indices
- `μ`: Precomputed mean
- `σ`: Precomputed standard deviations
- `prior_gmrf`: Original prior GMRF (before observations)
- `Q_prior_block`: Precomputed prior precision restricted to the active set —
  reused to build Q_conditional at every spline node
- `conditional_workspace`: Persistent GMRFWorkspace keyed to Q_prior_block's
  sparsity pattern. Reused across all spline-node evaluations for this
  cache, saving the CHOLMOD symbolic + numeric factorization that a fresh
  `GMRF(μ, Q_conditional)` would otherwise pay per call.

This struct caches the expensive computations that are independent of x_i,
allowing efficient evaluation at multiple points.
"""
struct LaplaceApproximationCache{G, O, P, Qb, W}
    base_gmrf::G
    obs_lik::O
    conditioning_index::Int
    conditional_column::Vector{Float64}
    active_set::Vector{Int}
    μ::Vector{Float64}  # Precomputed mean
    σ::Vector{Float64}  # Precomputed standard deviations
    prior_gmrf::P  # Original prior GMRF (complete with mean and precision)
    Q_prior_block::Qb  # Prior precision restricted to active_set (pattern-stable)
    conditional_workspace::W
end

"""
    LaplaceApproximationCache(base_gmrf, obs_lik, conditioning_index, μ, σ, prior_gmrf; threshold=0.001)

Create a cache for Laplace approximation computations.

# Arguments
- `base_gmrf`: The base GMRF from Gaussian approximation
- `obs_lik`: Materialized observation likelihood (contains data and hyperparameters)
- `conditioning_index`: Index of conditioning variable
- `μ`: Precomputed mean vector
- `σ`: Precomputed standard deviation vector
- `prior_gmrf`: Original prior GMRF (before observations)
- `threshold`: Threshold for active set selection
"""
function LaplaceApproximationCache(base_gmrf, obs_lik, conditioning_index::Int, μ::Vector{Float64}, σ::Vector{Float64}, prior_gmrf; threshold = 0.001)
    conditional_column, active_set = setup_conditional_computation(base_gmrf, conditioning_index, μ, σ; threshold = threshold)
    Q_prior_block, conditional_workspace = _build_conditional_workspace(prior_gmrf, obs_lik, μ, active_set)
    return LaplaceApproximationCache(
        base_gmrf, obs_lik, conditioning_index,
        conditional_column, active_set, μ, σ, prior_gmrf,
        Q_prior_block, conditional_workspace
    )
end

"""
    _build_conditional_workspace(prior_gmrf, obs_lik, μ, active_set)

Precompute the prior-precision restriction to `active_set` and allocate a
`GMRFWorkspace` whose sparsity pattern covers `Q_prior_block - H_obs_block`.
This workspace is then reused across every spline-node evaluation of
`evaluate_laplace_logpdf`, avoiding a fresh CHOLMOD factor per call.
"""
function _build_conditional_workspace(prior_gmrf, obs_lik, μ::Vector{Float64}, active_set::Vector{Int})
    Q_prior = precision_matrix(prior_gmrf)
    Q_prior_block = submatrix(Q_prior, active_set)
    # Seed the workspace at μ so any initial factor is well-defined. The
    # pattern is what matters; values get rewritten per spline node.
    H_obs = loghessian(μ, obs_lik)
    H_obs_block = submatrix(H_obs, active_set)
    Q_init = sparse(Q_prior_block - H_obs_block)
    ws = GaussianMarkovRandomFields.GMRFWorkspace(Q_init)
    return Q_prior_block, ws
end

"""
    LaplaceApproximationCache(base_gmrf, obs_lik, conditioning_index, μ, σ, prior_gmrf, lsc; threshold=0.001)

Create a cache using the GMRF's precomputed LinearSolve cache to avoid redundant factorizations.
The `lsc` should be obtained via `linsolve_cache(base_gmrf)`.
"""
function LaplaceApproximationCache(base_gmrf, obs_lik, conditioning_index::Int, μ::Vector{Float64}, σ::Vector{Float64}, prior_gmrf, lsc::LinearSolve.LinearCache; threshold = 0.001)
    conditional_column, active_set = setup_conditional_computation(base_gmrf, conditioning_index, μ, σ, lsc; threshold = threshold)
    Q_prior_block, conditional_workspace = _build_conditional_workspace(prior_gmrf, obs_lik, μ, active_set)
    return LaplaceApproximationCache(
        base_gmrf, obs_lik, conditioning_index,
        conditional_column, active_set, μ, σ, prior_gmrf,
        Q_prior_block, conditional_workspace
    )
end

"""
    LaplaceApproximationCache(base_gmrf, obs_model, conditioning_index, prior_gmrf; threshold=0.001)

Create a cache for Laplace approximation computations (convenience constructor).
"""
function LaplaceApproximationCache(base_gmrf, obs_model, conditioning_index::Int, prior_gmrf; threshold = 0.001)
    μ = mean(base_gmrf)
    σ = std(base_gmrf)
    return LaplaceApproximationCache(base_gmrf, obs_model, conditioning_index, μ, σ, prior_gmrf; threshold = threshold)
end

"""
    _solve_with_cache!(lsc::LinearSolve.LinearCache, e_i::Vector{Float64})

Solve Q * v = e_i using an existing LinearSolve cache, reusing the factorization.
Returns a copy of the solution vector.
"""
function _solve_with_cache!(lsc::LinearSolve.LinearCache, e_i::Vector{Float64})
    lsc.b .= e_i
    sol = solve!(lsc)
    return copy(sol.u)
end

function _compute_conditional_column(base_gmrf::AbstractGMRF, conditioning_index::Int)
    lsc = GaussianMarkovRandomFields.linsolve_cache(base_gmrf)
    return _compute_conditional_column(base_gmrf, conditioning_index, lsc)
end

function _compute_conditional_column(base_gmrf::AbstractGMRF, conditioning_index::Int, lsc::LinearSolve.LinearCache)
    n = length(base_gmrf)

    # Solve Q * v = e_i to get Q^{-1}[:,i] using precomputed factorization
    e_i = zeros(n)
    e_i[conditioning_index] = 1.0
    return _solve_with_cache!(lsc, e_i)
end

# WorkspaceGMRF path: use the workspace's solver (reuses factorization without
# a LinearSolve cache). Constrained case adds a Woodbury correction against the
# workspace's embedded ConstraintInfo.
function _compute_conditional_column(base_gmrf::GaussianMarkovRandomFields.WorkspaceGMRF, conditioning_index::Int)
    GaussianMarkovRandomFields.ensure_loaded!(base_gmrf)
    n = length(base_gmrf)
    e_i = zeros(n)
    e_i[conditioning_index] = 1.0
    base = GaussianMarkovRandomFields.workspace_solve(base_gmrf.workspace, e_i)
    if base_gmrf.constraints === nothing
        return base
    end
    ci = base_gmrf.constraints
    correction = ci.A_tilde_T * (ci.L_c \ (ci.A_tilde_T' * e_i))
    return base - correction
end

function _compute_conditional_column(constrained_gmrf::ConstrainedGMRF, conditioning_index::Int)
    lsc = GaussianMarkovRandomFields.linsolve_cache(constrained_gmrf.base_gmrf)
    return _compute_conditional_column(constrained_gmrf, conditioning_index, lsc)
end

function _compute_conditional_column(constrained_gmrf::ConstrainedGMRF, conditioning_index::Int, lsc::LinearSolve.LinearCache)
    # Account for existing constraints via a Woodbury correction term
    n = length(constrained_gmrf)
    e_i = zeros(n)
    e_i[conditioning_index] = 1.0

    # Q_base⁻¹ eᵢ via the shared linsolve cache
    base = _solve_with_cache!(lsc, e_i)
    # Woodbury correction for the constraint: Σ_constrained = Σ_base - Ã' L_c⁻¹ Ã
    correction = constrained_gmrf.A_tilde_T * (constrained_gmrf.L_c \ (constrained_gmrf.A_tilde_T' * e_i))
    return base - correction
end

"""
    setup_conditional_computation(base_gmrf, conditioning_index, μ, σ; threshold=0.001)

Precompute the conditioning_index-independent components for Laplace approximation.

# Arguments
- `base_gmrf`: The base GMRF from Gaussian approximation (π̃_G)
- `conditioning_index`: Index i of the component x_i we're conditioning on
- `μ`: Precomputed mean vector
- `σ`: Precomputed standard deviation vector
- `threshold`: Threshold for active set selection (default 0.001)

# Returns
- `conditional_column`: Q^{-1}[:,i] - the i-th column of the covariance matrix
- `active_set`: Indices of components with |a_{ij}| > threshold

This function performs the expensive computations that are independent of the
specific value of x_i, allowing efficient evaluation at multiple x_i points.
"""
function setup_conditional_computation(base_gmrf, conditioning_index::Int, μ::Vector{Float64}, σ::Vector{Float64}; threshold = 0.001)
    conditional_column = _compute_conditional_column(base_gmrf, conditioning_index)
    return _setup_active_set(conditional_column, conditioning_index, σ; threshold = threshold)
end

function setup_conditional_computation(base_gmrf, conditioning_index::Int, μ::Vector{Float64}, σ::Vector{Float64}, lsc::LinearSolve.LinearCache; threshold = 0.001)
    conditional_column = _compute_conditional_column(base_gmrf, conditioning_index, lsc)
    return _setup_active_set(conditional_column, conditioning_index, σ; threshold = threshold)
end

function _setup_active_set(conditional_column::Vector{Float64}, conditioning_index::Int, σ::Vector{Float64}; threshold = 0.001)
    # Use precomputed standard deviations
    # Compute influence coefficients a_{ij} = -(Q^{-1})_{ji} / (σ_j * σ_i)
    σ_i = σ[conditioning_index]
    a_coeffs = conditional_column ./ (σ * σ_i)

    # Find active set based on threshold
    active_set = findall(abs.(a_coeffs) .> threshold)

    return conditional_column, active_set
end

"""
    setup_conditional_computation(base_gmrf, conditioning_index; threshold=0.001)

Backward compatibility version that computes μ and σ internally.
"""
function setup_conditional_computation(base_gmrf, conditioning_index::Int; threshold = 0.001)
    μ = mean(base_gmrf)
    σ = std(base_gmrf)
    return setup_conditional_computation(base_gmrf, conditioning_index, μ, σ; threshold = threshold)
end

"""
    conditional_gmrf(base_gmrf, obs_model, conditioning_index, conditional_column,
                    active_set, μ_conditional, θ, y)

Construct the conditional GMRF π̃_GG(x_{R_i} | x_i, θ, y) for the Laplace approximation.

# Arguments
- `base_gmrf`: The base GMRF from Gaussian approximation
- `obs_model`: Observation model for computing Hessians
- `active_set`: Precomputed active set indices
- `μ_conditional`: Conditional mean configuration E[x | x_i]
- `θ`: Hyperparameters
- `y`: Observed data

# Returns
A GMRF representing π̃_GG(x_{active_set} | x_i, θ, y) with proper mean and precision.
"""
function conditional_gmrf(cache::LaplaceApproximationCache, active_set::Vector{Int}, μ_conditional::Vector{Float64})
    # Extract conditional mean for active set
    μ_cond_active = μ_conditional[active_set]

    # Compute observation Hessian at conditional configuration
    H_obs = loghessian(μ_conditional, cache.obs_lik)
    # Use structure-preserving indexing (H_obs is typically Diagonal for exponential family)
    H_obs_block = submatrix(H_obs, active_set)

    # Build conditional precision values: Q_prior_block (cached, pattern-stable)
    # minus H_obs_block. Sparsity pattern matches what the cached workspace
    # was built against, so `update_precision!` inside the WorkspaceGMRF
    # constructor only rewrites nzval — no fresh CHOLMOD factor.
    Q_conditional = sparse(cache.Q_prior_block - H_obs_block)

    # Construct a constrained workspace-backed GMRF directly. This reuses
    # the cache's symbolic factorization across all spline-node evaluations.
    c_idx = only(indexin(cache.conditioning_index, active_set))
    constraint_mat = zeros(1, length(μ_cond_active))
    constraint_mat[1, c_idx] = 1.0
    constraint_vec = [μ_conditional[c_idx]]

    return GaussianMarkovRandomFields.WorkspaceGMRF(
        μ_cond_active, Q_conditional, cache.conditional_workspace,
        constraint_mat, constraint_vec,
    )
end

"""
    evaluate_laplace_logpdf(cache::LaplaceApproximationCache, x_i, log_prior_θ)

Evaluate the Laplace approximation log-density π̃_LA(x_i | θ, y) using cached computations.

This implements the formula:
π̃_LA(x_i | θ, y) ∝ [π(x, θ, y) / π̃_GG(x_{-i} | x_i, θ, y)]|_{x_{-i} = x_{-i}^*(x_i, θ)}

# Arguments
- `cache`: Precomputed cache containing obs_lik (with data and hyperparameters) and all necessary data
- `x_i`: Value to evaluate the marginal at
- `log_prior_θ`: Log-density of hyperparameter prior log π(θ)

# Returns
The log-density value log π̃_LA(x_i | θ, y) (up to normalizing constant).
"""
function evaluate_laplace_logpdf(cache::LaplaceApproximationCache, x_i::Real, log_prior_θ::Real)
    # Extract cached values
    base_gmrf = cache.base_gmrf
    conditioning_index = cache.conditioning_index
    conditional_column = cache.conditional_column
    active_set = cache.active_set

    # Compute conditional configuration once
    μ_prior = cache.μ  # Use cached mean
    σ_i_squared = conditional_column[conditioning_index]
    μ_conditional = μ_prior - conditional_column * (μ_prior[conditioning_index] - x_i) / σ_i_squared

    # Build conditional GMRF using the computed conditional mean
    gmrf_gg = conditional_gmrf(cache, active_set, μ_conditional)

    # Evaluate joint log-density π(x, θ, y) at conditional configuration
    # This is: log π(θ) + log π(x | θ_prior) + log π(y | x, θ)
    # IMPORTANT: Use PRIOR GMRF for latent part, not the Gaussian approximation
    hyperparameter_logpdf = log_prior_θ
    latent_logpdf = logpdf(cache.prior_gmrf, μ_conditional)
    obs_logpdf = loglik(μ_conditional, cache.obs_lik)
    joint_logpdf = hyperparameter_logpdf + latent_logpdf + obs_logpdf

    # Conditional log-density π̃_GG(x_{active_set} | x_i, θ, y)
    # Evaluate at the conditional mean
    μ_cond_active = mean(gmrf_gg)
    conditional_logpdf = logpdf(gmrf_gg, μ_cond_active)

    # Laplace approximation: log π̃_LA = joint - conditional
    return joint_logpdf - conditional_logpdf
end

function _weighted_logsumexp(x, weights)
    x_max = maximum(x)
    return log(sum(weights .* exp.(x .- x_max))) + x_max
end

"""
    _compute_gauss_hermite_normalization(correction_values, w)

Compute normalization constant using original Gauss-Hermite quadrature points (fast but approximate).
"""
function _compute_gauss_hermite_normalization(correction_values, w)
    return _weighted_logsumexp(correction_values, w) - 0.5 * log(π)
end

"""
    _compute_accurate_normalization(spline, μ_i, σ_i)

Compute normalization constant using numerical integration over the fitted spline (accurate but slower).
"""
function _compute_accurate_normalization(spline, μ_i, σ_i)
    # Create Gaussian distribution once for efficiency
    gaussian_dist = Normal(μ_i, σ_i)

    # Define integrand: π̃_G(x) * exp(spline(x))
    function integrand(x_vec)
        x = x_vec[1]
        log_gaussian = logpdf(gaussian_dist, x)
        correction = spline(x)
        return exp(log_gaussian + correction)
    end

    # Integrate over a wider range than the original quadrature points
    integration_bounds = [μ_i - 6 * σ_i, μ_i + 6 * σ_i]
    integral_result, _ = hcubature(integrand, [integration_bounds[1]], [integration_bounds[2]], rtol = 1.0e-8)

    return log(integral_result)
end

"""
    fit_density_correction_spline(cache::LaplaceApproximationCache, θ, y, log_prior_θ;
                                  n_points=9, normalize_exactly=false)

Fit a cubic spline to the difference between Laplace and Gaussian approximation log-densities.

This implements the INLA density correction approach:
log π̃_LA(x_i | θ, y) ≈ log π̃_G(x_i | θ, y) + spline(x_i)

# Arguments
- `cache`: Precomputed cache for Laplace approximation
- `θ`: Hyperparameters
- `y`: Observed data
- `log_prior_θ`: Log-density of hyperparameter prior
- `n_points`: Number of Gauss-Hermite quadrature points (default 9)
- `normalize_exactly`: If true, use numerical integration for exact normalization; if false, use Gauss-Hermite approximation (default false)

# Returns
- `spline`: Interpolation object that can be evaluated at arbitrary points
- `log_norm_const`: Log normalization constant for the corrected density
- `nodes`: The Gauss-Hermite quadrature nodes used
- `correction_values`: The log-density correction values at the nodes

The returned spline can be used as:
log π̃_LA_normalized(x_i) ≈ log π̃_G(x_i) + spline(x_i) - log_norm_const
"""
function fit_density_correction_spline(
        cache::LaplaceApproximationCache, log_prior_θ::Real;
        n_points::Int = 9, normalize_exactly::Bool = false
    )

    # Get Gaussian marginal parameters from cache
    base_gmrf = cache.base_gmrf
    i = cache.conditioning_index
    μ_i = cache.μ[i]
    σ_i = cache.σ[i]

    # Get Gauss-Hermite quadrature nodes and weights
    # Transform from standard Hermite [-∞,∞] to N(μ_i, σ_i²)
    ξ, w = gausshermite(n_points)
    nodes = μ_i .+ σ_i .* sqrt(2) .* ξ  # Transform to N(μ_i, σ_i²) scale

    # Evaluate log-densities at quadrature points
    log_gaussian_values = Float64[]
    log_laplace_values = Float64[]

    for x_i in nodes
        # Gaussian approximation log-density (marginal)
        log_gaussian = logpdf(Normal(μ_i, σ_i), x_i)
        push!(log_gaussian_values, log_gaussian)

        # Laplace approximation log-density
        log_laplace = evaluate_laplace_logpdf(cache, x_i, log_prior_θ)
        push!(log_laplace_values, log_laplace)
    end

    # Compute correction: log π̃_LA - log π̃_G
    correction_values = log_laplace_values .- log_gaussian_values

    # Fit cubic spline to the correction with constant extrapolation
    spline = CubicSpline(correction_values, nodes; extrapolation = ExtrapolationType.Constant)

    # Compute normalization constant using chosen method
    if normalize_exactly
        log_norm_const = _compute_accurate_normalization(spline, μ_i, σ_i)
    else
        log_norm_const = _compute_gauss_hermite_normalization(correction_values, w)
    end

    return spline, log_norm_const, nodes, correction_values
end

"""
    evaluate_corrected_density(cache::LaplaceApproximationCache, spline, x_i, log_norm_const)

Evaluate the normalized spline-corrected density at a point.

Returns log π̃_LA_normalized(x_i) ≈ log π̃_G(x_i) + spline(x_i) - log_norm_const
"""
function evaluate_corrected_density(cache::LaplaceApproximationCache, spline, x_i::Real, log_norm_const::Real)
    i = cache.conditioning_index
    μ_i = cache.μ[i]
    σ_i = cache.σ[i]

    # Gaussian marginal log-density
    log_gaussian = logpdf(Normal(μ_i, σ_i), x_i)

    # Spline correction
    correction = spline(x_i)

    # Normalized log-density
    return log_gaussian + correction - log_norm_const
end
