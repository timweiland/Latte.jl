using LinearAlgebra

"""
    _diagonal_hessian_3pt_5pt(f, x, f0, h)

Compute diagonal Hessian elements using both 3-point and 5-point central difference
stencils at the same step size `h`. Returns `(diag_3pt, diag_5pt)` vectors.

The 3-point stencil: `H[i,i] = (f(x+h*eᵢ) - 2f(x) + f(x-h*eᵢ)) / h²`
The 5-point stencil: `H[i,i] = (-f(x+2h*eᵢ) + 16f(x+h*eᵢ) - 30f(x) + 16f(x-h*eᵢ) - f(x-2h*eᵢ)) / (12h²)`

Both stencils share the `f(x±h*eᵢ)` evaluations, so computing both costs 4 evals
per dimension (same as 5pt alone).
"""
function _diagonal_hessian_3pt_5pt(f, x::AbstractVector{T}, f0::Real, h::Real) where {T}
    n = length(x)
    diag_3pt = zeros(T, n)
    diag_5pt = zeros(T, n)
    xp = copy(x)

    for i in 1:n
        xp .= x
        xp[i] = x[i] + h;   fp1 = f(xp)
        xp[i] = x[i] - h;   fm1 = f(xp)
        xp[i] = x[i] + 2h;  fp2 = f(xp)
        xp[i] = x[i] - 2h;  fm2 = f(xp)
        xp[i] = x[i]

        diag_3pt[i] = (fp1 - 2 * f0 + fm1) / h^2
        diag_5pt[i] = (-fp2 + 16 * fp1 - 30 * f0 + 16 * fm1 - fm2) / (12 * h^2)
    end

    return diag_3pt, diag_5pt
end

"""
    _stencil_disagreement(diag_3pt, diag_5pt)

Compute the maximum relative disagreement between 3-point and 5-point diagonal
Hessian estimates. Returns a scalar measuring the worst-case relative error across
all dimensions.

At the optimal step size, both stencils agree well. At too-small `h` (noise-dominated),
the 5pt stencil diverges due to its 30× noise amplification on f₀ vs 2× for 3pt.
At too-large `h` (truncation-dominated), both diverge from the true value but the
higher-order 5pt stencil diverges differently.
"""
function _stencil_disagreement(diag_3pt::AbstractVector, diag_5pt::AbstractVector)
    max_disagreement = 0.0
    for i in eachindex(diag_3pt, diag_5pt)
        scale = max(abs(diag_3pt[i]), abs(diag_5pt[i]))
        if scale > 0
            disagreement = abs(diag_3pt[i] - diag_5pt[i]) / scale
            max_disagreement = max(max_disagreement, disagreement)
        end
    end
    return max_disagreement
end

"""
    _safe_stencil_error(f, x, f0, h)

Evaluate the stencil disagreement at step size `h`, returning `Inf` if the function
evaluation fails (e.g., due to non-positive-definite matrices in the inner solve).
"""
function _safe_stencil_error(f, x::AbstractVector, f0::Real, h::Real)
    try
        diag_3pt, diag_5pt = _diagonal_hessian_3pt_5pt(f, x, f0, h)
        return _stencil_disagreement(diag_3pt, diag_5pt)
    catch
        return Inf
    end
end

"""
    _refine_step_size(f, x, f0, log_h_lo, log_h_hi; maxiter=10)

Refine the optimal step size within a bracket `[10^log_h_lo, 10^log_h_hi]` using
golden section search in log₁₀(h) space to minimize stencil disagreement.

Returns `(h_opt, min_error)`.
"""
function _refine_step_size(
        f, x::AbstractVector, f0::Real, log_h_lo::Real, log_h_hi::Real;
        maxiter::Int = 10
    )
    # Golden section search in log-space
    φ = (sqrt(5) - 1) / 2  # golden ratio conjugate ≈ 0.618

    a = log_h_lo
    b = log_h_hi

    c = b - φ * (b - a)
    d = a + φ * (b - a)

    fc = _safe_stencil_error(f, x, f0, 10.0^c)
    fd = _safe_stencil_error(f, x, f0, 10.0^d)

    for _ in 1:maxiter
        if fc < fd
            b = d
            d = c
            fd = fc
            c = b - φ * (b - a)
            fc = _safe_stencil_error(f, x, f0, 10.0^c)
        else
            a = c
            c = d
            fc = fd
            d = a + φ * (b - a)
            fd = _safe_stencil_error(f, x, f0, 10.0^d)
        end
    end

    h_opt = 10.0^((a + b) / 2)
    min_error = _safe_stencil_error(f, x, f0, h_opt)

    return h_opt, min_error
end

"""
    _full_hessian_3pt(f, x, f0, h)

Compute the full Hessian using the 3-point central difference stencil (less noise
amplification than 5-point).

Diagonal: `H[i,i] = (f(x+h*eᵢ) - 2f(x) + f(x-h*eᵢ)) / h²`
Off-diagonal: `H[i,j] = (f(x+h*eᵢ+h*eⱼ) - f(x+h*eᵢ-h*eⱼ) - f(x-h*eᵢ+h*eⱼ) + f(x-h*eᵢ-h*eⱼ)) / (4h²)`
"""
function _full_hessian_3pt(f, x::AbstractVector{T}, f0::Real, h::Real) where {T}
    n = length(x)
    H = zeros(T, n, n)
    xp = copy(x)

    # Diagonal: 3-point stencil
    for i in 1:n
        xp .= x
        xp[i] = x[i] + h;  fp1 = f(xp)
        xp[i] = x[i] - h;  fm1 = f(xp)
        xp[i] = x[i]
        H[i, i] = (fp1 - 2 * f0 + fm1) / h^2
    end

    # Off-diagonal: 4-point stencil
    for i in 1:n, j in (i + 1):n
        xp .= x
        xp[i] = x[i] + h; xp[j] = x[j] + h; fpp = f(xp)
        xp[i] = x[i] + h; xp[j] = x[j] - h; fpm = f(xp)
        xp[i] = x[i] - h; xp[j] = x[j] + h; fmp = f(xp)
        xp[i] = x[i] - h; xp[j] = x[j] - h; fmm = f(xp)
        xp[i] = x[i]; xp[j] = x[j]
        H[i, j] = (fpp - fpm - fmp + fmm) / (4 * h^2)
        H[j, i] = H[i, j]
    end

    return H
end

"""
    adaptive_negative_hessian(f, x; h_candidates, max_error, refine, fallback_h)

Compute the negative Hessian of `f` at `x` using an adaptive step size selection
inspired by R-INLA. The algorithm compares 3-point and 5-point finite difference
stencils across candidate step sizes to find the `h` where both agree best,
indicating the noise-truncation sweet spot.

# Algorithm
1. **Coarse search**: Evaluate diagonal-only 3pt and 5pt stencils at log-spaced `h` values.
   Pick the `h` with minimum relative disagreement.
2. **Refinement**: Golden section search in log₁₀(h) space within the bracket around
   the best coarse candidate.
3. **Final Hessian**: Compute the full Hessian (including off-diagonals) at the optimal `h`
   using the 3pt stencil (which has less noise amplification than 5pt).

# Keyword Arguments
- `h_candidates`: Log-spaced candidate step sizes for coarse search (default: 9 values from 1e-4 to 1.0)
- `max_error`: Maximum acceptable stencil disagreement (default: 0.05)
- `refine`: Whether to refine with golden section search (default: true)
- `fallback_h`: Step size to use if adaptive search fails entirely (default: 0.005)

# Returns
- `Matrix{Float64}`: The negative Hessian matrix `-∇²f(x)`
"""
function adaptive_negative_hessian(
        f, x::AbstractVector;
        h_candidates = [1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1, 3.0e-1, 1.0],
        max_error::Real = 0.05,
        refine::Bool = true,
        fallback_h::Real = 0.005,
    )
    f0 = f(x)

    # Phase 1: Coarse search over candidate step sizes
    errors = zeros(length(h_candidates))
    for (k, h) in enumerate(h_candidates)
        try
            diag_3pt, diag_5pt = _diagonal_hessian_3pt_5pt(f, x, f0, h)
            errors[k] = _stencil_disagreement(diag_3pt, diag_5pt)
        catch
            errors[k] = Inf
        end
    end

    best_idx = argmin(errors)
    best_error = errors[best_idx]
    h_opt = h_candidates[best_idx]

    # Phase 2: Refinement via golden section search in log-space
    if refine && length(h_candidates) >= 3
        # Bracket: one step below and above the best candidate in the candidate list
        lo_idx = max(1, best_idx - 1)
        hi_idx = min(length(h_candidates), best_idx + 1)
        log_lo = log10(h_candidates[lo_idx])
        log_hi = log10(h_candidates[hi_idx])

        h_refined, refined_error = _refine_step_size(f, x, f0, log_lo, log_hi)
        if refined_error < best_error
            h_opt = h_refined
            best_error = refined_error
        end
    end

    # Fallback cascade
    if !isfinite(best_error)
        @warn "Adaptive Hessian step size search failed. Falling back to h=$fallback_h."
        h_opt = fallback_h
    elseif best_error > max_error
        @warn "Adaptive Hessian: best stencil disagreement $(round(best_error, digits = 4)) exceeds threshold $max_error. Using h=$(round(h_opt, sigdigits = 3)) (best available)."
    end

    # Phase 3: Full Hessian at optimal step size using 3pt stencil
    return -_full_hessian_3pt(f, x, f0, h_opt)
end
