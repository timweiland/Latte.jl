# Generalised Pareto Distribution fit via the Zhang-Stephens estimator.
#
# ATTRIBUTION
# ===========
# This implementation is vendored from ParetoSmooth.jl (Carlos Parada, MIT
# licence, 2021): https://github.com/TuringLang/ParetoSmooth.jl/blob/master/src/GPD.jl
# Minor adaptation for our interface; the algorithm is unchanged.
#
# WHY VENDORED
# ============
# ParetoSmooth.jl's compat bounds pin older versions of dependencies that
# conflict with Latte's DynamicPPL pin. Rather than downgrade, we vendor
# this single ~40-LoC function. When upstream compat loosens, swap in
# `using ParetoSmooth: gpd_fit` and delete this file.

using StatsFuns: logsumexp

"""
    gpd_fit_zhang_stephens(sample; wip=true, min_grid_pts=30, sort_sample=false)
        -> (ξ::Real, σ::Real)

Zhang-Stephens Bayesian-weighted estimator for Generalised Pareto
Distribution parameters. Returns the shape `ξ` (a.k.a. `k̂` in the PSIS
convention — `k̂ > 0` ⇒ heavy tail; `k̂ > 0.7` ⇒ IS unreliable) and the
scale `σ`.

# Arguments
- `sample`: positive exceedances above a threshold (PSIS: top-fraction of
  weights relative to the cutoff).
- `wip`: weakly-informative shrinkage of `ξ` toward 0.5 (Vehtari et al.).
- `min_grid_pts`: base grid size; actual grid is `min_grid_pts + isqrt(n)`.
- `sort_sample`: sort the input first (caller usually pre-sorts).

# Reference
Zhang, J. and Stephens, M. A. (2009). A new and efficient estimation
method for the generalized Pareto distribution. *Technometrics* 51(3).
"""
function gpd_fit_zhang_stephens(
        sample::AbstractVector{T};
        wip::Bool = true, min_grid_pts::Integer = 30, sort_sample::Bool = false,
    ) where {T <: Real}
    sort_sample && (sample = sort(sample; alg = QuickSort))
    len = length(sample)

    grid_size = min_grid_pts + isqrt(len)
    n_0 = 10                                    # weakly informative prior strength
    x_star = inv(3 * sample[(len + 2) ÷ 4])     # 25th-percentile trick
    invmax = inv(sample[len])

    θ_hats = similar(sample, grid_size)
    @. θ_hats = invmax + (1 - sqrt((grid_size + 1) / $(1:grid_size))) * x_star
    ξ_hats = similar(θ_hats)
    for i in eachindex(ξ_hats, θ_hats)
        ξh = zero(eltype(ξ_hats))
        for j in eachindex(sample)
            ξh += log1p(-θ_hats[i] * sample[j])
        end
        ξ_hats[i] = ξh / len
    end

    # Profile log-likelihood at each grid point
    log_like = similar(ξ_hats)
    for i in eachindex(ξ_hats, θ_hats)
        log_like[i] = len * (log(-θ_hats[i] / ξ_hats[i]) - ξ_hats[i] - 1)
    end

    # Posterior weights + weighted θ estimate
    log_norm = logsumexp(log_like)
    weights = exp.(log_like .- log_norm)
    θ = sum(θ_hats .* weights)

    # Derived ξ
    ξ_final = zero(T)
    for i in eachindex(sample)
        ξ_final += log1p(-θ * sample[i])
    end
    ξ_final /= len
    σ = -ξ_final / θ

    # Weakly-informative shrinkage toward 0.5 (Vehtari et al.)
    if wip
        ξ_final = (ξ_final * len + 0.5 * n_0) / (len + n_0)
    end
    return ξ_final, σ
end
