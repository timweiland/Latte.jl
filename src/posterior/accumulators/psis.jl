# Pareto-Smoothed Importance Sampling (Vehtari, Simpson, Gelman, Yao,
# Gabry — 2017/2024). Self-contained: a single public helper that takes
# log importance weights and returns smoothed weights + the GPD shape
# parameter k̂. Only the sample-based CPO path currently consumes this,
# but it's a general primitive — keep it independent of the accumulator
# trait machinery.

using StatsFuns: logsumexp

"""
    _psis_smooth_log_weights(log_w) -> (smoothed_log_w, k_hat)

Pareto-smooth the upper tail of log importance weights `log_w`. Replaces
the top `M ≈ min(n/5, 3√n)` weights with order-statistic quantiles of a
generalized Pareto distribution fitted to that tail. Returns the modified
log-weight vector and the GPD shape parameter `k̂`.

`k̂ ≥ 0.7` indicates the IS estimator is unreliable and the underlying
LOO estimate should be flagged.
"""
function _psis_smooth_log_weights(log_w::AbstractVector{<:Real})
    n = length(log_w)
    smoothed = collect(Float64.(log_w))
    if n < 5
        return (smoothed, Inf)
    end

    M = min(div(n, 5), ceil(Int, 3 * sqrt(n)))
    M = max(M, 5)
    # Sort indices by log_w descending; top M go through GPD smoothing.
    sorted_idx = sortperm(smoothed; rev = true)
    top_idx = @view sorted_idx[1:M]
    # Cutoff: the (M+1)-th largest log weight. Subtract for numerical
    # stability and to express the GPD's support as (0, ∞).
    cutoff = smoothed[sorted_idx[M + 1]]
    excess = sort([exp(smoothed[i] - cutoff) for i in top_idx])
    σ, k = _gpd_fit_zhang_stephens(excess)
    if !isfinite(σ) || !isfinite(k) || k >= 1.0
        # Fit failed or distribution too heavy-tailed; return raw weights
        # with a sentinel k̂ that triggers the failure-score check.
        return (smoothed, isfinite(k) ? k : Inf)
    end
    # Replace top-M log weights with GPD quantiles at (s − 0.5)/M, applied
    # in ascending order so the smallest perturbation lands on the
    # smallest of the top weights.
    asc = reverse(top_idx)
    for s in 1:M
        q = (s - 0.5) / M
        new_val = if abs(k) < 1.0e-10
            σ * (-log1p(-q))
        else
            σ * ((1 - q)^(-k) - 1) / k
        end
        new_val = max(new_val, 1.0e-300)
        smoothed[asc[s]] = cutoff + log(new_val)
    end
    return (smoothed, k)
end

# Zhang–Stephens (2009) profile-likelihood GPD fit. `x` ascending, > 0.
function _gpd_fit_zhang_stephens(x::AbstractVector{<:Real})
    n = length(x)
    n < 5 && return (NaN, NaN)
    prior = 3.0
    M = 30 + floor(Int, sqrt(n))
    x_n = x[end]
    x_q = x[max(1, ceil(Int, n / 4))]
    bs = Vector{Float64}(undef, M)
    for j in 1:M
        bs[j] = 1 / x_n + (1 - sqrt(M / (j - 0.5))) / (prior * x_q)
    end
    ks = Vector{Float64}(undef, M)
    Ls = Vector{Float64}(undef, M)
    for j in 1:M
        b = bs[j]
        s = 0.0
        for xi in x
            s += log1p(-b * xi)
        end
        kj = s / n
        ks[j] = kj
        Ls[j] = n * (log(-b / kj) - kj - 1)
    end
    max_L = maximum(Ls)
    ws = [exp(L - max_L) for L in Ls]
    ws ./= sum(ws)
    b_hat = sum(b * w for (b, w) in zip(bs, ws))
    s = 0.0
    for xi in x
        s += log1p(-b_hat * xi)
    end
    k_hat = s / n
    σ_hat = -k_hat / b_hat
    return (σ_hat, k_hat)
end
