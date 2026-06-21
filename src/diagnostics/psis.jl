# Pareto-smoothed importance sampling primitives.
#
# Given a set of log-importance-weights `log_w = log p(x) - log q(x)`, these
# compute the k̂ diagnostic and ESS that tell you whether q is a trustworthy
# importance proposal for p.

"""
    ess_is(log_w::AbstractVector{<:Real}) -> Float64

Effective sample size of a set of importance weights (Kong 1992):
`ESS = (Σw)² / Σw²`. Max value is `length(log_w)` when all weights equal.
"""
function ess_is(log_w::AbstractVector{<:Real})
    m = maximum(log_w)
    w = exp.(log_w .- m)
    return sum(w)^2 / sum(abs2, w)
end

"""
    rel_ess_is(log_w) -> Float64

Relative effective sample size: `ESS / length(log_w) ∈ (0, 1]`. Close to 1
⇒ weights nearly constant (proposal matches target well); close to 0 ⇒ a
few weights dominate (proposal is broken).
"""
rel_ess_is(log_w::AbstractVector{<:Real}) = ess_is(log_w) / length(log_w)

"""
    pareto_k(log_w::AbstractVector{<:Real}; frac_tail = 0.2) -> Float64

Compute the PSIS-k̂ diagnostic via a Zhang-Stephens GPD fit on the
top-`frac_tail` fraction of weights. Interpretation:

- `k̂ < 0.5`   excellent proposal (finite variance IS)
- `k̂ < 0.7`   acceptable (standard threshold)
- `k̂ ≥ 0.7`   unreliable — IS estimates have heavy tails; proposal is a
               poor match for the target.

Returns `NaN` when the tail is too short (< 5 samples).
"""
function pareto_k(log_w::AbstractVector{<:Real}; frac_tail::Real = 0.2)
    n = length(log_w)
    n_tail = max(5, min(Int(floor(frac_tail * n)), Int(floor(3 * sqrt(n)))))
    (n_tail < 5 || n_tail >= n) && return NaN

    sorted = sort(log_w)
    cutoff = sorted[end - n_tail]
    tail_sorted = sorted[(end - n_tail + 1):end]
    x = exp.(tail_sorted .- cutoff) .- 1.0
    x[1] = max(x[1], 0.0)
    # x[1] is the exceedance at the cutoff itself (= 0 by construction);
    # drop it so the GPD fit only sees strictly-positive exceedances.
    x_pos = x[2:end]
    length(x_pos) < 5 && return NaN
    k̂, _ = gpd_fit_zhang_stephens(x_pos)
    return k̂
end

"""
    trust_verdict(rel_ess::Real) -> Symbol

Categorical verdict based on relative ESS — robust across light-tail and
heavy-tail regimes since ESS itself always converges.

- `> 0.5` → `:excellent`
- `> 0.2` → `:acceptable`
- else    → `:unreliable`
"""
function trust_verdict(rel_ess::Real)
    rel_ess > 0.5 && return :excellent
    rel_ess > 0.2 && return :acceptable
    return :unreliable
end
