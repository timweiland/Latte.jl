# Deterministic quadrature oracle for `toy_iid_poisson`. The model
# factorises into 1D integrals; we use 64-node Gauss-Hermite on the
# conditional density at each τ-grid point and trapezoidal rule on
# the log-τ grid for the outer integral.
#
# Output: per-parameter marginal CDFs + standard quantiles. No mean
# or SD — the PC precision prior's τ-tail makes them undefined.

using FastGaussQuadrature: gausshermite
using Distributions: logpdf

const _GH_NODES, _GH_WEIGHTS = gausshermite(64)
const _LOG_2π = log(2π)
const _SQRT2 = sqrt(2.0)

# ─── Inner: conditional moments and density at fixed τ ────────────────

# Newton on f'(x) = y - exp(x) - τ x = 0. f is concave so this
# converges quadratically.
function _x_mode(y_i::Real, τ::Real; tol::Float64 = 1.0e-12, maxit::Int = 200)
    x = y_i > 0 ? log(y_i) : -2.0
    for _ in 1:maxit
        ex = exp(x)
        dx = -(y_i - ex - τ * x) / (-ex - τ)
        x += dx
        abs(dx) < tol && return x
    end
    return x
end

# Returns (logZ, mean, var, x_star, σ_lap) for p(x_i | τ, y_i).
# `logZ` excludes the y_i! constant — fine, since that factor cancels
# out of p(τ | y) and the final expectations.
function _conditional_moments(y_i::Real, τ::Real)
    x_star = _x_mode(y_i, τ)
    σ_lap = inv(sqrt(exp(x_star) + τ))
    half_log_τ_2π = 0.5 * (log(τ) - _LOG_2π)

    s0 = 0.0; s1 = 0.0; s2 = 0.0
    @inbounds for k in eachindex(_GH_NODES)
        u = _GH_NODES[k]
        x = x_star + σ_lap * _SQRT2 * u
        f_minus_norm = y_i * x - exp(x) - 0.5 * τ * x * x
        w = _GH_WEIGHTS[k] * exp(f_minus_norm + u * u)
        s0 += w
        s1 += w * x
        s2 += w * x * x
    end
    s0 *= σ_lap * _SQRT2
    s1 *= σ_lap * _SQRT2
    s2 *= σ_lap * _SQRT2

    log_Z = half_log_τ_2π + log(s0)
    mean_x = s1 / s0
    var_x = max(s2 / s0 - mean_x^2, 0.0)
    return (
        logZ = log_Z, mean = mean_x, var = var_x,
        x_star = x_star, σ_lap = σ_lap,
    )
end

# Conditional density p(x | τ, y) on a uniform x-grid, normalized via
# trapezoidal rule. Used to build the per-latent marginal CDFs.
function _conditional_density_on_grid(y_i::Real, τ::Real, x_grid::AbstractVector)
    half_log_τ_2π = 0.5 * (log(τ) - _LOG_2π)
    raw = Vector{Float64}(undef, length(x_grid))
    @inbounds for k in eachindex(x_grid)
        x = x_grid[k]
        raw[k] = exp(half_log_τ_2π + y_i * x - exp(x) - 0.5 * τ * x * x)
    end
    Δx = x_grid[2] - x_grid[1]
    norm = Δx * (sum(raw) - 0.5 * (raw[1] + raw[end]))
    return raw ./ norm
end

# ─── Outer: τ posterior and marginal CDFs ─────────────────────────────

"""
    oracle_summary(data; n_grid_τ, n_grid_x, log_τ_min, log_τ_max,
                         x_pad_sds) -> NamedTuple

Compute the deterministic marginal posteriors for the toy scenario.

Returned fields match what `ReferenceSummary` consumes: `parameter_names`,
per-parameter `cdf_grids` and `cdf_values`, and a fixed set of
quantiles (`q025, q25, median, q75, q975, q99`). No means or SDs —
see the file-level docstring.

# Arguments
- `n_grid_τ`: number of points in the working-space (`log τ`) grid.
- `n_grid_x`: number of points in each per-latent x-grid.
- `log_τ_min`, `log_τ_max`: integration domain in `log τ`.
- `x_pad_sds`: how many Laplace SDs to pad each x-grid by on either
  side of its conditional-mode envelope.
"""
function oracle_summary(
        data;
        n_grid_τ::Int = 801,
        n_grid_x::Int = 401,
        log_τ_min::Float64 = -10.0,
        log_τ_max::Float64 = 8.0,
        x_pad_sds::Float64 = 6.0,
    )
    y = data.y
    n = length(y)
    prior = Latte.PCPrior.Precision(1.0; α = 0.01)

    η_grid = collect(range(log_τ_min, log_τ_max, length = n_grid_τ))
    τ_grid = exp.(η_grid)
    Δη = η_grid[2] - η_grid[1]

    # Pass 1: conditional mode + Laplace SD per (τ_j, y_i). Used in
    # Pass 2 to size each latent's x-grid adaptively.
    log_post_eta = Vector{Float64}(undef, n_grid_τ)
    cond_x_star = Matrix{Float64}(undef, n_grid_τ, n)
    cond_σ_lap = Matrix{Float64}(undef, n_grid_τ, n)

    @inbounds for j in 1:n_grid_τ
        τ = τ_grid[j]
        log_lik = 0.0
        for i in 1:n
            cm = _conditional_moments(y[i], τ)
            log_lik += cm.logZ
            cond_x_star[j, i] = cm.x_star
            cond_σ_lap[j, i] = cm.σ_lap
        end
        log_post_eta[j] = logpdf(prior, τ) + η_grid[j] + log_lik
    end

    # Normalised p(η | y) on the grid.
    log_post_eta .-= maximum(log_post_eta)
    w_eta = exp.(log_post_eta)
    Z = _trap(w_eta, Δη)
    w_eta ./= Z

    # ── τ marginal CDF and quantiles ──────────────────────────────────
    # τ = exp(η) is monotone, so F_τ at exp(η_j) equals F_η(η_j).
    cdf_eta = _cumulative_trap(w_eta, Δη)
    τ_cdf_grid = τ_grid
    τ_cdf_vals = cdf_eta

    τ_q = (
        q025 = exp(_inverse_cdf(η_grid, cdf_eta, 0.025)),
        q25 = exp(_inverse_cdf(η_grid, cdf_eta, 0.25)),
        median = exp(_inverse_cdf(η_grid, cdf_eta, 0.5)),
        q75 = exp(_inverse_cdf(η_grid, cdf_eta, 0.75)),
        q975 = exp(_inverse_cdf(η_grid, cdf_eta, 0.975)),
        q99 = exp(_inverse_cdf(η_grid, cdf_eta, 0.99)),
    )

    # ── Per-latent marginal CDFs via 2D quadrature ────────────────────
    # p(x_i | y) ≈ ∑_j w_η[j] · p(x_i | τ_j, y_i). x-grid sized
    # adaptively from the conditional-mode envelope.
    cdf_grids = Vector{Vector{Float64}}(undef, n + 1)
    cdf_values = Vector{Vector{Float64}}(undef, n + 1)
    cdf_grids[1] = collect(τ_cdf_grid)
    cdf_values[1] = collect(τ_cdf_vals)

    x_q025 = Vector{Float64}(undef, n)
    x_q25 = Vector{Float64}(undef, n)
    x_med = Vector{Float64}(undef, n)
    x_q75 = Vector{Float64}(undef, n)
    x_q975 = Vector{Float64}(undef, n)
    x_q99 = Vector{Float64}(undef, n)

    @inbounds for i in 1:n
        # Tight x-grid: ignore τ-grid points with negligible mass.
        keep = w_eta .> 1.0e-10 * maximum(w_eta)
        modes = view(cond_x_star, keep, i)
        sds = view(cond_σ_lap, keep, i)
        x_lo = minimum(modes .- x_pad_sds .* sds)
        x_hi = maximum(modes .+ x_pad_sds .* sds)
        x_grid = collect(range(x_lo, x_hi, length = n_grid_x))
        Δx = x_grid[2] - x_grid[1]

        marginal_pdf = zeros(Float64, n_grid_x)
        for j in 1:n_grid_τ
            w_eta[j] < 1.0e-12 && continue
            pdf_j = _conditional_density_on_grid(y[i], τ_grid[j], x_grid)
            wt = (j == 1 || j == n_grid_τ) ? 0.5 * Δη : Δη
            @. marginal_pdf += wt * w_eta[j] * pdf_j
        end

        # Re-normalise to absorb residual trap error.
        norm_x = Δx * (sum(marginal_pdf) - 0.5 * (marginal_pdf[1] + marginal_pdf[end]))
        marginal_pdf ./= norm_x

        cdf = _cumulative_trap(marginal_pdf, Δx)
        cdf_grids[i + 1] = x_grid
        cdf_values[i + 1] = cdf

        x_q025[i] = _inverse_cdf(x_grid, cdf, 0.025)
        x_q25[i] = _inverse_cdf(x_grid, cdf, 0.25)
        x_med[i] = _inverse_cdf(x_grid, cdf, 0.5)
        x_q75[i] = _inverse_cdf(x_grid, cdf, 0.75)
        x_q975[i] = _inverse_cdf(x_grid, cdf, 0.975)
        x_q99[i] = _inverse_cdf(x_grid, cdf, 0.99)
    end

    parameter_names = String["τ"; ["x[$(i)]" for i in 1:n]]

    return (
        parameter_names = parameter_names,
        cdf_grids = cdf_grids,
        cdf_values = cdf_values,
        q025 = vcat(τ_q.q025, x_q025),
        q25 = vcat(τ_q.q25, x_q25),
        median = vcat(τ_q.median, x_med),
        q75 = vcat(τ_q.q75, x_q75),
        q975 = vcat(τ_q.q975, x_q975),
        q99 = vcat(τ_q.q99, x_q99),
        n_grid_τ = n_grid_τ, n_grid_x = n_grid_x,
        η_range = (log_τ_min, log_τ_max),
    )
end

# ─── Quadrature primitives ────────────────────────────────────────────

# Trapezoidal integral of `vals` over a grid with uniform spacing Δ.
_trap(vals, Δ::Float64) = Δ * (sum(vals) - 0.5 * (first(vals) + last(vals)))

# Cumulative trapezoidal: cdf[k] = ∫_{grid_1}^{grid_k} f dx.
function _cumulative_trap(w, Δ::Float64)
    n = length(w)
    cdf = Vector{Float64}(undef, n)
    cdf[1] = 0.0
    for k in 2:n
        cdf[k] = cdf[k - 1] + 0.5 * (w[k - 1] + w[k]) * Δ
    end
    return cdf
end

# Linearly interpolate `grid[k]` at the cumulative-probability `p`.
function _inverse_cdf(grid, cdf, p::Float64)
    n = length(cdf)
    k = searchsortedfirst(cdf, p)
    k <= 1 && return grid[1]
    k > n && return grid[end]
    t = (p - cdf[k - 1]) / (cdf[k] - cdf[k - 1])
    return grid[k - 1] + t * (grid[k] - grid[k - 1])
end
