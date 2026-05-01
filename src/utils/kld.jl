using Distributions

export symmetric_kld, quadrature_symmetric_kld, moment_symmetric_kld, gaussian_kl_from_moments

"""
    gaussian_kl_from_moments(μ1, σ1, μ2, σ2) -> Float64

Closed-form KL(N(μ1, σ1²) ‖ N(μ2, σ2²)) — moment-only, no integration:

    KL = log(σ2/σ1) + (σ1² + (μ1-μ2)²) / (2 σ2²) - 1/2

Used by `moment_symmetric_kld` to compute the symmetric variant in O(1) per
pair. Matches R-INLA's `GMRFLib_mkld` (`gmrflib/density.c:1598-1612`) up to
algebraic rearrangement.
"""
function gaussian_kl_from_moments(μ1::Real, σ1::Real, μ2::Real, σ2::Real)
    σ1 > 0 || return 0.0
    σ2 > 0 || return 0.0
    return log(σ2 / σ1) + (σ1^2 + (μ1 - μ2)^2) / (2 * σ2^2) - 0.5
end

"""
    moment_symmetric_kld(p, q) -> Float64
    moment_symmetric_kld(μ1, σ1, μ2, σ2) -> Float64

Symmetric KL between two distributions, computed using only their first
two moments — i.e. treating both as Gaussians with the same `mean`/`std`:

    SKLD_moment(p, q) = (KL(N_p ‖ N_q) + KL(N_q ‖ N_p)) / 2

Closed-form, O(1) per pair. Matches R-INLA's `GMRFLib_mkld_sym`
(`gmrflib/density.c:1628-1634`), the default fast-mode KLD R-INLA writes
to `symmetric-kld.dat`.

For two distributions with **identical first two moments** (e.g. a
SkewNormal moment-matched to its Gaussian baseline), this returns 0
regardless of higher-order shape differences. Use the quadrature variant
or a shape-specific metric (e.g. `abs(skewness(...))`) when shape
distinguishability is required.
"""
function moment_symmetric_kld(μ1::Real, σ1::Real, μ2::Real, σ2::Real)
    kl_pq = gaussian_kl_from_moments(μ1, σ1, μ2, σ2)
    kl_qp = gaussian_kl_from_moments(μ2, σ2, μ1, σ1)
    return max(0.0, (kl_pq + kl_qp) / 2)
end

function moment_symmetric_kld(
        p::ContinuousUnivariateDistribution, q::ContinuousUnivariateDistribution
    )
    return moment_symmetric_kld(mean(p), std(p), mean(q), std(q))
end

"""
    quadrature_symmetric_kld(p, q; n_points=200, n_sigma=6.0) -> Float64

Numerical (trapezoidal) symmetric KLD between two univariate distributions
covering ±`n_sigma` standard deviations from each. O(n_points) `logpdf` /
`pdf` evaluations.

Use when shape differences beyond the first two moments matter — for
example to distinguish a SkewNormal from a Gaussian with the same mean
and standard deviation. For pure Gaussian-vs-Gaussian comparison
[`moment_symmetric_kld`](@ref) is closed-form and faster.
"""
function quadrature_symmetric_kld(
        p::ContinuousUnivariateDistribution, q::ContinuousUnivariateDistribution;
        n_points::Int = 200, n_sigma::Float64 = 6.0
    )
    lo = min(mean(p) - n_sigma * std(p), mean(q) - n_sigma * std(q))
    hi = max(mean(p) + n_sigma * std(p), mean(q) + n_sigma * std(q))

    xs = range(lo, hi; length = n_points)
    dx = step(xs)

    kl_pq = 0.0
    kl_qp = 0.0

    for (k, x) in enumerate(xs)
        lp = logpdf(p, x)
        lq = logpdf(q, x)
        px = exp(lp)
        qx = exp(lq)

        if px > 1.0e-300 && qx > 1.0e-300
            diff = lp - lq
            w = (k == 1 || k == n_points) ? 0.5 : 1.0
            kl_pq += w * px * diff
            kl_qp -= w * qx * diff
        end
    end

    return max((kl_pq + kl_qp) * dx, 0.0)
end

"""
    symmetric_kld(p, q; n_points=200, n_sigma=6.0) -> Float64

Backward-compatible alias of [`quadrature_symmetric_kld`](@ref). Existing
callers are unchanged. New code should pick explicitly between
[`moment_symmetric_kld`](@ref) (closed-form moment-based, fast) and
`quadrature_symmetric_kld` (integration, captures higher-order shape).
"""
function symmetric_kld(
        p::ContinuousUnivariateDistribution, q::ContinuousUnivariateDistribution;
        n_points::Int = 200, n_sigma::Float64 = 6.0
    )
    return quadrature_symmetric_kld(p, q; n_points = n_points, n_sigma = n_sigma)
end
