using Distributions

export symmetric_kld

"""
    symmetric_kld(p::Distribution, q::Distribution; n_points=200, n_sigma=6.0)

Compute the symmetric Kullback-Leibler divergence (SKLD) between two univariate distributions.

SKLD(p, q) = KL(p || q) + KL(q || p)

Uses numerical integration (trapezoidal rule) over a range covering ±`n_sigma` standard
deviations from both distributions' means.

Returns 0.0 if both distributions are identical (same type and parameters).
"""
function symmetric_kld(
        p::ContinuousUnivariateDistribution, q::ContinuousUnivariateDistribution;
        n_points::Int = 200, n_sigma::Float64 = 6.0
    )
    # Compute integration range covering both distributions
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
            # Trapezoidal rule: half-weight at endpoints
            w = (k == 1 || k == n_points) ? 0.5 : 1.0
            kl_pq += w * px * diff
            kl_qp -= w * qx * diff
        end
    end

    return max((kl_pq + kl_qp) * dx, 0.0)
end
