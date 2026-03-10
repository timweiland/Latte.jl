using Distributions: SkewNormal, ContinuousUnivariateDistribution
using StatsFuns: normcdf, norminvcdf

# Type piracy: extending Distributions.cdf and Distributions.quantile for SkewNormal.
# SkewNormal in Distributions.jl v0.25 lacks cdf/quantile implementations.
# TODO: PR to Distributions.jl once validated; see StatsFuns.jl#99 for Owen's T.

"""
    cdf(d::SkewNormal, x::Real)

CDF of the skew-normal distribution using Owen's T function.

F(x | ξ, ω, α) = Φ((x-ξ)/ω) - 2·T((x-ξ)/ω, α)
"""
function Distributions.cdf(d::SkewNormal, x::Real)
    z = (x - d.ξ) / d.ω
    return normcdf(z) - 2 * owens_t(z, d.α)
end

"""
    quantile(d::SkewNormal, p::Real)

Quantile function of the skew-normal distribution.

Uses bisection to find a bracket, then Newton's method for fast convergence.
"""
function Distributions.quantile(d::SkewNormal, p::Real)
    0 <= p <= 1 || throw(DomainError(p, "quantile argument must be in [0,1]"))

    # Edge cases
    p == 0 && return -Inf
    p == 1 && return Inf

    # Find bracket [lo, hi] such that cdf(lo) < p < cdf(hi)
    μ = mean(d)
    σ = std(d)
    lo = μ - 6 * σ
    hi = μ + 6 * σ

    while cdf(d, lo) > p
        lo -= 3 * σ
    end
    while cdf(d, hi) < p
        hi += 3 * σ
    end

    # Newton's method within bracket
    x = (lo + hi) / 2  # safe initial guess within bracket
    for _ in 1:50
        fx = cdf(d, x) - p
        abs(fx) < 1.0e-12 && break

        dfx = pdf(d, x)
        if dfx > 1.0e-300
            x_new = x - fx / dfx
            # Stay within bracket
            if lo < x_new < hi
                x = x_new
            else
                # Bisection fallback
                x = (lo + hi) / 2
            end
        else
            x = (lo + hi) / 2
        end

        # Tighten bracket
        if cdf(d, x) < p
            lo = x
        else
            hi = x
        end
    end

    return x
end
