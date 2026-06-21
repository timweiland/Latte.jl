using FastGaussQuadrature: gausslegendre
using StatsFuns: normcdf, erfc

# Precompute 20-point Gauss-Legendre nodes/weights on [0,1]
# (transformed from [-1,1]: u = (x+1)/2, du = dx/2)
const _GL_NODES_20, _GL_WEIGHTS_20 = let
    x, w = gausslegendre(20)
    (x .+ 1) ./ 2, w ./ 2
end

"""
    owens_t(h, a)

Compute Owen's T function: T(h, a) = (1/2π) ∫₀ᵃ exp(-h²(1+t²)/2) / (1+t²) dt.

Uses Gauss-Legendre quadrature for |a| ≤ 1 and the reduction identity for |a| > 1.

# References
- Patefield, M. and Tandy, D. (2000). "Fast and Accurate Calculation of Owen's T Function."
  Journal of Statistical Software, 5(5), 1-25.
- Implementation approach adapted from Andrew Gough's work in StatsFuns.jl#99.
"""
function owens_t(h::Real, a::Real)
    # Special cases
    a == 0 && return 0.0
    h == 0 && return atan(a) / (2π)

    # Symmetry: T(h, -a) = -T(h, a), T(-h, a) = T(h, a)
    sign_a = sign(a)
    a = abs(a)
    h = abs(h)

    if a > 1
        # Identity: T(h,a) = ½[Φ(h)+Φ(ah)] - Φ(h)Φ(ah) - T(ah, 1/a)
        Φh = normcdf(h)
        Φah = normcdf(a * h)
        result = 0.5 * (Φh + Φah) - Φh * Φah - _owens_t_quad(a * h, 1 / a)
    elseif a == 1
        invsqrt2 = 1 / sqrt(2.0)
        result = 0.125 * erfc(-h * invsqrt2) * erfc(h * invsqrt2)
    else
        result = _owens_t_quad(h, a)
    end

    return sign_a * result
end

"""
    _owens_t_quad(h, a)

Compute Owen's T for 0 < a < 1, h ≥ 0 via Gauss-Legendre quadrature.
T(h, a) = (a/2π) ∫₀¹ exp(-h²(1+a²u²)/2) / (1+a²u²) du
"""
function _owens_t_quad(h::Real, a::Real)
    h2 = h * h
    a2 = a * a
    s = 0.0
    @inbounds for i in eachindex(_GL_NODES_20)
        u = _GL_NODES_20[i]
        w = _GL_WEIGHTS_20[i]
        a2u2 = a2 * u * u
        s += w * exp(-h2 * (1 + a2u2) / 2) / (1 + a2u2)
    end
    return a * s / (2π)
end
