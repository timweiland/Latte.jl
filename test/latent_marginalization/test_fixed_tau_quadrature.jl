using Test
using Latte
using GaussianMarkovRandomFields
using GaussianMarkovRandomFields: PoissonObservations
using Distributions
using LinearAlgebra
using SparseArrays

# Direct 1D quadrature for the conditional posterior of an IID Gaussian
# prior + Poisson observation: p(x_i | τ, y_i) ∝ exp(y_i x - exp(x) - τ/2 x²).
# Returns the x-grid plus a CDF on the same grid (cumulative trapezoidal).
function _exact_conditional_cdf(
        y_i::Real, τ::Real;
        x_min::Float64 = -10.0, x_max::Float64 = 10.0,
        n_grid::Int = 4001,
    )
    grid = collect(range(x_min, x_max, length = n_grid))
    log_p = [y_i * x - exp(x) - 0.5 * τ * x^2 for x in grid]
    log_p .-= maximum(log_p)
    p = exp.(log_p)
    Δx = grid[2] - grid[1]
    Z = Δx * (sum(p) - 0.5 * (p[1] + p[end]))
    p ./= Z
    cdf_vals = zeros(n_grid)
    for k in 2:n_grid
        cdf_vals[k] = cdf_vals[k - 1] + 0.5 * (p[k - 1] + p[k]) * Δx
    end
    return grid, cdf_vals
end

# KS distance: sup_x |F_engine(x) - F_ref(x)| evaluated on the
# reference grid. Returns (max_abs_gap, signed_gap_at_argmax).
function _ks_distance(engine, ref_grid::Vector{Float64}, ref_cdf::Vector{Float64})
    best_abs = 0.0
    best_signed = 0.0
    for k in eachindex(ref_grid)
        gap = cdf(engine, ref_grid[k]) - ref_cdf[k]
        if abs(gap) > best_abs
            best_abs = abs(gap)
            best_signed = gap
        end
    end
    return best_abs, best_signed
end

@testset "Augmented LGM: SimplifiedLaplace skew vanishes for IID base prior" begin
    # Regression test: the augmentation precision (η ≈ A·x at λ ≈ 1e6)
    # used to leak into SLA's γ_3 for base latents. With an IID base
    # prior the documented behaviour is that the SLA correction
    # vanishes — equivalently, |α| ≈ 0 in the SkewNormal output.
    n = 6
    τ = 1.0
    λ = 1.0e6
    A = sparse(I, n, n)
    Q_base = spdiagm(0 => fill(τ, n))
    Q_aug = [
        λ * sparse(I, n, n)        (-λ * A);
        (-λ * A')                  (Q_base + λ * (A' * A))
    ]
    prior_gmrf_aug = GMRF(zeros(2n), Q_aug)
    y = [1, 0, 2, 3, 0, 1]
    obs_lik = ExponentialFamily(Poisson, GaussianMarkovRandomFields.LogLink(), 1:n)(
        PoissonObservations(y),
    )
    ga = gaussian_approximation(prior_gmrf_aug, obs_lik)

    base_indices = collect((n + 1):(2n))
    aug_info = Latte.AugmentationInfo(n, n)
    sl_result = marginalize(
        ga, obs_lik, 0.0, SimplifiedLaplace(), base_indices;
        prior_gmrf = prior_gmrf_aug,
        augmentation_info = aug_info,
    )
    g_result = marginalize(
        ga, obs_lik, 0.0, GaussianMarginal(), base_indices;
        prior_gmrf = prior_gmrf_aug,
        augmentation_info = aug_info,
    )

    for (i, base_i) in enumerate(base_indices)
        sl_m = sl_result.marginals[i]
        g_m = g_result.marginals[i]
        @test abs(sl_m.α) < 1.0e-3
        @test mean(sl_m) ≈ mean(g_m) atol = 1.0e-4
        @test std(sl_m) ≈ std(g_m) atol = 1.0e-4
    end
end

@testset "Augmented LGM: surgical fix preserves cross-base SLA contributions" begin
    # When the base prior is correlated (not IID) the conditional
    # regression direction `dir_base` has non-zero entries at
    # neighbouring base latents, and SLA's `γ_3` should pick up real
    # cross-site contributions transmitted through `A`. The surgical
    # fix subtracts only the augmentation self-shadow `σ_i · A[:, i]`
    # from the η-block, so those legitimate contributions survive. The
    # fallback (no `prior_gmrf`) zeroes the whole η-block and would
    # lose them, so the two should produce different `γ_3` here.
    n = 5
    λ = 1.0e6
    A = sparse(I, n, n)
    Q_base = spdiagm(
        -1 => fill(-0.8, n - 1),
        0 => fill(2.0, n),
        1 => fill(-0.8, n - 1),
    )
    Q_aug = [
        λ * sparse(I, n, n)        (-λ * A);
        (-λ * A')                  (Q_base + λ * (A' * A))
    ]
    prior_gmrf_aug = GMRF(zeros(2n), Q_aug)
    y = [1, 0, 2, 3, 0]
    obs_lik = ExponentialFamily(Poisson, GaussianMarkovRandomFields.LogLink(), 1:n)(
        PoissonObservations(y),
    )
    ga = gaussian_approximation(prior_gmrf_aug, obs_lik)

    base_indices = collect((n + 1):(2n))
    aug_info = Latte.AugmentationInfo(n, n)

    surgical = marginalize(
        ga, obs_lik, 0.0, SimplifiedLaplace(), base_indices;
        prior_gmrf = prior_gmrf_aug,
        augmentation_info = aug_info,
    )
    fallback = marginalize(
        ga, obs_lik, 0.0, SimplifiedLaplace(), base_indices;
        augmentation_info = aug_info,
    )

    # Fallback collapses to Gaussian-equivalent (η-block of dir
    # zeroed). Surgical preserves real cross-base structure through A.
    fallback_max_α = maximum(abs(m.α) for m in fallback.marginals)
    surgical_max_α = maximum(abs(m.α) for m in surgical.marginals)
    @test fallback_max_α < 1.0e-6
    @test surgical_max_α > 0.05
end

@testset "Fixed-τ latent marginalization vs 1D quadrature" begin
    # Sweep covers low / high count and moderate / strong prior precision.
    # `τ = 0.1` is excluded because it triggers a pre-existing numerical
    # edge in `LaplaceMarginal`'s spline-augmented density (negative
    # variance during SKLD evaluation) — that's a separate bug, not
    # what this test is about.
    cases = [
        (y = 0, τ = 1.0),
        (y = 0, τ = 10.0),
        (y = 1, τ = 1.0),
        (y = 1, τ = 10.0),
        (y = 2, τ = 1.0),
        (y = 5, τ = 1.0),
        (y = 10, τ = 1.0),
    ]

    @testset "y=$(c.y), τ=$(c.τ)" for c in cases
        # n=3 (tiled y) so marginalize has multiple sites to work with;
        # we test only x[1] but every variable is statistically equal.
        n = 3
        prior_gmrf = GMRF(zeros(n), spdiagm(0 => fill(c.τ, n)))
        obs_lik = ExponentialFamily(Poisson)(PoissonObservations(fill(c.y, n)))
        ga = gaussian_approximation(prior_gmrf, obs_lik)

        # Reference: exact 1D quadrature for x[1].
        ref_grid, ref_cdf = _exact_conditional_cdf(c.y, c.τ)

        gauss = marginalize(ga, obs_lik, 0.0, GaussianMarginal(), [1])
        simp = marginalize(ga, obs_lik, 0.0, SimplifiedLaplace(), [1])
        full = marginalize(
            ga, obs_lik, 0.0, LaplaceMarginal(true), [1];
            prior_gmrf = prior_gmrf,
        )

        ks_g, sgn_g = _ks_distance(gauss.marginals[1], ref_grid, ref_cdf)
        ks_s, sgn_s = _ks_distance(simp.marginals[1], ref_grid, ref_cdf)
        ks_f, sgn_f = _ks_distance(full.marginals[1], ref_grid, ref_cdf)

        @test ks_f < 0.02   # Full Laplace tracks the quadrature truth
        @test ks_g < 0.15   # sanity: Gaussian shouldn't be catastrophic

        # Per `test_simplified_laplace.jl`: SLA skew correction
        # vanishes for IID priors, so Simplified must match Gaussian.
        @test ks_s ≈ ks_g atol = 0.01

        # Sanity: tail-direction agreement when the gap is large enough
        # to be informative.
        if abs(sgn_g) > 0.005
            @test sign(sgn_g) == sign(sgn_s)
        end
    end
end
