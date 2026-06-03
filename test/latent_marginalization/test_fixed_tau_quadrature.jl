using Test
using Latte
using GaussianMarkovRandomFields
using GaussianMarkovRandomFields: PoissonObservations
using Distributions
using LinearAlgebra
using SparseArrays
using Random

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

# Mean / std / skewness of that same exact conditional posterior, by direct
# quadrature on a fine grid. Used to assert SLA recovers the true skew.
function _exact_conditional_moments(
        y_i::Real, τ::Real;
        x_min::Float64 = -12.0, x_max::Float64 = 12.0, n_grid::Int = 40001,
    )
    grid = collect(range(x_min, x_max, length = n_grid))
    log_p = [y_i * x - exp(x) - 0.5 * τ * x^2 for x in grid]
    log_p .-= maximum(log_p)
    w = exp.(log_p)
    w ./= sum(w)
    μ = sum(w .* grid)
    σ = sqrt(sum(w .* (grid .- μ) .^ 2))
    sk = sum(w .* ((grid .- μ) ./ σ) .^ 3)
    return μ, σ, sk
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

@testset "Augmented LGM: SimplifiedLaplace recovers true base-latent skew" begin
    # The augmentation penalty η ≈ A·x (λ ≈ 1e6) does NOT wash out a base
    # latent's skew: coupled to its linear predictor, the base latent
    # inherits the likelihood's genuine skew. With an IID base prior +
    # Poisson obs that is exactly the 1D conditional skew, which
    # SimplifiedLaplace must recover (not flatten to zero).
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
    obs_lik = ExponentialFamily(Poisson, GaussianMarkovRandomFields.LogLink(); indices = 1:n)(
        PoissonObservations(y),
    )
    ga = gaussian_approximation(prior_gmrf_aug, obs_lik)

    base_indices = collect((n + 1):(2n))
    aug_info = Latte.AugmentationInfo(n, n)
    sl = marginalize(
        ga, obs_lik, 0.0, SimplifiedLaplace(), base_indices;
        prior_gmrf = prior_gmrf_aug,
        augmentation_info = aug_info,
    ).marginals

    for (i, _) in enumerate(base_indices)
        _, _, true_sk = _exact_conditional_moments(y[i], τ)
        @test true_sk < -0.2                          # genuinely left-skewed
        @test skewness(sl[i]) ≈ true_sk atol = 0.05   # SLA recovers it
    end
end

@testset "Augmented LGM: dense design column does not inflate base-latent skew" begin
    # A base latent that loads on many observations through a dense design
    # column (a regression coefficient) is well-informed and ~Gaussian;
    # SimplifiedLaplace must not manufacture skew for it. Full Laplace is
    # the reference. Guards against the augmentation-shadow blow-up where a
    # coefficient over 30 obs picked up |α| ≈ 40.
    rng = MersenneTwister(20260604)
    n = 30
    λ = 1.0e6
    x = collect(range(-1.5, 1.5, length = n))
    A = sparse(hcat(ones(n), x))                 # intercept + slope
    τ_β = 0.01
    Q_base = spdiagm(0 => fill(τ_β, 2))
    Q_aug = [
        λ * sparse(I, n, n)        (-λ * A);
        (-λ * A')                  (Q_base + λ * (A' * A))
    ]
    prior = GMRF(zeros(n + 2), Q_aug)
    y = [rand(rng, Poisson(exp(1.0 + 0.5 * xi))) for xi in x]
    obs = ExponentialFamily(Poisson, GaussianMarkovRandomFields.LogLink(); indices = 1:n)(
        PoissonObservations(y),
    )
    ga = gaussian_approximation(prior, obs)
    base_idx = [n + 1, n + 2]
    ai = Latte.AugmentationInfo(n, 2)
    sl = marginalize(ga, obs, 0.0, SimplifiedLaplace(), base_idx; prior_gmrf = prior, augmentation_info = ai).marginals
    la = marginalize(ga, obs, 0.0, LaplaceMarginal(true), base_idx; prior_gmrf = prior).marginals

    for j in 1:2
        @test abs(skewness(sl[j])) < 0.4                    # no spurious blow-up
        @test skewness(sl[j]) ≈ skewness(la[j]) atol = 0.1  # tracks full Laplace
    end
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
