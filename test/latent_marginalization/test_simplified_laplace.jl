using Test
using IntegratedNestedLaplace
using IntegratedNestedLaplace: _get_skew_params
using GaussianMarkovRandomFields
using Distributions
using LinearAlgebra
using SparseArrays
using Random
using HCubature

@testset "SimplifiedLaplace" begin

    @testset "_get_skew_params edge cases" begin
        μ_i, σ_i = 2.0, 1.5

        @testset "γ₃ = 0 → shape α = 0 (Normal)" begin
            γ_1 = 0.0
            γ_3 = 0.0
            ξ, ω, a = _get_skew_params(γ_1, γ_3, μ_i, σ_i)

            @test a == 0.0
            @test ω > 0.0
            sn = SkewNormal(ξ, ω, a)
            @test mean(sn) ≈ μ_i atol = 1.0e-10
            @test std(sn) ≈ σ_i atol = 1.0e-10
        end

        @testset "γ₁ nonzero, γ₃ = 0 → shifted Normal" begin
            γ_1 = 0.3
            γ_3 = 0.0
            ξ, ω, a = _get_skew_params(γ_1, γ_3, μ_i, σ_i)

            @test a == 0.0
            sn = SkewNormal(ξ, ω, a)
            # With a=0, mean = ξ, and ξ should incorporate γ_1 shift
            @test mean(sn) ≈ μ_i + γ_1 * σ_i atol = 1.0e-10
        end

        @testset "Positive and negative γ₃" begin
            γ_1 = 0.0
            _, _, a_pos = _get_skew_params(γ_1, 0.5, μ_i, σ_i)
            _, _, a_neg = _get_skew_params(γ_1, -0.5, μ_i, σ_i)

            @test a_pos > 0  # Positive skewness
            @test a_neg < 0  # Negative skewness
            @test a_pos ≈ -a_neg atol = 1.0e-10  # Antisymmetric
        end
    end

    @testset "Gaussian Likelihood - matches GaussianMarginal" begin
        Random.seed!(42)
        n = 6

        σ_prior = 0.5
        ρ = 0.3
        Q_prior = spdiagm(
            -1 => -ρ * ones(n - 1),
            0 => ones(n),
            1 => -ρ * ones(n - 1),
        ) ./ σ_prior^2
        prior_gmrf = GMRF(zeros(n), Q_prior)

        obs_model = ExponentialFamily(Normal)
        x_true = rand(prior_gmrf)
        y = x_true + 0.8 * randn(n)
        obs_lik = obs_model(y; σ = 0.8)
        ga = gaussian_approximation(prior_gmrf, obs_lik)

        test_indices = [1, 3, 5]
        gauss_result = marginalize(ga, obs_lik, 0.0, GaussianMarginal(), test_indices)
        sl_result = marginalize(ga, obs_lik, 0.0, SimplifiedLaplace(), test_indices)

        for (i, idx) in enumerate(test_indices)
            g = gauss_result.marginals[i]
            s = sl_result.marginals[i]

            @test s isa SkewNormal
            @test mean(s) ≈ mean(g) atol = 1.0e-6
            @test var(s) ≈ var(g) atol = 1.0e-6

            # Shape should be (near-)zero for Gaussian obs
            @test s.α ≈ 0.0 atol = 1.0e-10

            # PDF comparison
            for k in -2:0.5:2
                x = mean(g) + k * std(g)
                @test pdf(s, x) ≈ pdf(g, x) rtol = 1.0e-4
            end
        end
    end

    @testset "Non-Gaussian Likelihood - nonzero skewness" begin
        # A correlated prior is needed for nonzero skewness — IID priors have
        # zero cross-correlations so the simplified Laplace correction vanishes.
        Random.seed!(123)
        n = 8

        Q_prior = spdiagm(
            0 => fill(2.0, n),
            -1 => fill(-0.8, n - 1),
            1 => fill(-0.8, n - 1),
        )
        prior_gmrf = GMRF(zeros(n), Q_prior)

        @testset "Bernoulli" begin
            obs_lik = ExponentialFamily(Bernoulli)([1, 0, 1, 1, 0, 0, 1, 0])
            ga = gaussian_approximation(prior_gmrf, obs_lik)

            sl_result = marginalize(ga, obs_lik, 0.0, SimplifiedLaplace(), [2, 4, 6])

            for marginal in sl_result.marginals
                @test marginal isa SkewNormal
                @test abs(marginal.α) > 1.0e-6
                @test isfinite(mean(marginal))
                @test var(marginal) > 0
            end
        end

        @testset "Poisson" begin
            using GaussianMarkovRandomFields: PoissonObservations
            obs_lik = ExponentialFamily(Poisson)(PoissonObservations([3, 0, 5, 1, 2, 0, 4, 1]))
            ga = gaussian_approximation(prior_gmrf, obs_lik)

            sl_result = marginalize(ga, obs_lik, 0.0, SimplifiedLaplace(), [2, 4, 6])

            for marginal in sl_result.marginals
                @test marginal isa SkewNormal
                @test abs(marginal.α) > 1.0e-6
                @test isfinite(mean(marginal))
                @test var(marginal) > 0
            end
        end
    end

    @testset "SimplifiedLaplace vs LaplaceMarginal - SKLD" begin
        Random.seed!(456)
        n = 8

        Q_prior = spdiagm(
            0 => fill(2.0, n),
            -1 => fill(-1.0, n - 1),
            1 => fill(-1.0, n - 1),
        )
        prior_gmrf = GMRF(zeros(n), Q_prior)

        obs_lik = ExponentialFamily(Bernoulli)([1, 0, 1, 0, 1, 0, 1, 0])
        ga = gaussian_approximation(prior_gmrf, obs_lik)

        test_indices = [2, 4, 6]
        sl_result = marginalize(ga, obs_lik, 0.0, SimplifiedLaplace(), test_indices)
        la_result = marginalize(
            ga, obs_lik, 0.0, LaplaceMarginal(true), test_indices;
            prior_gmrf = prior_gmrf,
        )

        for (i, idx) in enumerate(test_indices)
            sl_m = sl_result.marginals[i]
            la_m = la_result.marginals[i]

            # Means and variances should be close
            @test mean(sl_m) ≈ mean(la_m) atol = 0.15
            @test var(sl_m) ≈ var(la_m) rtol = 0.3

            # Symmetric KLD via quadrature
            μ_center = 0.5 * (mean(sl_m) + mean(la_m))
            σ_range = max(std(sl_m), std(la_m))
            lo, hi = μ_center - 6 * σ_range, μ_center + 6 * σ_range

            kl_pq, _ = hcubature(
                x -> begin
                    p = pdf(sl_m, x[1])
                    q = pdf(la_m, x[1])
                    (p > 1.0e-15 && q > 1.0e-15) ? p * log(p / q) : 0.0
                end,
                [lo], [hi], rtol = 1.0e-6,
            )

            kl_qp, _ = hcubature(
                x -> begin
                    p = pdf(sl_m, x[1])
                    q = pdf(la_m, x[1])
                    (p > 1.0e-15 && q > 1.0e-15) ? q * log(q / p) : 0.0
                end,
                [lo], [hi], rtol = 1.0e-6,
            )

            skld = kl_pq + kl_qp
            @test skld < 0.05
        end
    end

    @testset "Edge cases" begin
        n = 4
        Q = spdiagm(0 => fill(1.0, n))
        prior_gmrf = GMRF(zeros(n), Q)
        obs_lik = ExponentialFamily(Bernoulli)([1, 0, 1, 0])
        ga = gaussian_approximation(prior_gmrf, obs_lik)

        # Empty indices
        result = marginalize(ga, obs_lik, 0.0, SimplifiedLaplace(), Int[])
        @test length(result.marginals) == 0

        # All indices
        result_all = marginalize(ga, obs_lik, 0.0, SimplifiedLaplace())
        @test length(result_all.marginals) == n
        @test all(m isa SkewNormal for m in result_all.marginals)
    end
end
