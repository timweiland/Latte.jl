using Test
using Latte
using GaussianMarkovRandomFields
using Distributions
using LinearAlgebra
using SparseArrays
using Random
using HCubature

@testset "SplineAugmentedGaussian" begin

    # Construct a SplineAugmentedGaussian via LaplaceMarginal on a Bernoulli problem
    Random.seed!(42)
    n = 6
    Q_prior = spdiagm(
        0 => fill(2.0, n),
        -1 => fill(-1.0, n - 1),
        1 => fill(-1.0, n - 1),
    )
    prior_gmrf = GMRF(zeros(n), Q_prior)

    obs_lik = ExponentialFamily(Bernoulli)([1, 0, 1, 0, 1, 0])
    ga = gaussian_approximation(prior_gmrf, obs_lik)

    result = marginalize(
        ga, obs_lik, 0.0, LaplaceMarginal(true), [1, 3];
        prior_gmrf = prior_gmrf,
    )
    d = result.marginals[1]

    @test d isa SplineAugmentedGaussian

    @testset "logpdf/pdf consistency" begin
        test_points = range(mean(d) - 3 * std(d), mean(d) + 3 * std(d), length = 20)
        for x in test_points
            @test pdf(d, x) ≈ exp(logpdf(d, x)) rtol = 1.0e-10
        end
    end

    @testset "Normalization" begin
        μ_d, σ_d = mean(d), std(d)
        integral, _ = hcubature(
            x -> pdf(d, x[1]),
            [μ_d - 8 * σ_d], [μ_d + 8 * σ_d], rtol = 1.0e-6,
        )
        @test integral ≈ 1.0 atol = 1.0e-3
    end

    @testset "CDF/quantile round-trip" begin
        test_points = range(mean(d) - 2 * std(d), mean(d) + 2 * std(d), length = 10)
        for x in test_points
            p = cdf(d, x)
            @test 0.0 <= p <= 1.0
            x_roundtrip = quantile(d, p)
            @test x_roundtrip ≈ x atol = 0.05 * std(d)
        end

        # Standard probability levels
        for q in [0.025, 0.25, 0.5, 0.75, 0.975]
            x_q = quantile(d, q)
            @test cdf(d, x_q) ≈ q atol = 0.01
        end
    end

    @testset "Sampling consistency" begin
        Random.seed!(789)
        samples = [rand(d) for _ in 1:5000]

        # Empirical CDF at quantiles
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]
            x_q = quantile(d, q)
            empirical_p = count(s -> s <= x_q, samples) / length(samples)
            @test empirical_p ≈ q atol = 0.05
        end

        # Empirical mean and variance
        @test mean(samples) ≈ mean(d) atol = 3 * std(d) / sqrt(5000)
        @test var(samples) ≈ var(d) rtol = 0.15
    end
end
