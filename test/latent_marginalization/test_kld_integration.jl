using Test
using Latte
using GaussianMarkovRandomFields
using Distributions
using SparseArrays
using Random

@testset "KLD in MarginalResult" begin

    Random.seed!(42)
    n = 8
    Q_prior = spdiagm(
        0 => fill(2.0, n),
        -1 => fill(-0.8, n - 1),
        1 => fill(-0.8, n - 1),
    )
    prior_gmrf = GMRF(zeros(n), Q_prior)
    obs_lik = ExponentialFamily(Bernoulli)([1, 0, 1, 1, 0, 0, 1, 0])
    ga = gaussian_approximation(prior_gmrf, obs_lik)

    test_indices = [2, 4, 6]

    @testset "GaussianMarginal → KLD all zero" begin
        result = marginalize(ga, obs_lik, 0.0, GaussianMarginal(), test_indices)
        @test length(result.kld_values) == length(test_indices)
        @test all(result.kld_values .== 0.0)
    end

    @testset "SimplifiedLaplace → KLD all positive" begin
        result = marginalize(ga, obs_lik, 0.0, SimplifiedLaplace(), test_indices)
        @test length(result.kld_values) == length(test_indices)
        @test all(result.kld_values .> 0.0)
        @test all(isfinite.(result.kld_values))
    end

    @testset "LaplaceMarginal → KLD all positive" begin
        result = marginalize(
            ga, obs_lik, 0.0, LaplaceMarginal(true), test_indices;
            prior_gmrf = prior_gmrf,
        )
        @test length(result.kld_values) == length(test_indices)
        @test all(result.kld_values .> 0.0)
        @test all(isfinite.(result.kld_values))
    end

    @testset "Empty indices → empty KLD" begin
        result = marginalize(ga, obs_lik, 0.0, SimplifiedLaplace(), Int[])
        @test length(result.kld_values) == 0
    end
end
