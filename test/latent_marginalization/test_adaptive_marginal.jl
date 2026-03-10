using Test
using IntegratedNestedLaplace
using IntegratedNestedLaplace: SplineAugmentedGaussian
using GaussianMarkovRandomFields
using Distributions
using SparseArrays
using Random

@testset "AdaptiveMarginal" begin

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

    @testset "Default constructor" begin
        am = AdaptiveMarginal()
        @test am.kld_threshold == 0.1
        am2 = AdaptiveMarginal(0.05)
        @test am2.kld_threshold == 0.05
    end

    @testset "Invalid threshold rejected" begin
        @test_throws ArgumentError AdaptiveMarginal(-0.1)
        @test_throws ArgumentError AdaptiveMarginal(NaN)
        @test_throws ArgumentError AdaptiveMarginal(Inf)
    end

    @testset "High threshold → all stay SimplifiedLaplace" begin
        # With a very high threshold, nothing should be upgraded
        result = marginalize(
            ga, obs_lik, 0.0, AdaptiveMarginal(100.0), test_indices;
            prior_gmrf = prior_gmrf,
        )
        @test length(result.marginals) == length(test_indices)
        @test all(m isa SkewNormal for m in result.marginals)
        @test all(result.kld_values .>= 0.0)
    end

    @testset "Very low threshold → some upgraded to LaplaceMarginal" begin
        # With a very low threshold, variables with any non-Gaussianity should be upgraded
        result = marginalize(
            ga, obs_lik, 0.0, AdaptiveMarginal(1.0e-10), test_indices;
            prior_gmrf = prior_gmrf,
        )
        @test length(result.marginals) == length(test_indices)
        # At least some should be upgraded to SplineAugmentedGaussian
        @test any(m isa SplineAugmentedGaussian for m in result.marginals)
        @test all(result.kld_values .>= 0.0)
    end

    @testset "Lower threshold → more upgrades" begin
        result_strict = marginalize(
            ga, obs_lik, 0.0, AdaptiveMarginal(1.0e-10), test_indices;
            prior_gmrf = prior_gmrf,
        )
        result_loose = marginalize(
            ga, obs_lik, 0.0, AdaptiveMarginal(100.0), test_indices;
            prior_gmrf = prior_gmrf,
        )
        n_upgraded_strict = count(m isa SplineAugmentedGaussian for m in result_strict.marginals)
        n_upgraded_loose = count(m isa SplineAugmentedGaussian for m in result_loose.marginals)
        @test n_upgraded_strict >= n_upgraded_loose
    end

    @testset "SL variables match direct SimplifiedLaplace" begin
        # With high threshold (no upgrades), should match direct SL call
        result_adaptive = marginalize(
            ga, obs_lik, 0.0, AdaptiveMarginal(100.0), test_indices;
            prior_gmrf = prior_gmrf,
        )
        result_sl = marginalize(ga, obs_lik, 0.0, SimplifiedLaplace(), test_indices)

        for i in eachindex(test_indices)
            @test mean(result_adaptive.marginals[i]) ≈ mean(result_sl.marginals[i]) atol = 1.0e-10
            @test std(result_adaptive.marginals[i]) ≈ std(result_sl.marginals[i]) atol = 1.0e-10
        end
    end

    @testset "Upgraded variables match direct LaplaceMarginal" begin
        # With threshold 0 (upgrade everything), should match direct LA call
        result_adaptive = marginalize(
            ga, obs_lik, 0.0, AdaptiveMarginal(0.0), test_indices;
            prior_gmrf = prior_gmrf,
        )
        result_la = marginalize(
            ga, obs_lik, 0.0, LaplaceMarginal(true), test_indices;
            prior_gmrf = prior_gmrf,
        )

        for i in eachindex(test_indices)
            if result_adaptive.marginals[i] isa SplineAugmentedGaussian
                @test mean(result_adaptive.marginals[i]) ≈ mean(result_la.marginals[i]) atol = 1.0e-10
                @test std(result_adaptive.marginals[i]) ≈ std(result_la.marginals[i]) atol = 1.0e-10
            end
        end
    end

    @testset "Empty indices" begin
        result = marginalize(
            ga, obs_lik, 0.0, AdaptiveMarginal(), Int[];
            prior_gmrf = prior_gmrf,
        )
        @test length(result.marginals) == 0
        @test length(result.kld_values) == 0
    end

    @testset "KLD values correct length" begin
        result = marginalize(
            ga, obs_lik, 0.0, AdaptiveMarginal(), 1:n;
            prior_gmrf = prior_gmrf,
        )
        @test length(result.kld_values) == n
        @test all(isfinite.(result.kld_values))
    end
end
