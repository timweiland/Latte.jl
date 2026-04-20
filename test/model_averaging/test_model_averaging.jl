using Test
using Latte
using GaussianMarkovRandomFields
using Distributions
using Random

@testset "model_average" begin
    Random.seed!(42)

    # Shared setup: Poisson regression with two different priors
    n = 15
    x_cov = randn(n)
    β_true = [1.0, 0.5]
    η_true = β_true[1] .+ β_true[2] .* x_cov
    y = rand.(Poisson.(exp.(η_true)))

    A = hcat(ones(n), x_cov)
    n_base = 2

    base_latent = IIDModel(n_base)
    base_obs = ExponentialFamily(Poisson)
    obs_model = LinearlyTransformedObservationModel(base_obs, A)

    # Model 1: tight prior
    hp1 = @hyperparams begin
        (τ ~ Exponential(10.0), transform = log, space = natural)
    end
    model1 = LatentGaussianModel(hp1, base_latent, obs_model)

    # Model 2: diffuse prior
    hp2 = @hyperparams begin
        (τ ~ Exponential(0.1), transform = log, space = natural)
    end
    model2 = LatentGaussianModel(hp2, base_latent, obs_model)

    result1 = inla(model1, y; progress = false)
    result2 = inla(model2, y; progress = false)

    @testset "Basic API" begin
        bma = model_average([result1, result2])

        # Returns the right type
        @test bma isa BMAResult

        # Model weights sum to 1
        @test sum(bma.model_weights) ≈ 1.0

        # Weights are non-negative
        @test all(w >= 0 for w in bma.model_weights)

        # Correct number of latent marginals (augmented: n + n_base)
        @test length(bma.latent_marginals) == length(result1.latent_marginals)

        # Each marginal is a valid distribution
        for m in bma.latent_marginals
            @test m isa WeightedMixture
            @test isfinite(mean(m))
            @test var(m) > 0
        end

        # Log marginal likelihoods stored
        @test length(bma.log_marginal_likelihoods) == 2
        @test all(isfinite, bma.log_marginal_likelihoods)
    end

    @testset "Custom prior model weights" begin
        # Unequal prior weights
        bma = model_average([result1, result2]; prior_weights = [0.9, 0.1])
        @test sum(bma.model_weights) ≈ 1.0
        @test all(w >= 0 for w in bma.model_weights)
    end

    @testset "Single model degeneracy" begin
        bma = model_average([result1])
        @test bma.model_weights ≈ [1.0]
        @test length(bma.latent_marginals) == length(result1.latent_marginals)

        # With one model, averaged marginals should be identical to the original
        for (avg, orig) in zip(bma.latent_marginals, result1.latent_marginals)
            @test mean(avg) ≈ mean(orig) atol = 1.0e-10
        end
    end

    @testset "Averaged marginals are proper mixtures" begin
        bma = model_average([result1, result2])

        for m in bma.latent_marginals
            # PDF integrates to ~1 (spot check via CDF)
            @test cdf(m, 100.0) ≈ 1.0 atol = 1.0e-6
            @test cdf(m, -100.0) ≈ 0.0 atol = 1.0e-6

            # Can compute quantiles
            q50 = quantile(m, 0.5)
            @test isfinite(q50)
        end
    end

    @testset "summary_df works on BMA marginals" begin
        bma = model_average([result1, result2])
        df = summary_df(bma.latent_marginals)
        @test size(df, 1) == length(bma.latent_marginals)
        @test :mean in propertynames(df)
        @test :q2_5 in propertynames(df)
    end

    @testset "Validation errors" begin
        # Empty vector
        @test_throws ArgumentError model_average(INLAResult[])

        # Mismatched latent dimensions
        small_latent = IIDModel(1)
        small_obs = ExponentialFamily(Poisson)
        hp_small = @hyperparams begin
            (τ ~ Exponential(1.0), transform = log, space = natural)
        end
        model_small = LatentGaussianModel(hp_small, small_latent, small_obs)
        y_small = rand(Poisson(3.0), 1)
        result_small = inla(model_small, y_small; progress = false)

        @test_throws DimensionMismatch model_average([result1, result_small])

        # Invalid prior weights length
        @test_throws ArgumentError model_average(
            [result1, result2]; prior_weights = [0.5, 0.3, 0.2]
        )
    end
end
