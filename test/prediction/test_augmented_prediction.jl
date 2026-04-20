using Test
using Latte
using Latte: _prepare_for_prediction
using GaussianMarkovRandomFields
using Distributions
using SparseArrays
using Random
using LinearAlgebra

@testset "Prediction with Augmented Models" begin

    @testset "_prepare_for_prediction accepts augmented models with missing values" begin
        n_base = 5
        n_obs = 10
        A = randn(n_obs, n_base)
        base_model = IIDModel(n_base)
        ltom = LinearlyTransformedObservationModel(ExponentialFamily(Poisson), A)
        spec = @hyperparams begin
            (τ ~ Gamma(2, 1), transform = log, space = natural)
        end
        model = LatentGaussianModel(spec, base_model, ltom)

        y = Union{Missing, Int}[1, missing, 3, 4, missing, 6, 7, 8, 9, 10]

        # Should not throw
        y_obs, model_pred, pred_info = _prepare_for_prediction(model, y)

        # Prediction info should track η indices for missing observations
        @test pred_info isa PredictionInfo
        @test pred_info.prediction_indices == [2, 5]
        @test pred_info.observed_indices == [1, 3, 4, 6, 7, 8, 9, 10]

        # y_obs should contain only observed values
        @test length(y_obs) == 8

        # Model should still have the same augmented latent prior
        @test model_pred.latent_prior === model.latent_prior

        # Observation model should be restricted to observed η indices
        @test model_pred.observation_model !== model.observation_model
    end

    @testset "End-to-end prediction with augmented Normal model" begin
        Random.seed!(42)
        n_base = 5
        n_obs = 15
        A = randn(n_obs, n_base) / sqrt(n_base)

        hp = @hyperparams begin
            (σ ~ InverseGamma(2, 1), transform = log, space = natural)
            (τ ~ Gamma(2, 1), transform = log, space = natural)
        end

        ltom = LinearlyTransformedObservationModel(ExponentialFamily(Normal), A)
        model = LatentGaussianModel(hp, IIDModel(n_base), ltom)

        # Generate data and mark some as missing
        y_full = randn(n_obs)
        y = Vector{Union{Missing, Float64}}(y_full)
        y[3] = missing
        y[7] = missing
        y[12] = missing

        result = inla(model, y; progress = false, diff_strategy = FiniteDiffStrategy())

        # Should have prediction info
        @test result.prediction_info isa PredictionInfo
        @test result.prediction_info.prediction_indices == [3, 7, 12]

        # predicted_marginals should return distributions at missing locations
        pred_m = predicted_marginals(result)
        @test length(pred_m) == 3
        for m in pred_m
            @test isfinite(mean(m))
            @test std(m) > 0
        end

        # observed_marginals should return the rest
        obs_m = observed_marginals(result)
        @test length(obs_m) == 12
    end

    @testset "End-to-end prediction with augmented Poisson model" begin
        Random.seed!(123)
        n_base = 4
        n_obs = 10
        A = abs.(randn(n_obs, n_base)) / sqrt(n_base)

        hp = @hyperparams begin
            (τ ~ Gamma(2, 1), transform = log, space = natural)
        end

        ltom = LinearlyTransformedObservationModel(ExponentialFamily(Poisson), A)
        model = LatentGaussianModel(hp, IIDModel(n_base), ltom)

        y = Union{Missing, Int}[3, 1, missing, 2, 5, missing, 1, 4, 2, 3]

        result = inla(model, y; progress = false, diff_strategy = FiniteDiffStrategy())

        @test result.prediction_info isa PredictionInfo
        @test result.prediction_info.prediction_indices == [3, 6]

        pred_m = predicted_marginals(result)
        @test length(pred_m) == 2
        for m in pred_m
            @test isfinite(mean(m))
        end
    end

    @testset "Augmented model without missing values still works" begin
        Random.seed!(42)
        n_base = 5
        n_obs = 10
        A = randn(n_obs, n_base) / sqrt(n_base)

        hp = @hyperparams begin
            (σ ~ InverseGamma(2, 1), transform = log, space = natural)
            (τ ~ Gamma(2, 1), transform = log, space = natural)
        end

        ltom = LinearlyTransformedObservationModel(ExponentialFamily(Normal), A)
        model = LatentGaussianModel(hp, IIDModel(n_base), ltom)

        y = randn(n_obs)
        result = inla(model, y; progress = false, diff_strategy = FiniteDiffStrategy())

        # No prediction info when no missing values
        @test result.prediction_info === nothing
        @test_throws ArgumentError predicted_marginals(result)
    end

end
