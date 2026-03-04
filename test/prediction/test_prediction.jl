using Test
using IntegratedNestedLaplace
using GaussianMarkovRandomFields
using GaussianMarkovRandomFields: PoissonObservations
using Distributions
using SparseArrays

@testset "Prediction via Missing Values" begin

    # Shared helpers
    function make_normal_model(n)
        spec = @hyperparams begin
            (σ ~ InverseGamma(2, 1), transform = log, space = natural)
        end
        function latent_func(; σ, kwargs...)
            Q = spdiagm(0 => fill(1 / σ^2, n))
            return GMRF(zeros(n), Q)
        end
        obs_model = ExponentialFamily(Normal)
        return INLAModel(spec, FunctionLatentModel(latent_func, n), obs_model)
    end

    function make_poisson_model(n)
        spec = @hyperparams begin
            (τ ~ Gamma(2, 1), transform = log, space = natural)
        end
        function latent_func(; τ, kwargs...)
            Q = spdiagm(0 => fill(τ, n))
            return GMRF(zeros(n), Q)
        end
        obs_model = ExponentialFamily(Poisson)
        return INLAModel(spec, FunctionLatentModel(latent_func, n), obs_model)
    end

    @testset "PredictionInfo" begin
        mask = [true, false, true, true, false]
        info = PredictionInfo(5, mask)
        @test info.n_latent == 5
        @test info.observed_indices == [1, 3, 4]
        @test info.prediction_indices == [2, 5]

        # Display
        str = sprint(show, info)
        @test occursin("3 observed", str)
        @test occursin("2 predicted", str)
    end

    @testset "_normalize_observations" begin
        poisson_obs = ExponentialFamily(Poisson)
        normal_obs = ExponentialFamily(Normal)

        # Plain integer vector + Poisson → PoissonObservations
        y_int = [3, 1, 4, 1, 5]
        y_norm = IntegratedNestedLaplace._normalize_observations(y_int, poisson_obs)
        @test y_norm isa PoissonObservations
        @test y_norm.counts == y_int

        # Float vector + Normal → passthrough
        y_float = [1.0, 2.0, 3.0]
        @test IntegratedNestedLaplace._normalize_observations(y_float, normal_obs) === y_float

        # Already-wrapped PoissonObservations → passthrough
        y_po = PoissonObservations([1, 2, 3])
        @test IntegratedNestedLaplace._normalize_observations(y_po, poisson_obs) === y_po
    end

    @testset "_prepare_for_prediction passthrough" begin
        model = make_normal_model(5)
        y = [1.0, 2.0, 3.0, 4.0, 5.0]

        y_out, model_out, pred_info = IntegratedNestedLaplace._prepare_for_prediction(model, y)
        @test y_out == y
        @test model_out === model
        @test pred_info === nothing
    end

    @testset "_prepare_for_prediction passthrough normalizes Poisson" begin
        model = make_poisson_model(5)
        y = [3, 1, 4, 1, 5]

        y_out, model_out, pred_info = IntegratedNestedLaplace._prepare_for_prediction(model, y)
        @test y_out isa PoissonObservations
        @test y_out.counts == y
        @test model_out === model
        @test pred_info === nothing
    end

    @testset "_prepare_for_prediction with Normal missing" begin
        model = make_normal_model(5)
        y = [1.0, missing, 3.0, 4.0, missing]

        y_obs, model_pred, pred_info = IntegratedNestedLaplace._prepare_for_prediction(model, y)

        @test y_obs == [1.0, 3.0, 4.0]
        @test pred_info isa PredictionInfo
        @test pred_info.n_latent == 5
        @test pred_info.observed_indices == [1, 3, 4]
        @test pred_info.prediction_indices == [2, 5]
        @test model_pred !== model
        @test model_pred.latent_prior === model.latent_prior
    end

    @testset "_prepare_for_prediction with Poisson missing" begin
        model = make_poisson_model(5)
        y = Union{Missing, Int}[3, missing, 4, 1, missing]

        y_obs, model_pred, pred_info = IntegratedNestedLaplace._prepare_for_prediction(model, y)

        @test y_obs isa PoissonObservations
        @test y_obs.counts == [3, 4, 1]
        @test pred_info.observed_indices == [1, 3, 4]
        @test pred_info.prediction_indices == [2, 5]
    end

    @testset "_prepare_for_prediction with Poisson + exposure" begin
        model = make_poisson_model(4)
        y = poisson_observations(
            counts = [3, missing, 4, missing],
            exposure = [1.0, 2.0, 0.5, 1.5]
        )

        y_obs, model_pred, pred_info = IntegratedNestedLaplace._prepare_for_prediction(model, y)

        @test y_obs isa PoissonObservations
        @test y_obs.counts == [3, 4]
        @test y_obs.exposure ≈ [1.0, 0.5]
        @test pred_info.observed_indices == [1, 3]
        @test pred_info.prediction_indices == [2, 4]
    end

    @testset "poisson_observations factory" begin
        # No missing, no exposure
        y1 = poisson_observations(counts = [1, 2, 3])
        @test y1 isa PoissonObservations
        @test y1.counts == [1, 2, 3]

        # No missing, with exposure
        y2 = poisson_observations(counts = [1, 2, 3], exposure = [1.0, 2.0, 3.0])
        @test y2 isa PoissonObservations
        @test y2.counts == [1, 2, 3]
        @test y2.exposure ≈ [1.0, 2.0, 3.0]

        # With missing, no exposure
        y3 = poisson_observations(counts = [1, missing, 3])
        @test y3 isa IntegratedNestedLaplace.MissingPoissonObservations

        # With missing, with exposure
        y4 = poisson_observations(counts = [1, missing, 3], exposure = [1.0, 2.0, 3.0])
        @test y4 isa IntegratedNestedLaplace.MissingPoissonObservations
        @test y4.exposure == [1.0, 2.0, 3.0]
    end

    @testset "Error cases" begin
        model = make_normal_model(3)

        # All missing
        @test_throws ArgumentError IntegratedNestedLaplace._prepare_for_prediction(
            model, [missing, missing, missing]
        )

        # Augmented model (LTOM) with missing values
        n_base = 5
        n_obs = 10
        A = randn(n_obs, n_base)
        base_model = IIDModel(n_base)
        ltom = LinearlyTransformedObservationModel(ExponentialFamily(Poisson), A)
        spec = @hyperparams begin
            (τ ~ Gamma(2, 1), transform = log, space = natural)
        end
        augmented_model = INLAModel(spec, base_model, ltom)
        y_with_missing = Union{Missing, Int}[1, missing, 3, 4, 5, 6, 7, 8, 9, 10]
        @test_throws ArgumentError IntegratedNestedLaplace._prepare_for_prediction(
            augmented_model, y_with_missing
        )
    end

    @testset "predicted_marginals and observed_marginals accessors" begin
        # Without prediction: observed_marginals returns all, predicted_marginals throws
        model = make_normal_model(5)
        y = randn(5)

        result = inla(model, y; progress = false)

        @test observed_marginals(result) == result.latent_marginals
        @test_throws ArgumentError predicted_marginals(result)
    end

    @testset "End-to-end prediction with Normal model" begin
        n = 10
        model = make_normal_model(n)

        # Create y with 2 missing values
        y = Vector{Union{Missing, Float64}}(randn(n))
        y[3] = missing
        y[7] = missing

        result = inla(model, y; progress = false)

        # Result should have prediction info
        @test result.prediction_info isa PredictionInfo
        @test result.prediction_info.observed_indices == [1, 2, 4, 5, 6, 8, 9, 10]
        @test result.prediction_info.prediction_indices == [3, 7]

        # Latent marginals cover all n positions
        @test length(result.latent_marginals) == n

        # Accessors work
        obs_m = observed_marginals(result)
        pred_m = predicted_marginals(result)
        @test length(obs_m) == 8
        @test length(pred_m) == 2

        # All marginals should be finite
        for m in result.latent_marginals
            @test isfinite(mean(m))
            @test std(m) > 0
        end
    end
end
