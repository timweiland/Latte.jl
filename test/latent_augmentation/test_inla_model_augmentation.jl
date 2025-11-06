using Test
using IntegratedNestedLaplace
using GaussianMarkovRandomFields
using LinearAlgebra
using Distributions

@testset "INLAModel Augmentation Tests" begin
    @testset "Automatic Augmentation with LatentModel" begin
        # Create a simple test case
        n_base = 5
        n_obs = 10

        # Base latent model
        base_model = IIDModel(n_base)

        # Design matrix
        A = randn(n_obs, n_base)

        # Observation model
        base_obs = ExponentialFamily(Poisson)
        obs_model = LinearlyTransformedObservationModel(base_obs, A)

        # Hyperparameter spec
        hp_spec = @hyperparams begin
            (τ ~ Exponential(1.0), transform = log, space = natural)
        end

        # Create INLAModel (should automatically augment)
        model = INLAModel(hp_spec, base_model, obs_model)

        # Check that augmentation occurred
        @test model.augmentation_info !== nothing
        @test model.augmentation_info.n_linear_predictors == n_obs
        @test model.augmentation_info.n_base_latent == n_base

        # Check that observation model was unwrapped
        @test model.observation_model === base_obs

        # Check that latent_prior is now an AugmentedLatentModel
        @test model.latent_prior isa AugmentedLatentModel
    end

    @testset "Automatic Augmentation with Function" begin
        # Test with latent_prior as a function
        n_base = 5
        n_obs = 10

        # Latent prior function
        function my_latent_prior(; τ)
            Q = Diagonal(fill(τ, n_base))
            return GMRF(zeros(n_base), Q)
        end

        # Design matrix
        A = randn(n_obs, n_base)

        # Observation model
        base_obs = ExponentialFamily(Poisson)
        obs_model = LinearlyTransformedObservationModel(base_obs, A)

        # Hyperparameter spec
        hp_spec = @hyperparams begin
            (τ ~ Exponential(1.0), transform = log, space = natural)
        end

        # Create INLAModel (should wrap function and augment)
        model = INLAModel(hp_spec, my_latent_prior, obs_model)

        # Check that augmentation occurred
        @test model.augmentation_info !== nothing
        @test model.augmentation_info.n_linear_predictors == n_obs
        @test model.augmentation_info.n_base_latent == n_base

        # latent_prior should be AugmentedLatentModel wrapping the function
        @test model.latent_prior isa AugmentedLatentModel
    end

    @testset "Opt-out of Augmentation" begin
        # Same setup as above
        n_base = 5
        n_obs = 10

        base_model = IIDModel(n_base)
        A = randn(n_obs, n_base)
        base_obs = ExponentialFamily(Poisson)
        obs_model = LinearlyTransformedObservationModel(base_obs, A)

        hp_spec = @hyperparams begin
            (τ ~ Exponential(1.0), transform = log, space = natural)
        end

        # Create with augment_latent=false
        model = INLAModel(hp_spec, base_model, obs_model; augment_latent = false)

        # Check that augmentation did NOT occur
        @test model.augmentation_info === nothing
        @test model.observation_model === obs_model  # Not unwrapped
        @test model.latent_prior === base_model      # Not augmented
    end

    @testset "Dimension Mismatch Detection" begin
        # Create mismatched dimensions
        n_base = 5
        n_obs = 10
        n_wrong = 7  # Wrong base dimension

        wrong_base_model = IIDModel(n_wrong)
        A = randn(n_obs, n_base)  # Expects n_base columns
        base_obs = ExponentialFamily(Poisson)
        obs_model = LinearlyTransformedObservationModel(base_obs, A)

        hp_spec = @hyperparams begin
            (τ ~ Exponential(1.0), transform = log, space = natural)
        end

        # Should throw dimension mismatch error
        @test_throws ErrorException INLAModel(hp_spec, wrong_base_model, obs_model)
    end

    @testset "Custom Linear Predictor Precision" begin
        n_base = 5
        n_obs = 10

        base_model = IIDModel(n_base)
        A = randn(n_obs, n_base)
        base_obs = ExponentialFamily(Poisson)
        obs_model = LinearlyTransformedObservationModel(base_obs, A)

        hp_spec = @hyperparams begin
            (τ ~ Exponential(1.0), transform = log, space = natural)
        end

        # Create with custom precision
        custom_precision = 1.0e3
        model = INLAModel(
            hp_spec, base_model, obs_model;
            linear_predictor_precision = custom_precision
        )

        @test model.augmentation_info !== nothing

        # Check that the augmented model uses the custom precision
        @test model.latent_prior isa AugmentedLatentModel
        @test model.latent_prior.linear_predictor_precision == custom_precision
    end

    @testset "GMRF Generation from Augmented Model" begin
        n_base = 3
        n_obs = 5

        base_model = IIDModel(n_base)
        A = randn(n_obs, n_base)
        base_obs = ExponentialFamily(Poisson)
        obs_model = LinearlyTransformedObservationModel(base_obs, A)

        hp_spec = @hyperparams begin
            (τ ~ Exponential(1.0), transform = log, space = natural)
        end

        model = INLAModel(hp_spec, base_model, obs_model)

        # Generate GMRF using the model
        θ_named = (τ = 2.0,)
        gmrf = latent_gmrf(model, θ_named)

        # Check dimensions
        @test length(mean(gmrf)) == n_obs + n_base
        Q = precision_matrix(gmrf)
        @test size(Q) == (n_obs + n_base, n_obs + n_base)
    end
end
