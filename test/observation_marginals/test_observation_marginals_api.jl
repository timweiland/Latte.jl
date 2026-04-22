using Test
using Latte
using GaussianMarkovRandomFields
using Distributions
using Random

@testset "observation_marginals API Tests" begin
    Random.seed!(54321)

    @testset "Poisson Regression with Augmented Model" begin
        # Setup: Simple Poisson regression with augmented latent field
        n_base = 10
        n_obs = 20
        A = randn(n_obs, n_base) / sqrt(n_base)

        base_latent_model = IIDModel(n_base)
        base_obs_model = ExponentialFamily(Poisson)
        obs_model = LinearlyTransformedObservationModel(base_obs_model, A)

        hp_spec = @hyperparams begin
            (τ ~ Exponential(1.0), transform = log, space = natural)
        end

        # Create augmented model
        model = LatentGaussianModel(hp_spec, base_latent_model, obs_model)
        @test model.augmentation_info !== nothing

        # Generate data
        τ_true = 3.0
        gmrf = base_latent_model(τ = τ_true)
        x_base = rand(gmrf)
        η_true = A * x_base
        λ_true = exp.(η_true)
        y = rand.(Poisson.(λ_true))

        # Run INLA
        result = inla(model, y; progress = false)

        # Get observation marginals
        obs_marginals = observation_marginals(result)

        # Check dimensions
        @test length(obs_marginals) == n_obs
        @test all(m isa TransformedWeightedMixture for m in obs_marginals)

        # Check that bijector is correct (log for Poisson)
        @test obs_marginals[1].bijector === elementwise(log)

        # Check that marginals are positive (Poisson rates)
        for i in 1:n_obs
            @test minimum(obs_marginals[i]) >= 0.0
            @test mean(obs_marginals[i]) > 0.0
        end

        # Marginal means should be reasonable estimates of true λ
        for i in 1:5  # Check first few
            λ_est = mean(obs_marginals[i])
            @test λ_est > 0  # Positive rate
            # Not checking closeness to true value as INLA is approximate
        end
    end

    @testset "Error: Non-augmented Model" begin
        # Create a model without augmentation
        latent_model = IIDModel(10)
        obs_model = ExponentialFamily(Poisson)

        hp_spec = @hyperparams begin
            (τ ~ Exponential(1.0), transform = log, space = natural)
        end

        model = LatentGaussianModel(hp_spec, latent_model, obs_model)

        # Generate simple data
        y = rand(Poisson(5.0), 10)

        # Run INLA
        result = inla(model, y; progress = false)

        # Should error because no augmentation
        @test_throws ErrorException observation_marginals(result)
    end

    @testset "Different Link Functions" begin
        # Test that different link functions are correctly identified

        @testset "LogLink (Poisson)" begin
            n_base = 5
            n_obs = 10
            A = randn(n_obs, n_base) / sqrt(n_base)

            base_model = IIDModel(n_base)
            base_obs = ExponentialFamily(Poisson)  # LogLink
            obs_model = LinearlyTransformedObservationModel(base_obs, A)

            hp_spec = @hyperparams begin
                (τ ~ Exponential(1.0), transform = log, space = natural)
            end

            model = LatentGaussianModel(hp_spec, base_model, obs_model)
            y = fill(5, n_obs)

            result = inla(model, y; progress = false)
            obs_marginals = observation_marginals(result)

            # Should use elementwise(log) bijector
            @test obs_marginals[1].bijector === elementwise(log)
        end

        @testset "LogitLink (Binomial)" begin
            n_base = 5
            n_obs = 10
            A = randn(n_obs, n_base) / sqrt(n_base)

            base_model = IIDModel(n_base)
            base_obs = ExponentialFamily(Binomial, LogitLink())
            obs_model = LinearlyTransformedObservationModel(base_obs, A)

            hp_spec = @hyperparams begin
                (τ ~ Exponential(1.0), transform = log, space = natural)
            end

            model = LatentGaussianModel(hp_spec, base_model, obs_model)
            y = BinomialObservations(fill(5, n_obs), fill(10, n_obs))

            result = inla(model, y; progress = false)
            obs_marginals = observation_marginals(result)

            # Should use Logit bijector
            @test obs_marginals[1].bijector isa Bijectors.Logit

            # Marginals should be in (0, 1) for probabilities
            for i in 1:n_obs
                @test 0.0 <= minimum(obs_marginals[i])
                @test maximum(obs_marginals[i]) <= 1.0
                @test 0.0 < mean(obs_marginals[i]) < 1.0
            end
        end
    end

    @testset "Observation Marginals Statistics" begin
        # Test that we can compute various statistics
        n_base = 5
        n_obs = 10
        A = randn(n_obs, n_base) / sqrt(n_base)

        base_model = IIDModel(n_base)
        base_obs = ExponentialFamily(Poisson)
        obs_model = LinearlyTransformedObservationModel(base_obs, A)

        hp_spec = @hyperparams begin
            (τ ~ Exponential(1.0), transform = log, space = natural)
        end

        model = LatentGaussianModel(hp_spec, base_model, obs_model)
        y = fill(5, n_obs)

        result = inla(model, y; progress = false)
        obs_marginals = observation_marginals(result)

        # Test that we can compute various statistics
        for i in 1:3
            m = obs_marginals[i]

            # Mean and variance
            μ = mean(m)
            σ² = var(m)
            @test μ > 0.0
            @test σ² > 0.0

            # Quantiles for credible intervals
            q025 = quantile(m, 0.025)
            q50 = quantile(m, 0.5)
            q975 = quantile(m, 0.975)
            @test q025 < q50 < q975

            # PDF evaluation
            @test pdf(m, μ) > 0.0

            # Sampling
            samples = [rand(m) for _ in 1:100]
            @test all(s > 0 for s in samples)  # Positive for Poisson
        end
    end

    @testset "OffsetObservationModel — offset included in fitted values (R-INLA convention)" begin
        # `observation_marginals` on a model whose obs model is wrapped in
        # `OffsetObservationModel` must return marginals of `g⁻¹(η + offsetᵢ)`,
        # not `g⁻¹(η)` — matching R-INLA's "fitted values include offset"
        # convention.
        #
        # Verified via the median identity for strictly-monotonic transforms:
        # if y = g⁻¹(η + c), median(y) = g⁻¹(median(η) + c).
        Random.seed!(0xbeef)
        n_base = 6
        n_obs = 15
        A = randn(n_obs, n_base) / sqrt(n_base)
        offset = randn(n_obs) .* 0.5   # distinctive, well away from zero

        base_latent = IIDModel(n_base)
        base_obs = ExponentialFamily(Poisson)     # LogLink default
        offset_obs = Latte.OffsetObservationModel(base_obs, offset)
        obs_model = LinearlyTransformedObservationModel(offset_obs, A)

        hp_spec = @hyperparams begin
            (τ ~ Exponential(1.0), transform = log, space = natural)
        end
        model = LatentGaussianModel(hp_spec, base_latent, obs_model)

        # Data generated with the offset present in the linear predictor
        x_base = rand(base_latent(τ = 3.0))
        y = rand.(Poisson.(exp.(A * x_base .+ offset)))

        result = inla(model, y; progress = false)
        obs_marg = observation_marginals(result)
        η_marg = result.linear_predictor_marginals

        # Exact identity (monotonic transform): median commutes.
        for i in 1:n_obs
            η_med = median(η_marg[i])
            @test median(obs_marg[i]) ≈ exp(η_med + offset[i]) rtol = 1.0e-6
        end

        # Sanity: `exp(η_med)` (the offset-dropped answer) must differ from
        # the fitted value, otherwise the identity above would hold even if
        # the offset were silently ignored.
        mismatches = [
            !isapprox(median(obs_marg[i]), exp(median(η_marg[i])); rtol = 1.0e-3)
                for i in 1:n_obs
        ]
        @test count(mismatches) > n_obs ÷ 2
    end
end
