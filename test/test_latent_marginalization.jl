using Test
using IntegratedNestedLaplace
using GaussianMarkovRandomFields
using LinearAlgebra
using SparseArrays
using Distributions
using Random

@testset "Marginalization" begin

    Random.seed!(42)

    @testset "Gaussian Likelihood - Laplace should match Gaussian exactly" begin
        # Test case where Gaussian and Laplace should give identical results
        # since the likelihood is Gaussian (no correction needed)

        n = 6

        # Create AR(1) GMRF prior
        function ar_precision(ρ, k)
            Q = spdiagm(-1 => -ρ * ones(k - 1), 0 => ones(k), 1 => -ρ * ones(k - 1))
            return Q
        end

        σ_prior = 0.5
        ρ = 0.3
        Q_prior = ar_precision(ρ, n) ./ σ_prior^2
        μ_prior = zeros(n)
        prior_gmrf = GMRF(μ_prior, Q_prior)

        # Gaussian observation model (canonical link)
        obs_model = ExponentialFamily(Normal)
        θ = (σ = 0.8,)  # Observation noise

        # Generate synthetic data
        x_true = rand(prior_gmrf)
        y = x_true + θ.σ * randn(n)  # Add observation noise

        # Compute Gaussian approximation
        obs_lik = obs_model(y; θ...)
        ga = gaussian_approximation(prior_gmrf, obs_lik)
        log_prior_θ = 0.0

        # Test indices
        test_indices = [1, 3, 5]

        # Compute marginals with both methods
        gauss_result = marginalize(ga, obs_lik, log_prior_θ, GaussianMarginal(), test_indices)
        laplace_result = marginalize(ga, obs_lik, log_prior_θ, LaplaceMarginal(true), test_indices; prior_gmrf = prior_gmrf)

        @test length(gauss_result.marginals) == length(laplace_result.marginals)

        # For Gaussian likelihoods, Laplace should match Gaussian exactly
        for (i, idx) in enumerate(test_indices)
            gauss_marginal = gauss_result.marginals[i]
            laplace_marginal = laplace_result.marginals[i]

            # Check that means and variances match closely
            @test mean(gauss_marginal) ≈ mean(laplace_marginal) atol = 1.0e-6
            @test var(gauss_marginal) ≈ var(laplace_marginal) atol = 1.0e-6

            # Check that PDFs match at several points
            test_points = [mean(gauss_marginal) + k * std(gauss_marginal) for k in -2:0.5:2]
            for x in test_points
                @test pdf(gauss_marginal, x) ≈ pdf(laplace_marginal, x) rtol = 1.0e-4
            end
        end
    end

    @testset "Bernoulli Likelihood - Non-Gaussian Case" begin
        # Test case with non-Gaussian likelihood where Laplace correction matters

        n = 8

        # Create simple white noise GMRF prior (guaranteed positive definite)
        σ_prior = 0.8
        Q_prior = spdiagm(0 => fill(1 / σ_prior^2, n))

        μ_prior = zeros(n)
        prior_gmrf = GMRF(μ_prior, Q_prior)

        # Bernoulli observation model with logit link (canonical)
        obs_model = ExponentialFamily(Bernoulli)
        θ = NamedTuple()  # No hyperparameters for Bernoulli

        # Generate synthetic data
        x_true = rand(prior_gmrf)
        p_true = 1 ./ (1 .+ exp.(-x_true))  # logit^{-1}(x)
        y = [rand(Bernoulli(p)) for p in p_true]

        # Compute Gaussian approximation
        obs_lik = obs_model(y; θ...)
        ga = gaussian_approximation(prior_gmrf, obs_lik)
        log_prior_θ = 0.0

        # Test different marginalization methods
        test_indices = [2, 4, 6]

        gauss_result = marginalize(ga, obs_lik, log_prior_θ, GaussianMarginal(), test_indices)
        laplace_result = marginalize(ga, obs_lik, log_prior_θ, LaplaceMarginal(true), test_indices; prior_gmrf = prior_gmrf)

        @test length(gauss_result.marginals) == length(laplace_result.marginals)

        # For non-Gaussian likelihoods, methods should differ significantly
        for (i, idx) in enumerate(test_indices)
            gauss_marginal = gauss_result.marginals[i]
            laplace_marginal = laplace_result.marginals[i]

            # Basic sanity checks
            @test isfinite(mean(gauss_marginal))
            @test isfinite(mean(laplace_marginal))
            @test var(gauss_marginal) > 0
            @test var(laplace_marginal) > 0

            # Check types
            @test isa(gauss_marginal, Normal)
            @test isa(laplace_marginal, SplineAugmentedGaussian)

            # Integration should sum to 1
            using HCubature
            μ_l, σ_l = mean(laplace_marginal), std(laplace_marginal)
            integral, _ = hcubature(x -> pdf(laplace_marginal, x[1]), [μ_l - 5 * σ_l], [μ_l + 5 * σ_l], rtol = 1.0e-4)
            @test integral ≈ 1.0 atol = 1.0e-2
        end

        # Test that Laplace correction has meaningful effect
        # (means should be different, though both reasonable)
        mean_diff = abs(mean(gauss_result.marginals[1]) - mean(laplace_result.marginals[1]))
        @test mean_diff > 1.0e-4  # Should see some difference
    end

    @testset "Poisson Likelihood - Log Link" begin
        # Test with Poisson likelihood and log link

        n = 5

        # Simple white noise GMRF prior
        σ_prior = 1.2
        Q_prior = spdiagm(0 => fill(1 / σ_prior^2, n))
        μ_prior = zeros(n)
        prior_gmrf = GMRF(μ_prior, Q_prior)

        # Poisson observation model with log link (canonical)
        obs_model = ExponentialFamily(Poisson)
        θ = NamedTuple()  # No hyperparameters for Poisson

        # Generate synthetic data
        x_true = rand(prior_gmrf)
        λ_true = exp.(x_true)  # log^{-1}(x) = exp(x)
        y = [rand(Poisson(λ)) for λ in λ_true]

        # Compute Gaussian approximation
        obs_lik = obs_model(y; θ...)
        ga = gaussian_approximation(prior_gmrf, obs_lik)
        log_prior_θ = 0.0

        # Test single and multiple indices
        single_result = marginalize(ga, obs_lik, log_prior_θ, LaplaceMarginal(false), [1]; prior_gmrf = prior_gmrf)
        @test length(single_result.marginals) == 1
        @test isa(single_result.marginals[1], SplineAugmentedGaussian)

        multi_result = marginalize(ga, obs_lik, log_prior_θ, LaplaceMarginal(false), [1, 3, 5]; prior_gmrf = prior_gmrf)
        @test length(multi_result.marginals) == 3
        @test all(isa.(multi_result.marginals, SplineAugmentedGaussian))

        # Check that marginals have reasonable properties
        for marginal in multi_result.marginals
            @test isfinite(mean(marginal))
            @test var(marginal) > 0
            @test isfinite(pdf(marginal, mean(marginal)))
        end
    end

    @testset "Method Comparison - Different Normalizations" begin
        # Test different normalization methods for LaplaceMarginal

        n = 4
        Q_prior = spdiagm(0 => fill(2.0, n), -1 => fill(-1.0, n - 1), 1 => fill(-1.0, n - 1))
        μ_prior = zeros(n)
        prior_gmrf = GMRF(μ_prior, Q_prior)

        obs_model = ExponentialFamily(Bernoulli)
        θ = NamedTuple()
        y = [1, 0, 1, 0]  # Binary observations

        # Compute Gaussian approximation
        obs_lik = obs_model(y; θ...)
        ga = gaussian_approximation(prior_gmrf, obs_lik)
        log_prior_θ = 0.0

        # Compare exact vs approximate normalization
        exact_result = marginalize(ga, obs_lik, log_prior_θ, LaplaceMarginal(true), [1, 2]; prior_gmrf = prior_gmrf)
        approx_result = marginalize(ga, obs_lik, log_prior_θ, LaplaceMarginal(false), [1, 2]; prior_gmrf = prior_gmrf)

        @test length(exact_result.marginals) == length(approx_result.marginals)

        # Results should be similar but not identical
        for i in 1:length(exact_result.marginals)
            exact_marginal = exact_result.marginals[i]
            approx_marginal = approx_result.marginals[i]

            # Should be close but not exactly the same
            @test abs(mean(exact_marginal) - mean(approx_marginal)) < 0.1
            @test abs(var(exact_marginal) - var(approx_marginal)) < 0.1
        end
    end

    @testset "Edge Cases and Error Handling" begin
        # Setup minimal case for edge testing
        n = 3
        Q_prior = spdiagm(0 => fill(1.0, n))
        μ_prior = zeros(n)
        prior_gmrf = GMRF(μ_prior, Q_prior)

        obs_model = ExponentialFamily(Normal)
        θ = (σ = 1.0,)
        y = [0.0, 1.0, -0.5]

        obs_lik = obs_model(y; θ...)
        ga = gaussian_approximation(prior_gmrf, obs_lik)
        log_prior_θ = 0.0

        # Empty indices
        empty_result = marginalize(ga, obs_lik, log_prior_θ, GaussianMarginal(), Int[])
        @test length(empty_result.marginals) == 0
        @test empty_result.indices == Int[]

        # Default (all variables)
        all_result = marginalize(ga, obs_lik, log_prior_θ, GaussianMarginal())
        @test length(all_result.marginals) == n
        @test all_result.indices == collect(1:n)

        # Error cases
        @test_throws BoundsError marginalize(ga, obs_lik, log_prior_θ, GaussianMarginal(), [n + 1])
        @test_throws ArgumentError marginalize(ga, obs_lik, log_prior_θ, GaussianMarginal(), [1, 1])

        # Test result structure
        result = marginalize(ga, obs_lik, log_prior_θ, GaussianMarginal(), [1])
        @test isa(result, MarginalResult)
        @test isa(result.computation_time, Float64)
        @test result.computation_time > 0
    end
end
