using Test
using IntegratedNestedLaplace
using GaussianMarkovRandomFields
using Distributions
using LinearAlgebra
using SparseArrays
using Random
using Turing
using StatsBase
using LDLFactorizations

@testset "End-to-End Test: AR-1 Poisson Model" begin

    # Set reproducible seed
    Random.seed!(83498)

    # AR-1 precision matrix function
    function ar_precision(ρ, k)
        return spdiagm(-1 => -ρ * ones(k - 1), 0 => ones(k) .+ ρ^2, 1 => -ρ * ones(k - 1))
    end

    # Test parameters (smaller than example for speed)
    k = 200
    σ_gmrf_true = 0.3
    ρ_true = 0.98
    τ_gmrf_log_true = log(1 / σ_gmrf_true^2)
    η_true = atanh(ρ_true)

    # Model setup
    desired_std_dev = 0.5 * (atanh(0.98) - atanh(0.95))
    θ_prior = HyperparameterPrior((τ_gmrf_log = Normal(0, 1), η = Normal(atanh(0.95), desired_std_dev)))

    function latent_gmrf(θ)
        τ = exp(θ.τ_gmrf_log)
        ρ = tanh(θ.η)

        # Create AR-1 precision matrix (from the example)
        Q = ar_precision(ρ, k) .* τ

        μ₀ = log(1000.0)
        μ = μ₀ .* [ρ^i for i in 1:k]

        return (μ, Q)
    end
    obs_model = ExponentialFamily(Poisson)
    model = INLAModel(θ_prior, latent_gmrf, obs_model)

    # Generate synthetic data
    x_gt = rand(GMRF(latent_gmrf((τ_gmrf_log = τ_gmrf_log_true, η = η_true)...)))
    y_gt = rand(obs_model; x = x_gt, θ_named = NamedTuple())

    # Run INLA inference
    inla_start_time = time()
    inla_result = inla(model, y_gt, progress = false, latent_marginalization_method = LaplaceMarginal())
    inla_time = time() - inla_start_time

    # Run MCMC reference
    function latent_gmrf_ad(θ)
        τ = exp(θ.τ_gmrf_log)
        ρ = tanh(θ.η)
        Q = ar_precision(ρ, k) .* τ
        μ₀ = log(1000.0)
        μ = μ₀ .* [ρ^i for i in 1:k]
        return (μ, Q)
    end
    @model function mcmc_model(y)
        τ_gmrf_log ~ Normal(0, 1)
        η ~ Normal(atanh(0.95), desired_std_dev)
        x ~ latent_gmrf_ad((τ_gmrf_log = τ_gmrf_log, η = η))
        y ~ likelihood(ExponentialFamily(Poisson), x, ())
    end

    mcmc_start_time = time()
    chain = sample(mcmc_model(y_gt), NUTS(), 800, progress = true)
    mcmc_time = time() - mcmc_start_time

    τ_gmrf_log_samples = vec(chain[:τ_gmrf_log])
    η_samples = vec(chain[:η])

    # Extract latent field samples (x variables)
    x_samples = hcat([vec(chain[Symbol("x[$i]")]) for i in 1:k]...)

    @testset "INLA Result Structure" begin
        @test isa(inla_result, INLAResult)
        @test length(inla_result.hyperparameter_marginals) == 2
        @test isa(inla_result.latent_marginals, Vector{WeightedMixture})
        @test length(inla_result.latent_marginals) == k
        @test length(inla_result.hyperparameter_mode) == 2
        @test inla_result.convergence.mode_converged == true
    end

    @testset "Statistical Comparison" begin
        # MCMC samples are in working space (τ_gmrf_log, η)
        # Transform to natural space for comparison with INLA marginals (which are now in natural space)
        τ_gmrf_samples_natural = exp.(τ_gmrf_log_samples)  # Transform from log(τ) to τ
        η_samples_natural = η_samples  # η has identity transform

        τ_gmrf_marginal = inla_result.hyperparameter_marginals.τ_gmrf_log
        η_marginal = inla_result.hyperparameter_marginals.η

        # Compare hyperparameter posterior means in natural space
        @test mean(τ_gmrf_marginal) ≈ mean(τ_gmrf_samples_natural) rtol = 0.1
        @test mean(η_marginal) ≈ mean(η_samples_natural) rtol = 0.1

        # Compare hyperparameter credible interval bounds in natural space
        inla_τ_ci = quantile(τ_gmrf_marginal, [0.025, 0.975])
        inla_η_ci = quantile(η_marginal, [0.025, 0.975])
        mcmc_τ_ci = quantile(τ_gmrf_samples_natural, [0.025, 0.975])
        mcmc_η_ci = quantile(η_samples_natural, [0.025, 0.975])

        @test inla_τ_ci[1] ≈ mcmc_τ_ci[1] rtol = 0.2  # Lower bound
        @test inla_τ_ci[2] ≈ mcmc_τ_ci[2] rtol = 0.2  # Upper bound
        @test inla_η_ci[1] ≈ mcmc_η_ci[1] rtol = 0.2
        @test inla_η_ci[2] ≈ mcmc_η_ci[2] rtol = 0.2

        # Compare latent field marginals (test a subset for speed)
        test_indices = [1, 10, 50, 100, 150, 200]  # Sample across the field
        for i in test_indices
            inla_latent_marginal = inla_result.latent_marginals[i]
            mcmc_latent_samples = x_samples[:, i]

            # Compare means
            @test mean(inla_latent_marginal) ≈ mean(mcmc_latent_samples) rtol = 0.15 atol = 0.1

            # Compare standard deviations
            @test std(inla_latent_marginal) ≈ std(mcmc_latent_samples) rtol = 0.2 atol = 0.1

            # Compare quantiles
            inla_latent_ci = quantile(inla_latent_marginal, [0.025, 0.975])
            mcmc_latent_ci = quantile(mcmc_latent_samples, [0.025, 0.975])

            @test inla_latent_ci[1] ≈ mcmc_latent_ci[1] rtol = 0.25 atol = 0.1
            @test inla_latent_ci[2] ≈ mcmc_latent_ci[2] rtol = 0.25 atol = 0.1
        end
    end

    @testset "Model Properties" begin
        # Verify nonlinear model handling
        @test isa(inla_result.model.observation_model, ExponentialFamily{Poisson})

        # Latent field should give reasonable Poisson rates
        latent_means = [mean(m) for m in inla_result.latent_marginals[1:10]]
        poisson_rates = exp.(latent_means)
        @test all(0 < r < 10000 for r in poisson_rates)
    end

    @testset "Performance" begin
        @test inla_time < 60.0
        @test inla_time < mcmc_time
        @test mcmc_time / inla_time > 2.0  # INLA should be significantly faster
    end

    @testset "Error Handling" begin
        @test_throws ArgumentError inla(model, Float64[])
        @test_throws ArgumentError inla(model, y_gt, latent_indices = Int[])
        @test_throws ArgumentError inla(model, y_gt, latent_indices = [k + 1])
    end
end
