using Test
using IntegratedNestedLaplace
using GaussianMarkovRandomFields
using Distributions
using LinearAlgebra
using SparseArrays
using Random
using LDLFactorizations
using JLD2

@testset "End-to-End Test: AR-1 Poisson Model (Fast)" begin

    # Load pre-computed MCMC reference data
    reference_file = joinpath(@__DIR__, "reference_data.jld2")

    if !isfile(reference_file)
        @warn "Reference file not found: $reference_file"
        @warn "Please run generate_reference.jl first to create the reference data"
        @test_skip "Reference data not available"
        return
    end

    @load reference_file y_gt τ_gmrf_log_samples η_samples x_samples model_params

    # Extract model parameters
    k = model_params.k
    σ_gmrf_true = model_params.σ_gmrf_true
    ρ_true = model_params.ρ_true
    τ_gmrf_log_true = model_params.τ_gmrf_log_true
    η_true = model_params.η_true
    desired_std_dev = model_params.desired_std_dev

    # Set same seed for reproducibility
    Random.seed!(model_params.seed)

    # AR-1 precision matrix function
    function ar_precision(ρ, k)
        return spdiagm(-1 => -ρ * ones(k - 1), 0 => ones(k) .+ ρ^2, 1 => -ρ * ones(k - 1))
    end

    # Model setup (same as reference generation, using new API)
    spec = @hyperparams begin
        (τ_gmrf ~ Normal(0, 1), transform = log, space = working)
        (η ~ Normal(atanh(0.95), desired_std_dev), transform = identity, space = working)
    end

    function latent_gmrf(; τ_gmrf, η, kwargs...)
        ρ = tanh(η)
        Q = ar_precision(ρ, k) .* τ_gmrf
        μ₀ = log(1000.0)
        μ = μ₀ .* [ρ^i for i in 1:k]
        return GMRF(μ, Q)
    end

    obs_model = ExponentialFamily(Poisson)
    model = INLAModel(spec, latent_gmrf, obs_model)

    # Run INLA inference (fast!)
    inla_start_time = time()
    inla_result = inla(model, y_gt, progress = false, marginalization_method = LaplaceMarginal())
    inla_time = time() - inla_start_time

    @testset "Reference Data Validation" begin
        @test length(y_gt) == k
        @test length(τ_gmrf_log_samples) > 1000  # Should have many samples
        @test length(η_samples) > 1000
        @test size(x_samples, 2) == k
        @test size(x_samples, 1) == length(τ_gmrf_log_samples)
    end

    @testset "INLA Result Structure" begin
        @test isa(inla_result, INLAResult)
        @test length(inla_result.hyperparameter_marginals) == 2
        @test isa(inla_result.latent_marginals, Vector{WeightedMixture})
        @test length(inla_result.latent_marginals) == k
        @test length(inla_result.hyperparameter_mode) == 2
        @test inla_result.convergence.mode_converged == true
    end

    @testset "Statistical Comparison" begin
        τ_gmrf_log_marginal = inla_result.hyperparameter_marginals[1]
        η_marginal = inla_result.hyperparameter_marginals[2]

        # Compare hyperparameter posterior means in transformed space
        @test mean(τ_gmrf_log_marginal) ≈ mean(τ_gmrf_log_samples) rtol = 0.1
        @test mean(η_marginal) ≈ mean(η_samples) rtol = 0.1

        # Compare hyperparameter credible interval bounds in transformed space
        inla_τ_ci = quantile(τ_gmrf_log_marginal, [0.025, 0.975])
        inla_η_ci = quantile(η_marginal, [0.025, 0.975])
        mcmc_τ_ci = quantile(τ_gmrf_log_samples, [0.025, 0.975])
        mcmc_η_ci = quantile(η_samples, [0.025, 0.975])

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
        @test inla_time < 30.0  # Should be very fast without MCMC
        @test inla_time < model_params.mcmc_time  # Should be faster than reference MCMC
        @test model_params.mcmc_time / inla_time > 5.0  # INLA should be much faster
    end

    @testset "Error Handling" begin
        @test_throws ArgumentError inla(model, Float64[])
        @test_throws ArgumentError inla(model, y_gt, latent_indices = Int[])
        @test_throws ArgumentError inla(model, y_gt, latent_indices = [k + 1])
    end

    @testset "Reference Data Quality" begin
        # Verify reference data makes sense
        @test all(y_gt .>= 0)  # Poisson observations should be non-negative
        @test all(isfinite.(τ_gmrf_log_samples))  # MCMC samples should be finite
        @test all(isfinite.(η_samples))
        @test all(isfinite.(x_samples))

        # Check MCMC sample quality
        @test length(unique(τ_gmrf_log_samples)) > 100  # Should have good mixing
        @test length(unique(η_samples)) > 100

        # Parameters should be in reasonable ranges
        @test all(-10 .< τ_gmrf_log_samples .< 10)  # Log precision should be reasonable
        @test all(-5 .< η_samples .< 5)  # atanh(ρ) should be reasonable
    end
end
