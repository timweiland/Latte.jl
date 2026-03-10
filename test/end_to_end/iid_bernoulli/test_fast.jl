using Test
using IntegratedNestedLaplace
using GaussianMarkovRandomFields
using Distributions
using LinearAlgebra
using SparseArrays
using Random
using JLD2
using Statistics

@testset "End-to-End Test: IID Bernoulli Model" begin

    reference_file = joinpath(@__DIR__, "reference_data.jld2")

    if !isfile(reference_file)
        @warn "Reference file not found: $reference_file"
        @warn "Please run generate_reference.jl first"
        @test_skip "Reference data not available"
        return
    end

    @load reference_file y x_samples model_params

    n = model_params.n
    τ_fixed = model_params.τ_fixed

    # Model setup: IID prior with fixed precision (no hyperparameter exploration)
    # We use a very tight prior on τ to effectively fix it.
    spec = @hyperparams begin
        (log_τ ~ Normal(log(τ_fixed), 0.01), transform = identity, space = working)
    end

    function latent_gmrf(; log_τ, kwargs...)
        τ = exp(log_τ)
        Q = spdiagm(0 => fill(τ, n))
        return GMRF(zeros(n), Q)
    end

    obs_model = ExponentialFamily(Bernoulli)
    model = INLAModel(spec, FunctionLatentModel(latent_gmrf, n), obs_model)

    # Run with all three methods
    result_g = inla(
        model, y; progress = false,
        latent_marginalization_method = GaussianMarginal()
    )

    result_sl = inla(
        model, y; progress = false,
        latent_marginalization_method = SimplifiedLaplace()
    )

    result_la = inla(
        model, y; progress = false,
        latent_marginalization_method = LaplaceMarginal()
    )

    test_indices = [1, 5, 10, 15, 20]

    @testset "LaplaceMarginal vs MCMC" begin
        for i in test_indices
            mcmc_mean = mean(x_samples[:, i])
            mcmc_std = std(x_samples[:, i])

            @test mean(result_la.latent_marginals[i]) ≈ mcmc_mean atol = 0.3
            @test std(result_la.latent_marginals[i]) ≈ mcmc_std rtol = 0.3
        end
    end

    @testset "SimplifiedLaplace vs MCMC" begin
        for i in test_indices
            mcmc_mean = mean(x_samples[:, i])
            mcmc_std = std(x_samples[:, i])

            @test mean(result_sl.latent_marginals[i]) ≈ mcmc_mean atol = 0.3
            @test std(result_sl.latent_marginals[i]) ≈ mcmc_std rtol = 0.3
        end
    end

    @testset "Laplace correction matters for Bernoulli" begin
        # For at least some nodes, Laplace/SimplifiedLaplace should differ
        # meaningfully from the Gaussian marginal
        max_mean_diff = maximum(
            abs(mean(result_la.latent_marginals[i]) - mean(result_g.latent_marginals[i]))
                for i in 1:n
        )
        @test max_mean_diff > 0.01
    end

    @testset "SimplifiedLaplace vs LaplaceMarginal agreement" begin
        for i in 1:n
            @test mean(result_sl.latent_marginals[i]) ≈ mean(result_la.latent_marginals[i]) atol = 0.2
            @test std(result_sl.latent_marginals[i]) ≈ std(result_la.latent_marginals[i]) rtol = 0.25
        end
    end

    # Run AdaptiveMarginal
    result_adaptive = inla(
        model, y; progress = false,
        latent_marginalization_method = AdaptiveMarginal()
    )

    @testset "AdaptiveMarginal vs MCMC" begin
        for i in test_indices
            mcmc_mean = mean(x_samples[:, i])
            mcmc_std = std(x_samples[:, i])

            @test mean(result_adaptive.latent_marginals[i]) ≈ mcmc_mean atol = 0.3
            @test std(result_adaptive.latent_marginals[i]) ≈ mcmc_std rtol = 0.3
        end
    end

    @testset "KLD diagnostics" begin
        # All results should have KLD vectors
        @test result_g.kld !== nothing
        @test result_sl.kld !== nothing
        @test result_la.kld !== nothing

        @test length(result_g.kld) == n
        @test length(result_sl.kld) == n
        @test length(result_la.kld) == n

        # GaussianMarginal KLD should be zero
        @test all(result_g.kld .== 0.0)

        # Non-Gaussian methods should have non-negative KLD
        @test all(result_sl.kld .>= 0.0)
        @test all(result_la.kld .>= 0.0)

        # AdaptiveMarginal should also have KLD
        @test result_adaptive.kld !== nothing
        @test length(result_adaptive.kld) == n
        @test all(result_adaptive.kld .>= 0.0)
    end
end
