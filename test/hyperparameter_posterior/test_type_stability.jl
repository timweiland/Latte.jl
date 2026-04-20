using Test
using Latte
using GaussianMarkovRandomFields
using LDLFactorizations
using Distributions
using LinearAlgebra
using SparseArrays

@testset "Type Stability and Performance" begin

    @testset "Type Stability" begin
        spec = @hyperparams begin
            (τ_scale ~ Gamma(3, 2), transform = log, space = natural)
        end

        function beta_latent(; τ_scale, kwargs...)
            n = 2
            Q = spdiagm(0 => fill(τ_scale, n))
            return (zeros(n), Q)
        end
        obs_model = ExponentialFamily(Bernoulli)
        model = LatentGaussianModel(spec, FunctionLatentModel(beta_latent, 2), obs_model)

        y_test = [true, true]  # Biased data to avoid boundary issues
        θ_test_working = WorkingHyperparameters([1.0], spec)
        ws = make_workspace(model.latent_prior; τ_scale = 1.0)

        # Test type stability of key functions
        @inferred Float64 hyperparameter_logpdf(model, θ_test_working, y_test; ws = ws)

        θ_star, _, _ = find_hyperparameter_mode(model, y_test; collect_points = false)
        @inferred WorkingHyperparameters find_hyperparameter_mode(model, y_test; collect_points = false)[1]

        # Test initial hyperparameter guess function with spec
        initial_guess = Latte.initial_hyperparameter_guess(spec)
        @test initial_guess isa WorkingHyperparameters
    end

    @testset "Memory Allocation" begin
        spec = @hyperparams begin
            (σ ~ InverseGamma(4, 3), transform = log, space = natural)
        end

        function allocation_test_latent(; σ, kwargs...)
            n = 5
            Q = spdiagm(0 => fill(1 / σ^2, n))
            return (zeros(n), Q)
        end
        obs_model = ExponentialFamily(Normal)
        model = LatentGaussianModel(spec, FunctionLatentModel(allocation_test_latent, 5), obs_model)

        y_test = randn(5)
        θ_test_working = WorkingHyperparameters([1.5], spec)
        ws = make_workspace(model.latent_prior; σ = 1.0)

        # Warm up
        hyperparameter_logpdf(model, θ_test_working, y_test; ws = ws)

        # Test that repeated calls don't allocate excessively
        initial_memory = Base.gc_bytes()
        for i in 1:10
            hyperparameter_logpdf(model, θ_test_working, y_test; ws = ws)
        end
        final_memory = Base.gc_bytes()

        # Should not allocate too much memory for repeated evaluations
        memory_increase = final_memory - initial_memory
        @test memory_increase < 1_000_000  # Less than 1MB for 10 evaluations
    end

    @testset "Dimensional Scaling" begin
        # Test that algorithms work correctly across different dimensions

        # 1D case
        spec_1d = @hyperparams begin
            (τ ~ Gamma(2, 1), transform = log, space = natural)
        end

        function latent_1d(; τ, kwargs...)
            n = 3
            Q = spdiagm(0 => fill(τ, n))
            return (zeros(n), Q)
        end
        obs_model = ExponentialFamily(Bernoulli)
        model_1d = LatentGaussianModel(spec_1d, FunctionLatentModel(latent_1d, 3), obs_model)
        y_test_1d = [true, false, true]

        θ_star_1d, _, _ = find_hyperparameter_mode(model_1d, y_test_1d)
        @test length(θ_star_1d) == 1

        # 2D case
        spec_2d = @hyperparams begin
            (α ~ Gamma(2, 1), transform = log, space = natural)
            (β ~ Gamma(2, 1), transform = log, space = natural)
        end

        function latent_2d(; α, β, kwargs...)
            n = 4
            Q = spdiagm(0 => [α, α, β, β])
            return (zeros(n), Q)
        end
        model_2d = LatentGaussianModel(spec_2d, FunctionLatentModel(latent_2d, 4), obs_model)
        y_test_2d = [true, false, true, false]

        θ_star_2d, _, _ = find_hyperparameter_mode(model_2d, y_test_2d)
        @test length(θ_star_2d) == 2

        # 3D case
        spec_3d = @hyperparams begin
            (γ₁ ~ Gamma(2, 1), transform = log, space = natural)
            (γ₂ ~ Gamma(2, 1), transform = log, space = natural)
            (γ₃ ~ Gamma(2, 1), transform = log, space = natural)
        end

        function latent_3d(; γ₁, γ₂, γ₃, kwargs...)
            n = 6
            Q = spdiagm(0 => [γ₁, γ₁, γ₂, γ₂, γ₃, γ₃])
            return (zeros(n), Q)
        end
        model_3d = LatentGaussianModel(spec_3d, FunctionLatentModel(latent_3d, 6), obs_model)
        y_test_3d = [true, false, true, false, true, false]

        θ_star_3d, _, _ = find_hyperparameter_mode(model_3d, y_test_3d)
        @test length(θ_star_3d) == 3

        # All should be in valid range (natural space, all positive)
        @test all(x -> x > 0, values(θ_star_1d))
        @test all(x -> x > 0, values(θ_star_2d))
        @test all(x -> x > 0, values(θ_star_3d))
    end

    @testset "Numerical Stability" begin
        # Test with extreme parameter values
        spec = @hyperparams begin
            (σ ~ InverseGamma(0.1, 0.1), transform = log, space = natural)  # Very peaked prior
        end

        function extreme_latent(; σ, kwargs...)
            n = 3
            Q = spdiagm(0 => fill(1 / σ^2, n))
            return (zeros(n), Q)
        end
        obs_model = ExponentialFamily(Normal)
        model = LatentGaussianModel(spec, FunctionLatentModel(extreme_latent, 3), obs_model)

        # Test with extreme data
        y_extreme_large = [10.0, 12.0, 15.0]  # Large values
        y_extreme_small = [0.001, 0.002, 0.003]  # Small values

        # Should handle extreme cases without crashing
        θ_star_large, _, _ = find_hyperparameter_mode(model, y_extreme_large)
        θ_star_small, _, _ = find_hyperparameter_mode(model, y_extreme_small)


        @test isfinite(θ_star_large[1])
        @test isfinite(θ_star_small[1])

        # The different data should lead to different posterior modes
        @test θ_star_large[1] != θ_star_small[1]
    end

    @testset "Consistency Across Runs" begin
        # Test that results are consistent across multiple runs
        spec = @hyperparams begin
            (ρ ~ Beta(2, 2), transform = logit, space = natural)
        end

        function consistent_latent(; ρ, kwargs...)
            n = 4
            Q = spdiagm(0 => ones(n), 1 => fill(-ρ, n - 1), -1 => fill(-ρ, n - 1))
            Q[1, 1] = 1 + ρ^2; Q[n, n] = 1 + ρ^2
            for i in 2:(n - 1)
                Q[i, i] = 1 + 2 * ρ^2
            end
            return (zeros(n), Symmetric(Q))
        end
        obs_model = ExponentialFamily(Bernoulli)
        model = LatentGaussianModel(spec, FunctionLatentModel(consistent_latent, 4), obs_model)

        y_test = [true, false, true, false]

        # Run multiple times (θ_star is now a NamedTuple in natural space)
        modes = NamedTuple[]
        for i in 1:3
            θ_star, _, _ = find_hyperparameter_mode(model, y_test)
            push!(modes, θ_star)
        end

        # All runs should give same result (within numerical tolerance)
        # Compare the values in the NamedTuples
        for i in 2:length(modes)
            @test modes[i].ρ ≈ modes[1].ρ atol = 1.0e-6
        end
    end

end
