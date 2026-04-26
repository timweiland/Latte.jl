using Test
using Latte
using GaussianMarkovRandomFields
using LDLFactorizations
using Distributions
using LinearAlgebra
using SparseArrays
using Optim
using FiniteDiff

@testset "Mode Finding" begin

    @testset "Basic hyperparameter_logpdf" begin
        # Create a simple test model for consistent testing
        function create_simple_model(n = 10)
            # Single hyperparameter controlling latent field precision
            spec = @hyperparams begin
                (τ ~ Gamma(2, 1), transform = log, space = natural)
            end

            function simple_latent(; τ, kwargs...)
                Q = spdiagm(0 => fill(τ, n))  # White noise precision
                return (zeros(n), Q)
            end
            obs_model = ExponentialFamily(Bernoulli)  # No hyperparameters
            return LatentGaussianModel(spec, FunctionLatentModel(simple_latent, n), obs_model)
        end

        model = create_simple_model(5)
        y_test = [true, false, true, false, true]
        spec = model.hyperparameter_spec
        ws = make_workspace(model.latent_prior; τ = 1.0)

        # Test basic hyperparameter_logpdf evaluation
        θ_test_vec = [log(1.5)]  # Working space
        θ_test = WorkingHyperparameters(θ_test_vec, spec)
        logpdf_val = hyperparameter_logpdf(model, θ_test, y_test; ws = ws)
        @test isfinite(logpdf_val)

        # Test that function works at various points
        θ_low_vec = [log(0.5)]
        θ_low = WorkingHyperparameters(θ_low_vec, spec)
        logpdf_low = hyperparameter_logpdf(model, θ_low, y_test; ws = ws)
        @test isfinite(logpdf_low)
    end

    @testset "Optimality Conditions" begin
        # Create test model
        spec = @hyperparams begin
            (σ ~ InverseGamma(3, 2), transform = log, space = natural)
        end

        function precision_latent(; σ, kwargs...)
            n = 8
            Q = spdiagm(0 => fill(1 / σ^2, n))
            return (zeros(n), Q)
        end
        obs_model = ExponentialFamily(Normal)
        model = LatentGaussianModel(spec, FunctionLatentModel(precision_latent, 8), obs_model)

        # Generate test data
        y_test = randn(8)

        # Find mode
        θ_star, mode_points, mode_logdensities = find_hyperparameter_mode(model, y_test)

        @test θ_star isa WorkingHyperparameters
        @test length(θ_star) == 1  # WorkingHyperparameters with 1 free parameter

        # Convert to natural space to check the value
        θ_star_natural = convert(NaturalHyperparameters, θ_star)
        θ_star_nt = convert(NamedTuple, θ_star_natural)
        @test isfinite(θ_star_nt.σ)  # σ should be finite in natural space
        @test θ_star_nt.σ > 0  # σ must be positive in natural space

        # Test optimality condition: gradient should be ≈ 0 at mode
        # Use the working space vector for gradient computation
        θ_star_vec = θ_star.θ
        ws = make_workspace(model.latent_prior; σ = 1.0)
        function objective(θ_vec)
            θ_w = WorkingHyperparameters(θ_vec, spec)
            return hyperparameter_logpdf(model, θ_w, y_test; ws = ws)
        end

        grad_at_mode = FiniteDiff.finite_difference_gradient(objective, θ_star_vec)
        @test abs(grad_at_mode[1]) < 1.0e-3  # Gradient should be near zero

        # Test second-order condition: Hessian should be negative definite
        hess_at_mode = FiniteDiff.finite_difference_hessian(objective, θ_star_vec)
        @test hess_at_mode[1, 1] < 0  # Negative definite for 1D case

        # Test mode collection during optimization
        if mode_points !== nothing
            @test length(mode_points) > 0
            @test length(mode_points) == length(mode_logdensities)
            @test all(isfinite, mode_logdensities)
        end
    end

    @testset "Local optimality" begin
        spec = @hyperparams begin
            (τ ~ Gamma(2, 2), transform = log, space = natural)
        end

        function test_latent(; τ, kwargs...)
            n = 6
            Q = spdiagm(0 => fill(τ, n))
            return (zeros(n), Q)
        end
        obs_model = ExponentialFamily(Bernoulli)
        model = LatentGaussianModel(spec, FunctionLatentModel(test_latent, 6), obs_model)

        y_test = [true, true, false, true, false, false]

        # Find mode with default starting point
        θ_star1, _, _ = find_hyperparameter_mode(model, y_test)

        # Test that the mode is actually better than nearby points
        δ = 0.1
        θ_nearby = θ_star1 .+ δ  # Broadcasting preserves WorkingHyperparameters type
        ws = make_workspace(model.latent_prior; τ = 1.0)
        logpdf_mode = hyperparameter_logpdf(model, θ_star1, y_test; ws = ws)
        logpdf_nearby = hyperparameter_logpdf(model, θ_nearby, y_test; ws = ws)

        @test logpdf_mode >= logpdf_nearby  # Mode should be at least as good
    end

    @testset "Robustness and Edge Cases" begin
        # Test behavior with very peaked/flat posteriors
        spec = @hyperparams begin
            (λ ~ Exponential(1), transform = log, space = natural)
        end

        function exponential_latent(; λ, kwargs...)
            n = 3
            Q = spdiagm(0 => fill(λ + 1.0e-6, n))  # Add small regularization
            return (zeros(n), Q)
        end
        obs_model = ExponentialFamily(Bernoulli)
        model = LatentGaussianModel(spec, FunctionLatentModel(exponential_latent, 3), obs_model)

        # Test with moderate data (not too extreme to avoid numerical issues)
        y_moderate = [true, false, true]

        θ_star, _, _ = find_hyperparameter_mode(model, y_moderate)
        @test isfinite(θ_star[1])
        # θ_star is in working space (log scale)
        # Just verify we get a reasonable finite value
    end

    @testset "Initial hyperparameter guess" begin
        # Test the initial guess from HyperparameterSpec
        spec = @hyperparams begin
            (σ ~ InverseGamma(2, 1), transform = log, space = natural)
            (ρ ~ Beta(2, 2), transform = logit, space = natural)
            (λ ~ Exponential(1), transform = log, space = natural)
        end

        θ_init = initial_hyperparameter_guess(spec)

        @test length(θ_init) == 3
        @test all(isfinite, θ_init)
        # Initial guesses are working-space modes computed via Brent search
        # (see _working_space_mode_1d). Tolerance reflects Brent's accuracy.
        # Beta(2,2) under logit and Exp(1) under log both have working-space
        # modes at u = 0 by analytical argument.
        @test θ_init[1] < 0
        @test θ_init[2] ≈ 0 atol = 1.0e-6
        @test θ_init[3] ≈ 0 atol = 1.0e-6
    end

end
