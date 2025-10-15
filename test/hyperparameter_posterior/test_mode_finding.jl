using Test
using IntegratedNestedLaplace
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
            hp_prior = HyperparameterPrior((τ = Gamma(2, 1),))

            function simple_latent(θ_named)
                τ = θ_named.τ
                Q = spdiagm(0 => fill(τ, n))  # White noise precision
                return GMRF(zeros(n), Q)
            end

            obs_model = ExponentialFamily(Bernoulli)  # No hyperparameters
            return INLAModel(hp_prior, simple_latent, obs_model)
        end

        model = create_simple_model(5)
        y_test = [true, false, true, false, true]

        # Test basic hyperparameter_logpdf evaluation
        θ_test = [1.5]
        logpdf_val = hyperparameter_logpdf(model, θ_test, y_test)
        @test isfinite(logpdf_val)

        # Test that function returns -Inf outside support
        θ_negative = [-0.5]  # Gamma distribution has support (0, ∞)
        logpdf_negative = hyperparameter_logpdf(model, θ_negative, y_test)
        @test logpdf_negative == -Inf
    end

    @testset "Optimality Conditions" begin
        # Create test model
        hp_prior = HyperparameterPrior((σ = InverseGamma(3, 2),))

        function precision_latent(θ_named)
            σ = θ_named.σ
            n = 8
            Q = spdiagm(0 => fill(1 / σ^2, n))
            return GMRF(zeros(n), Q)
        end

        obs_model = ExponentialFamily(Normal)
        model = INLAModel(hp_prior, precision_latent, obs_model)

        # Generate test data
        y_test = randn(8)

        # Find mode
        θ_star, mode_points, mode_logdensities = find_hyperparameter_mode(model, y_test)

        @test length(θ_star) == 1
        @test isfinite(θ_star[1])
        @test θ_star[1] > 0  # Should be in support of InverseGamma

        # Test optimality condition: gradient should be ≈ 0 at mode
        function objective(θ)
            return hyperparameter_logpdf(model, θ, y_test)
        end

        grad_at_mode = FiniteDiff.finite_difference_gradient(objective, θ_star)
        @test abs(grad_at_mode[1]) < 1.0e-3  # Gradient should be near zero

        # Test second-order condition: Hessian should be negative definite
        hess_at_mode = FiniteDiff.finite_difference_hessian(objective, θ_star)
        @test hess_at_mode[1, 1] < 0  # Negative definite for 1D case

        # Test mode collection during optimization
        if mode_points !== nothing
            @test length(mode_points) > 0
            @test length(mode_points) == length(mode_logdensities)
            @test all(isfinite, mode_logdensities)
        end
    end

    @testset "Local optimality" begin
        hp_prior = HyperparameterPrior((τ = Gamma(2, 2),))

        function test_latent(θ_named)
            τ = θ_named.τ
            n = 6
            Q = spdiagm(0 => fill(τ, n))
            return GMRF(zeros(n), Q)
        end

        obs_model = ExponentialFamily(Bernoulli)
        model = INLAModel(hp_prior, test_latent, obs_model)

        y_test = [true, true, false, true, false, false]

        # Find mode with default starting point
        θ_star1, _, _ = find_hyperparameter_mode(model, y_test)

        # Test that the mode is actually better than nearby points
        δ = 0.1
        θ_nearby = [θ_star1[1] + δ]
        logpdf_mode = hyperparameter_logpdf(model, θ_star1, y_test)
        logpdf_nearby = hyperparameter_logpdf(model, θ_nearby, y_test)

        @test logpdf_mode >= logpdf_nearby  # Mode should be at least as good
    end

    @testset "Robustness and Edge Cases" begin
        # Test behavior with very peaked/flat posteriors
        hp_prior = HyperparameterPrior((λ = Exponential(1),))

        function exponential_latent(θ_named)
            λ = θ_named.λ
            n = 3
            Q = spdiagm(0 => fill(λ + 1.0e-6, n))  # Add small regularization
            return GMRF(zeros(n), Q)
        end

        obs_model = ExponentialFamily(Bernoulli)
        model = INLAModel(hp_prior, exponential_latent, obs_model)

        # Test with extreme data
        y_extreme = [true, true, true]  # All successes

        θ_star, _, _ = find_hyperparameter_mode(model, y_extreme)
        @test isfinite(θ_star[1])
        @test θ_star[1] ≈ 0.0 atol = 1.0e-3  # Should be near boundary for extreme data
    end

    @testset "Initial hyperparameter guess" begin
        # Test the basic mode computation from Product distributions
        prior_product = product_distribution([InverseGamma(2, 1), Beta(2, 2)])
        mode_vec = IntegratedNestedLaplace.initial_hyperparameter_guess(prior_product)

        @test length(mode_vec) == 2
        @test mode_vec[1] ≈ mode(InverseGamma(2, 1))
        @test mode_vec[2] ≈ mode(Beta(2, 2))
    end

end
