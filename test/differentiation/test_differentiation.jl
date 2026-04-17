using Test
using IntegratedNestedLaplace
using IntegratedNestedLaplace: ad_negative_hessian, pmap_executor
using GaussianMarkovRandomFields
using Distributions
using SparseArrays
using LinearAlgebra
using FiniteDiff
using ForwardDiff
using DifferentiationInterface
using ADTypes

@testset "DifferentiationStrategy" begin

    @testset "Type hierarchy" begin
        @test FiniteDiffStrategy <: DifferentiationStrategy
        @test ADStrategy <: DifferentiationStrategy

        # Default constructor uses AutoForwardDiff
        ad = ADStrategy()
        @test ad isa ADStrategy{<:AutoForwardDiff}
        @test ad.backend isa AutoForwardDiff

        # Custom backend
        ad_fd = ADStrategy(AutoFiniteDiff())
        @test ad_fd isa ADStrategy{<:AutoFiniteDiff}
    end

    @testset "ad_negative_hessian" begin
        # Quadratic test function with known Hessian: f(x) = -0.5 * x' * A * x
        # Hessian = -A, so negative Hessian = A
        A = [
            4.0 1.0 0.5;
            1.0 3.0 0.2;
            0.5 0.2 2.0
        ]
        f(x) = -0.5 * dot(x, A * x)
        x0 = zeros(3)

        @testset "matches known Hessian" begin
            H = ad_negative_hessian(f, x0, AutoForwardDiff())
            @test H ≈ A atol = 1.0e-4
        end

        @testset "matches adaptive_negative_hessian" begin
            using IntegratedNestedLaplace: adaptive_negative_hessian
            H_ad = ad_negative_hessian(f, x0, AutoForwardDiff())
            H_fd = adaptive_negative_hessian(f, x0)
            @test H_ad ≈ H_fd atol = 0.01
        end

        @testset "works with executor" begin
            H_seq = ad_negative_hessian(
                f, x0, AutoForwardDiff();
                executor = SequentialExecutor()
            )
            H_par = ad_negative_hessian(
                f, x0, AutoForwardDiff();
                executor = ThreadedExecutor(nworkers = 2)
            )
            @test H_par ≈ H_seq atol = 1.0e-10
        end

        @testset "1D case" begin
            g(x) = -2.0 * x[1]^2
            H = ad_negative_hessian(g, [0.0], AutoForwardDiff())
            @test H ≈ [4.0;;] atol = 1.0e-4
        end

        @testset "non-zero evaluation point" begin
            B = [3.0 0.5; 0.5 2.0]
            h(x) = -0.5 * dot(x, B * x) + sum(x)
            x1 = [1.0, -0.5]
            H = ad_negative_hessian(h, x1, AutoForwardDiff())
            @test H ≈ B atol = 1.0e-4
        end
    end

    @testset "AD gradient for mode finding" begin
        # Set up a simple model
        n = 8
        spec = @hyperparams begin
            (σ ~ InverseGamma(3, 2), transform = log, space = natural)
        end

        function precision_latent(; σ, kwargs...)
            Q = spdiagm(0 => fill(1 / σ^2, n))
            return (zeros(n), Q)
        end
        model = INLAModel(spec, FunctionLatentModel(precision_latent, n), ExponentialFamily(Normal))
        y_test = randn(n)

        @testset "ADStrategy finds mode" begin
            θ_star_ad, _, _ = find_hyperparameter_mode(
                model, y_test;
                diff_strategy = ADStrategy()
            )
            @test θ_star_ad isa WorkingHyperparameters
            @test length(θ_star_ad) == 1
            @test isfinite(θ_star_ad[1])

            # Check optimality: gradient should be ≈ 0 at mode
            ws = make_workspace(model.latent_prior; σ = 1.0)
            objective(θ_vec) = -hyperparameter_logpdf(
                model, WorkingHyperparameters(θ_vec, spec), y_test; ws = ws
            )
            grad = ForwardDiff.gradient(objective, θ_star_ad.θ)
            @test norm(grad) < 1.0e-3
        end

        @testset "ADStrategy and FiniteDiffStrategy find same mode" begin
            θ_star_ad, _, _ = find_hyperparameter_mode(
                model, y_test;
                diff_strategy = ADStrategy()
            )
            θ_star_fd, _, _ = find_hyperparameter_mode(
                model, y_test;
                diff_strategy = FiniteDiffStrategy()
            )

            @test θ_star_ad.θ ≈ θ_star_fd.θ atol = 1.0e-2
        end
    end

    @testset "AD Hessian in compute_reparameterization" begin
        using IntegratedNestedLaplace: compute_reparameterization, find_hyperparameter_mode

        n = 6
        spec = @hyperparams begin
            (σ ~ InverseGamma(2, 1), transform = log, space = natural)
        end
        function latent(; σ, kwargs...)
            Q = spdiagm(0 => fill(1 / σ^2, n))
            return (zeros(n), Q)
        end
        model = INLAModel(spec, FunctionLatentModel(latent, n), ExponentialFamily(Normal))
        y = randn(n)
        θ_star, _, _ = find_hyperparameter_mode(model, y)
        ws = make_workspace(model.latent_prior; σ = 1.0)

        @testset "ADStrategy produces valid reparameterization" begin
            t = compute_reparameterization(
                model, y, θ_star;
                ws = ws, diff_strategy = ADStrategy(),
            )
            @test all(isfinite, t.H)
            @test issymmetric(t.H) || t.H ≈ t.H'
        end

        @testset "ADStrategy matches FiniteDiffStrategy" begin
            t_ad = compute_reparameterization(
                model, y, θ_star;
                ws = ws, diff_strategy = ADStrategy(),
            )
            t_fd = compute_reparameterization(
                model, y, θ_star;
                ws = ws, diff_strategy = FiniteDiffStrategy(),
            )
            @test t_ad.H ≈ t_fd.H atol = 0.1
            @test t_ad.V ≈ t_fd.V atol = 0.1
        end

        @testset "ADStrategy with executor" begin
            t_seq = compute_reparameterization(
                model, y, θ_star;
                ws = ws, diff_strategy = ADStrategy(), executor = SequentialExecutor(),
            )
            t_par = compute_reparameterization(
                model, y, θ_star;
                ws = ws, diff_strategy = ADStrategy(), executor = ThreadedExecutor(nworkers = 2),
            )
            @test t_par.H ≈ t_seq.H atol = 1.0e-10
        end
    end

    @testset "inla() accepts diff_strategy" begin
        n = 10
        spec = @hyperparams begin
            (τ ~ Gamma(2, 1), transform = log, space = natural)
        end
        model = INLAModel(
            spec,
            FunctionLatentModel((; τ, kwargs...) -> (zeros(n), spdiagm(0 => fill(τ, n))), n),
            ExponentialFamily(Bernoulli)
        )
        y = rand(Bool, n)

        # Should run without error with both strategies
        result_ad = inla(model, y; diff_strategy = ADStrategy(), accumulators = ())
        result_fd = inla(model, y; diff_strategy = FiniteDiffStrategy(), accumulators = ())

        @test result_ad isa INLAResult
        @test result_fd isa INLAResult
    end
end
