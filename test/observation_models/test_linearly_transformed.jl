using Test
using LinearAlgebra
using ForwardDiff
using Distributions

using IntegratedNestedLaplace
using IntegratedNestedLaplace: LinearlyTransformedObservationModel, LinearlyTransformedLikelihood

@testset "LinearlyTransformedObservationModel" begin

    @testset "Basic Construction" begin
        base_model = ExponentialFamily(Poisson)
        A = [1.0 0.5; 1.0 1.0]  # 2 obs, 2 latent components

        ltom = LinearlyTransformedObservationModel(base_model, A)
        @test ltom.base_model === base_model
        @test ltom.design_matrix === A

        # Test hyperparameter delegation
        @test hyperparameters(ltom) == ()  # Poisson has no hyperparameters
    end

    @testset "Materialization" begin
        base_model = ExponentialFamily(Normal)
        A = [1.0 0.0; 0.0 1.0]  # Identity
        ltom = LinearlyTransformedObservationModel(base_model, A)

        y = [1.0, 2.0]
        ltlik = ltom(y; σ = 1.0)

        @test ltlik isa LinearlyTransformedLikelihood
        @test ltlik.design_matrix === A
    end

    @testset "Chain Rule Verification" begin
        base_model = ExponentialFamily(Normal)
        A = [1.0 0.5; 0.0 1.0]
        ltom = LinearlyTransformedObservationModel(base_model, A)

        y = [1.0, 2.0]
        ltlik = ltom(y; σ = 1.0)
        x_full = [0.5, 1.0]

        # Verify chain rule with ForwardDiff
        grad_fd = ForwardDiff.gradient(x -> loglik(ltlik, x), x_full)
        grad_analytical = loggrad(ltlik, x_full)
        @test grad_analytical ≈ grad_fd

        hess_fd = ForwardDiff.hessian(x -> loglik(ltlik, x), x_full)
        hess_analytical = loghessian(ltlik, x_full)
        @test hess_analytical ≈ hess_fd
    end

    @testset "Identity Matrix Equivalence" begin
        # When A = I, should match base model exactly
        base_model = ExponentialFamily(Poisson)
        A = Matrix{Float64}(I, 2, 2)
        ltom = LinearlyTransformedObservationModel(base_model, A)

        y = [1, 3]
        ltlik = ltom(y)
        base_lik = base_model(y)

        x = [0.5, 1.0]
        @test loglik(ltlik, x) ≈ loglik(base_lik, x)
    end

end
