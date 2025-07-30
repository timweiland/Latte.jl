using Test
using IntegratedNestedLaplace
using IntegratedNestedLaplace: CompositeObservations, CompositeObservationModel, CompositeLikelihood
using Distributions: Normal, Poisson

@testset "CompositeLikelihood evaluation" begin
    @testset "Basic loglik summation" begin
        gaussian_model = ExponentialFamily(Normal)
        poisson_model = ExponentialFamily(Poisson)
        composite_model = CompositeObservationModel((gaussian_model, poisson_model))

        y1 = [1.0, 2.0]  # 2 Gaussian observations
        y2 = [3, 4]      # 2 Poisson observations
        y_composite = CompositeObservations((y1, y2))

        composite_lik = composite_model(y_composite; σ = 1.0)

        # Both components target the full 2D latent field (since both y's are 2D)
        x = randn(2)
        ll_composite = loglik(composite_lik, x)

        # Should equal manual summation - both components see full x
        gaussian_lik = gaussian_model(y1; σ = 1.0)
        poisson_lik = poisson_model(y2)
        ll_manual = loglik(gaussian_lik, x) + loglik(poisson_lik, x)

        @test ll_composite ≈ ll_manual
    end

    @testset "Gradient summation" begin
        gaussian_model = ExponentialFamily(Normal)
        composite_model = CompositeObservationModel((gaussian_model, gaussian_model))

        y1 = [1.0, 2.0]  # 2 observations
        y2 = [3.0, 4.0]  # 2 observations
        y_composite = CompositeObservations((y1, y2))

        composite_lik = composite_model(y_composite; σ = 1.0)

        # Both components see the 2D latent field
        x = randn(2)
        grad_composite = loggrad(composite_lik, x)

        # Should equal sum of gradients - both components see full x
        lik1 = gaussian_model(y1; σ = 1.0)
        lik2 = gaussian_model(y2; σ = 1.0)
        grad_manual = loggrad(lik1, x) + loggrad(lik2, x)

        @test grad_composite ≈ grad_manual
    end

    @testset "Hessian summation" begin
        gaussian_model = ExponentialFamily(Normal)
        composite_model = CompositeObservationModel((gaussian_model, gaussian_model))

        y1 = [1.0]  # 1 observation -> 1D latent field
        y2 = [2.0]  # 1 observation -> 1D latent field
        y_composite = CompositeObservations((y1, y2))

        composite_lik = composite_model(y_composite; σ = 1.0)

        # Both components see the 1D latent field
        x = randn(1)
        hess_composite = loghessian(composite_lik, x)

        # Should equal sum of Hessians - both components see full x
        lik1 = gaussian_model(y1; σ = 1.0)
        lik2 = gaussian_model(y2; σ = 1.0)

        hess_manual = loghessian(lik1, x) + loghessian(lik2, x)

        @test hess_composite ≈ hess_manual
        @test size(hess_composite) == (1, 1)
    end

    @testset "Mixed likelihood types" begin
        # Test that different likelihood types can be combined
        gaussian_model = ExponentialFamily(Normal)
        poisson_model = ExponentialFamily(Poisson)
        composite_model = CompositeObservationModel((gaussian_model, poisson_model))

        y_composite = CompositeObservations(([1.0, 2.0], [3, 4]))  # Both 2D
        composite_lik = composite_model(y_composite; σ = 1.5)

        x = randn(2)  # 2D latent field to match observations

        # Should be able to evaluate with mixed types
        ll = loglik(composite_lik, x)
        @test ll isa Float64

        grad = loggrad(composite_lik, x)
        @test length(grad) == 2

        hess = loghessian(composite_lik, x)
        @test size(hess) == (2, 2)
    end

    @testset "Type stability" begin
        gaussian_model = ExponentialFamily(Normal)
        poisson_model = ExponentialFamily(Poisson)
        composite_model = CompositeObservationModel((gaussian_model, poisson_model))

        y_composite = CompositeObservations(([1.0], [2]))  # Both 1D -> 1D latent field
        composite_lik = composite_model(y_composite; σ = 1.0)

        x = randn(1)  # 1D latent field

        # All evaluation methods should be type stable
        @inferred loglik(composite_lik, x)
        @inferred loggrad(composite_lik, x)
        @inferred loghessian(composite_lik, x)
    end
end
