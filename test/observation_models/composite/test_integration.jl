using Test
using IntegratedNestedLaplace
using IntegratedNestedLaplace: CompositeObservations, CompositeObservationModel, CompositeLikelihood
using Distributions: Normal, Poisson

@testset "Composite Likelihood Integration Tests" begin
    @testset "End-to-end workflow with indexed models" begin
        # Create indexed observation models for different parts of latent field
        gaussian_model = ExponentialFamily(Normal, indices = 1:3)    # First 3 elements
        poisson_model = ExponentialFamily(Poisson, indices = 4:6)    # Next 3 elements

        # Create composite model
        composite_model = CompositeObservationModel((gaussian_model, poisson_model))

        # Prepare observation data
        y_gaussian = [1.0, 2.0, 1.5]  # 3 Gaussian observations
        y_poisson = [2, 3, 1]         # 3 Poisson observations
        y_composite = CompositeObservations((y_gaussian, y_poisson))

        # Materialize composite likelihood
        composite_lik = composite_model(y_composite; σ = 0.8)

        # Test evaluation on 6D latent field
        x = [0.9, 2.1, 1.4, log(2.2), log(2.9), log(1.1)]  # First 3 for Gaussian, last 3 (log-scale) for Poisson

        # Should evaluate correctly
        ll = loglik(composite_lik, x)
        @test ll isa Float64

        # Compare against manual computation
        gaussian_lik = gaussian_model(y_gaussian; σ = 0.8)
        poisson_lik = poisson_model(y_poisson)

        ll_manual = loglik(gaussian_lik, x) + loglik(poisson_lik, x)
        @test ll ≈ ll_manual

        # Test gradient
        grad = loggrad(composite_lik, x)
        @test length(grad) == 6

        grad_manual = loggrad(gaussian_lik, x) + loggrad(poisson_lik, x)
        @test grad ≈ grad_manual

        # Test Hessian
        hess = loghessian(composite_lik, x)
        @test size(hess) == (6, 6)

        hess_manual = loghessian(gaussian_lik, x) + loghessian(poisson_lik, x)
        @test hess ≈ hess_manual
    end

    @testset "Overlapping indices" begin
        # Test overlapping case with same likelihood type (so same hyperparameters work)
        model1 = ExponentialFamily(Normal, indices = 1:3)      # First 3 elements
        model2 = ExponentialFamily(Normal, indices = 2:4)      # Elements 2-4 (overlap on 2,3)

        composite_model = CompositeObservationModel((model1, model2))

        # Different observations for each component
        y1 = [1.0, 1.5, 2.0]
        y2 = [1.4, 2.1, 2.6]
        y_composite = CompositeObservations((y1, y2))

        # Materialize - both components use same σ
        composite_lik = composite_model(y_composite; σ = 1.0)

        # Evaluate on 4D latent field
        x = [1.0, 1.5, 2.0, 2.5]
        ll = loglik(composite_lik, x)

        # Manual computation: both models contribute, overlap adds up
        lik1 = model1(y1; σ = 1.0)
        lik2 = model2(y2; σ = 1.0)

        ll_manual = loglik(lik1, x) + loglik(lik2, x)
        @test ll ≈ ll_manual

        # Test gradient accumulation at overlapping indices
        grad = loggrad(composite_lik, x)
        grad_manual = loggrad(lik1, x) + loggrad(lik2, x)
        @test grad ≈ grad_manual
    end

    @testset "Performance: composite vs manual summation" begin
        # Test that composite likelihood has minimal overhead
        gaussian_model = ExponentialFamily(Normal, indices = 1:2)
        poisson_model = ExponentialFamily(Poisson, indices = 3:4)

        composite_model = CompositeObservationModel((gaussian_model, poisson_model))
        y_composite = CompositeObservations(([1.0, 2.0], [3, 4]))
        composite_lik = composite_model(y_composite; σ = 1.0)

        # Individual likelihoods for comparison
        gaussian_lik = gaussian_model([1.0, 2.0]; σ = 1.0)
        poisson_lik = poisson_model([3, 4])

        x = randn(4)

        # Both should give same results
        ll_composite = loglik(composite_lik, x)
        ll_manual = loglik(gaussian_lik, x) + loglik(poisson_lik, x)
        @test ll_composite ≈ ll_manual

        grad_composite = loggrad(composite_lik, x)
        grad_manual = loggrad(gaussian_lik, x) + loggrad(poisson_lik, x)
        @test grad_composite ≈ grad_manual
    end

    @testset "Type stability" begin
        # Ensure all operations are type stable
        gaussian_model = ExponentialFamily(Normal, indices = 1:2)
        poisson_model = ExponentialFamily(Poisson, indices = 3:4)

        composite_model = CompositeObservationModel((gaussian_model, poisson_model))
        y_composite = CompositeObservations(([1.0, 2.0], [3, 4]))
        composite_lik = composite_model(y_composite; σ = 1.0)

        x = randn(4)

        # All operations should be type stable
        @inferred loglik(composite_lik, x)
        @inferred loggrad(composite_lik, x)
        @inferred loghessian(composite_lik, x)
    end
end
