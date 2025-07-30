using Test
using IntegratedNestedLaplace
using IntegratedNestedLaplace: CompositeObservations, CompositeObservationModel, CompositeLikelihood
using Distributions: Normal, Poisson

@testset "CompositeObservationModel" begin
    @testset "Constructor" begin
        # Create observation models for testing
        gaussian_model = ExponentialFamily(Normal)
        poisson_model = ExponentialFamily(Poisson)

        composite_model = CompositeObservationModel((gaussian_model, poisson_model))
        @test composite_model isa CompositeObservationModel
        @test length(composite_model.components) == 2
    end

    @testset "Factory pattern - callable interface" begin
        gaussian_model = ExponentialFamily(Normal)
        poisson_model = ExponentialFamily(Poisson)
        composite_model = CompositeObservationModel((gaussian_model, poisson_model))

        # Test materialization
        y1 = randn(3)
        y2 = rand(1:10, 2)
        y_composite = CompositeObservations((y1, y2))

        # Should create CompositeLikelihood when called
        composite_lik = composite_model(y_composite; σ = 1.5)
        @test composite_lik isa CompositeLikelihood
    end

    @testset "Validation" begin
        gaussian_model = ExponentialFamily(Normal)

        # Should validate component count matches observation count
        y_mismatch = CompositeObservations(([1.0], [2.0], [3.0]))  # 3 components
        composite_model = CompositeObservationModel((gaussian_model,))  # 1 component

        @test_throws ArgumentError composite_model(y_mismatch; σ = 1.0)
    end

    @testset "Hyperparameter distribution" begin
        # Test that hyperparameters get distributed correctly to components
        gaussian_model = ExponentialFamily(Normal)
        composite_model = CompositeObservationModel((gaussian_model, gaussian_model))

        y1 = [1.0, 2.0]
        y2 = [3.0, 4.0]
        y_composite = CompositeObservations((y1, y2))

        # Both components should receive σ parameter
        composite_lik = composite_model(y_composite; σ = 2.0)
        @test composite_lik isa CompositeLikelihood

        # Test with mixed hyperparameters (some components won't use all)
        poisson_model = ExponentialFamily(Poisson)
        mixed_model = CompositeObservationModel((gaussian_model, poisson_model))
        mixed_composite = CompositeObservations(([1.0], [2]))

        # Poisson should ignore σ, Gaussian should use it
        mixed_lik = mixed_model(mixed_composite; σ = 1.0, unused_param = 999)
        @test mixed_lik isa CompositeLikelihood
    end
end
