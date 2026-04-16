using Test
using LinearAlgebra
using Distributions
using IntegratedNestedLaplace

@testset "ReparameterizationTransform Tests" begin
    # Setup test data — ReparameterizationTransform expects WorkingHyperparameters
    # as θ_star, so wrap the raw mode vector in a minimal 2-parameter spec.
    spec = HyperparameterSpec(
        free = (
            a = Hyperparameter(Normal(0, 1)),
            b = Hyperparameter(Normal(0, 1)),
        )
    )
    θ_star = WorkingHyperparameters([1.0, 2.0], spec)
    V = [1.0 0.0; 0.0 1.0]  # Identity matrix
    Λ_inv_sqrt = Diagonal([0.5, 0.8])
    H = [4.0 0.0; 0.0 1.5625]

    transform = ReparameterizationTransform(θ_star, V, Λ_inv_sqrt, H)

    @testset "Callable Interface" begin
        # Test with zero vector (should return mode)
        @test transform([0.0, 0.0]) ≈ θ_star

        # Test transformation formula: θ = θ_star + V * Λ_inv_sqrt * z
        z = [1.0, 1.0]
        expected = θ_star .+ V * Λ_inv_sqrt * z
        @test transform(z) ≈ expected
        @test transform(z) ≈ [1.5, 2.8]  # Manual calculation
    end

    @testset "logdet_jacobian" begin
        # For diagonal Λ_inv_sqrt, logdet is sum of log of diagonal elements
        expected = log(0.5) + log(0.8)
        @test logdet_jacobian(transform) ≈ expected
    end

    @testset "Type Stability" begin
        @inferred transform([1.0, 1.0])
        @inferred logdet_jacobian(transform)
    end
end
