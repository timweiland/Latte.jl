using Test
using LinearAlgebra
using Distributions
using StatsModels
using DataFrames
using GaussianMarkovRandomFields
using Latte

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

@testset "compute_reparameterization: non-PD AD Hessian → finite-difference fallback" begin
    # Regression: an RW2 + intercept Poisson model produces a non-positive-
    # definite second-order-AD Hessian at the mode (AD through the inner
    # Gaussian-approximation solve is unreliable here; the true curvature is
    # finite and positive). Clamping that to ~0 implied a near-infinite step in
    # working space, collapsing exploration to a single grid point and then
    # crashing the spline marginal build. compute_reparameterization retries
    # with finite differences, which recover the true curvature.
    quake_counts = [
        13, 14, 8, 10, 16, 26, 32, 27, 18, 32, 36, 24, 20, 23, 23, 18,
        12, 20, 22, 19, 13, 26, 13, 14, 22, 24, 21, 22, 26, 21, 23, 24, 20, 24, 24,
        22, 20, 10, 14, 19, 23, 18, 12, 13, 20, 26, 35, 14, 17, 19, 15, 18, 22, 22,
        17, 22, 15, 34, 10, 15, 22, 18, 15, 20, 13, 22, 23, 15, 21, 19, 20, 11, 20,
        13, 10, 8, 15, 18, 15, 9, 13, 13, 14, 9, 13, 16, 15, 8, 5, 11, 13, 7, 15, 12,
        23, 25, 22, 21, 20, 16, 14, 15, 13, 14, 17, 14, 11,
    ]
    eq_data = DataFrame(year = 1900:2006, quakes = quake_counts)
    rw2 = RandomWalk(2)
    hp = @hyperparams begin
        (τ_rw2 ~ PCPrior.Precision(1.0, α = 0.01), transform = log)
    end

    result = inla(
        @formula(quakes ~ 1 + rw2(year)), hp, eq_data;
        family = Poisson, progress = false
    )

    # Before the fix: exploration collapsed to 1 point and the spline build threw.
    @test length(result.exploration.grid_points) > 1
    @test all(isfinite ∘ mean, result.latent_marginals)
end
