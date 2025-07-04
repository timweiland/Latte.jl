using Test
using IntegratedNestedLaplace
using GaussianMarkovRandomFields
using LinearAlgebra
using SparseArrays
using Distributions

@testset "Gaussian Approximation" begin

    @testset "Fisher Scoring Step" begin
        # Set up simple test case: Normal observation model with known solution
        n = 10
        Q_prior = sparse(I, n, n)  # Unit precision
        μ_prior = zeros(n)
        prior_gmrf = GMRF(μ_prior, Q_prior, CholeskySolverBlueprint())

        obs_model = ExponentialFamily(Normal)
        θ_named = (σ = 1.0,)
        y = ones(n)  # All observations = 1

        current_μ = zeros(n)

        # Test Fisher scoring step
        options = NewtonOptions()
        μ_new, Q_new, Q_new_chol, step_stats = fisher_scoring_step(
            prior_gmrf, current_μ, obs_model, θ_named, y, options
        )

        # For Gaussian likelihood + Gaussian prior, should converge exactly in one step
        # Analytical solution: μ_posterior = (Q_prior + Q_obs)^(-1) * (Q_prior * μ_prior + Q_obs * y)
        # With Q_prior = I, μ_prior = 0, Q_obs = I/σ² = I, y = ones(n):
        # μ_posterior = (I + I)^(-1) * (0 + ones) = 0.5 * ones(n)
        @test μ_new ≈ fill(0.5, n) atol = 1.0e-12

        # New precision should be Q_prior + Q_obs = I + I = 2I
        @test Q_new ≈ 2.0 * I atol = 1.0e-12

        # Basic structure checks
        @test length(μ_new) == n
        @test size(Q_new) == (n, n)
        @test haskey(step_stats, :newton_decrement)
        @test haskey(step_stats, :step_size)
        @test haskey(step_stats, :gradient_norm)
    end

    @testset "Gaussian Approximation Convergence" begin
        # Simple case that should converge quickly
        n = 5
        Q_prior = sparse(2.0 * I, n, n)  # Strong prior
        μ_prior = zeros(n)
        prior_gmrf = GMRF(μ_prior, Q_prior, CholeskySolverBlueprint())

        obs_model = ExponentialFamily(Normal)
        θ_named = (σ = 0.5,)
        y = [0.1, 0.2, -0.1, 0.3, -0.2]  # Small deviations

        options = NewtonOptions(max_iterations = 20, verbose = false)
        result = gaussian_approximation(prior_gmrf, obs_model, θ_named, y; options = options)

        # Should converge
        @test result.converged
        @test result.iterations < 20
        @test length(result.μ) == n
        @test size(result.precision) == (n, n)
        @test length(result.stats) == result.iterations

        # Final gradient norm should be small
        @test result.stats[end].gradient_norm < 1.0e-5

        # Mode should be between prior mean and data
        @test all(abs.(result.μ) .< 1.0)  # Reasonable values
    end

    @testset "to_gmrf Conversion" begin
        # Create a simple result and convert to GMRF
        μ = [1.0, 2.0, 3.0]
        Q = sparse([2.0 -0.5 0; -0.5 2.0 -0.5; 0 -0.5 2.0])
        Q_chol = cholesky(Q)
        stats = [NewtonStats(1, 0.001, 0.01, 1.0e-6, true)]

        result = NewtonResult(μ, Q, Q_chol, stats, true, 1)
        gmrf = to_gmrf(result)

        @test isa(gmrf, GMRF)
        @test mean(gmrf) ≈ μ
        @test precision_matrix(gmrf) ≈ Q
    end

    @testset "Bernoulli Example" begin
        # Test with Bernoulli observation model (non-linear)
        n = 8
        Q_prior = sparse(Tridiagonal(fill(-0.5, n - 1), ones(n), fill(-0.5, n - 1)))
        μ_prior = zeros(n)
        prior_gmrf = GMRF(μ_prior, Q_prior, CholeskySolverBlueprint())

        obs_model = ExponentialFamily(Bernoulli)
        θ_named = NamedTuple()  # No hyperparameters
        y = [1, 1, 0, 1, 0, 0, 1, 0]  # Mixed binary data

        options = NewtonOptions(max_iterations = 50, verbose = false)
        result = gaussian_approximation(prior_gmrf, obs_model, θ_named, y; options = options)

        # Should converge (Bernoulli is well-behaved)
        @test result.converged
        @test result.iterations < 50

        # Mode should reflect the data pattern
        @test result.μ[1] > 0  # First observation is 1
        @test result.μ[3] < 0  # Third observation is 0

        # Precision matrix should be positive definite
        @test all(eigvals(Array(result.precision)) .> 0)
    end

    @testset "Zero-Step Convergence" begin
        # Test convergence in 0 iterations when starting exactly at posterior mode
        n = 3
        Q_prior = sparse(I, n, n)
        μ_prior = zeros(n)  # Prior mean = [0, 0, 0]
        prior_gmrf = GMRF(μ_prior, Q_prior, CholeskySolverBlueprint())

        obs_model = ExponentialFamily(Normal)
        θ_named = (σ = 1.0,)
        y = zeros(n)  # Observations = [0, 0, 0]

        # Posterior mode = (Q_prior + Q_obs)^(-1) * (Q_prior * μ_prior + Q_obs * y)
        # = (I + I)^(-1) * (I * 0 + I * 0) = 0
        # So posterior mode = prior mean = [0, 0, 0]

        options = NewtonOptions(max_iterations = 10, verbose = false)
        result = gaussian_approximation(prior_gmrf, obs_model, θ_named, y; options = options)

        # Should converge in exactly 1 iteration via early convergence
        @test result.converged
        @test result.iterations == 1
        @test length(result.stats) == 1

        # Should stay at zero (posterior mode = prior mean)
        @test result.μ ≈ zeros(n) atol = 1.0e-12

        # Early convergence should be detected (gradient is zero from start)
        @test result.stats[1].gradient_norm < options.tol_gradient
        @test result.stats[1].newton_decrement == 0.0
        @test result.stats[1].step_size == 0.0
    end

    @testset "Max Iterations Limit" begin
        # Test that max_iterations parameter is respected
        n = 5
        Q_prior = sparse(I, n, n)
        μ_prior = zeros(n)
        prior_gmrf = GMRF(μ_prior, Q_prior, CholeskySolverBlueprint())

        obs_model = ExponentialFamily(Bernoulli)
        θ_named = NamedTuple()
        y = [1, 0, 1, 0, 1]  # Alternating pattern

        # Force early termination
        options = NewtonOptions(max_iterations = 1, verbose = false)
        result = gaussian_approximation(prior_gmrf, obs_model, θ_named, y; options = options)

        # Should stop at max iterations (may or may not converge in 1 step)
        @test result.iterations <= 1
        @test length(result.stats) <= 1
    end
end
