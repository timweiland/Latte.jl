using Test
using IntegratedNestedLaplace
using GaussianMarkovRandomFields
using LinearAlgebra
using SparseArrays
using Distributions

@testset "Gaussian Approximation" begin

    @testset "Gaussian Likelihood - Analytical Solution" begin
        # For Gaussian likelihood, the Gaussian approximation should be exact
        n = 5
        Q_prior = sparse(2.0 * I, n, n)  # Strong prior
        μ_prior = zeros(n)
        prior_gmrf = GMRF(μ_prior, Q_prior, CholeskySolverBlueprint())

        obs_model = ExponentialFamily(Normal)
        θ_named = (σ = 0.5,)
        y = [0.1, 0.2, -0.1, 0.3, -0.2]  # Small deviations

        obs_lik = obs_model(y; θ_named...)
        result = gaussian_approximation(prior_gmrf, obs_lik)

        # Should return a valid GMRF
        @test result isa GMRF
        @test length(mean(result)) == n
        @test size(precision_matrix(result)) == (n, n)

        # For Gaussian case, can verify against analytical solution
        # Posterior precision: Q_post = Q_prior + Q_obs = Q_prior + I/σ²
        σ = θ_named.σ
        Q_obs = sparse(I, n, n) / σ^2
        Q_analytical = Q_prior + Q_obs

        # Posterior mean: μ_post = Q_post^(-1) * (Q_prior * μ_prior + Q_obs * y)
        μ_analytical = Q_analytical \ (Q_prior * μ_prior + Q_obs * y)

        @test precision_matrix(result) ≈ Q_analytical atol = 1.0e-10
        @test mean(result) ≈ μ_analytical atol = 1.0e-10
    end

    @testset "Bernoulli Likelihood - Mathematical Properties" begin
        # Test with Bernoulli observation model (non-linear)
        n = 8
        Q_prior = sparse(Tridiagonal(fill(-0.5, n - 1), ones(n), fill(-0.5, n - 1)))
        μ_prior = zeros(n)
        prior_gmrf = GMRF(μ_prior, Q_prior, CholeskySolverBlueprint())

        obs_model = ExponentialFamily(Bernoulli)
        θ_named = NamedTuple()  # No hyperparameters
        y = [1, 1, 0, 1, 0, 0, 1, 0]  # Mixed binary data

        obs_lik = obs_model(y; θ_named...)
        result = gaussian_approximation(prior_gmrf, obs_lik)

        # Should return a valid GMRF
        @test result isa GMRF
        @test length(mean(result)) == n
        @test size(precision_matrix(result)) == (n, n)

        # Mode should reflect the data pattern
        μ_result = mean(result)
        @test μ_result[1] > 0  # First observation is 1
        @test μ_result[3] < 0  # Third observation is 0

        # Precision matrix should be positive definite
        @test all(eigvals(Array(precision_matrix(result))) .> 0)
    end

    @testset "Poisson Likelihood - Mathematical Properties" begin
        # Test with Poisson observation model
        n = 6
        Q_prior = sparse(I, n, n)
        μ_prior = zeros(n)
        prior_gmrf = GMRF(μ_prior, Q_prior, CholeskySolverBlueprint())

        obs_model = ExponentialFamily(Poisson)
        θ_named = NamedTuple()  # No hyperparameters
        y = [1, 3, 0, 2, 4, 1]  # Count data

        obs_lik = obs_model(y; θ_named...)
        result = gaussian_approximation(prior_gmrf, obs_lik)

        # Should return a valid GMRF
        @test result isa GMRF
        @test length(mean(result)) == n
        @test size(precision_matrix(result)) == (n, n)

        # Mode should be reasonable for Poisson data
        μ_result = mean(result)
        @test all(isfinite.(μ_result))

        # For Poisson with log link, mode should reflect log of data pattern
        # Higher counts should correspond to higher modes
        @test μ_result[5] > μ_result[3]  # y[5]=4 > y[3]=0
        @test μ_result[2] > μ_result[3]  # y[2]=3 > y[3]=0

        # Precision matrix should be positive definite
        @test all(eigvals(Array(precision_matrix(result))) .> 0)
    end

    @testset "Prior-Posterior Consistency" begin
        # Test that approximation is reasonable relative to prior
        n = 4
        Q_prior = sparse(I, n, n)
        μ_prior = [1.0, -1.0, 0.5, -0.5]
        prior_gmrf = GMRF(μ_prior, Q_prior, CholeskySolverBlueprint())

        obs_model = ExponentialFamily(Normal)
        θ_named = (σ = 2.0,)  # Weak likelihood
        y = μ_prior + 0.1 * randn(n)  # Data close to prior mean

        obs_lik = obs_model(y; θ_named...)
        result = gaussian_approximation(prior_gmrf, obs_lik)

        # Should return a valid GMRF
        @test result isa GMRF

        # With weak likelihood, posterior should be close to prior
        μ_result = mean(result)
        @test norm(μ_result - μ_prior) < 0.5  # Should be reasonably close

        # Posterior precision should be larger than prior precision
        @test all(diag(precision_matrix(result)) .>= diag(Q_prior))
    end
end
