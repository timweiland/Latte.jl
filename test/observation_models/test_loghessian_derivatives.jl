using Test
using Latte
using Latte: loghessian_directional_derivative
using GaussianMarkovRandomFields
using LinearAlgebra
using SparseArrays
using ForwardDiff
using FiniteDiff
using Distributions
using Random

@testset "Loghessian Directional Derivative" begin

    Random.seed!(42)

    # Helper function for finite difference validation
    function finite_diff_loghessian_derivative(x0, v, obs_lik; h = 1.0e-6)
        # Compute (H(x0 + h*v) - H(x0)) / h
        H0 = loghessian(x0, obs_lik)
        H1 = loghessian(x0 + h * v, obs_lik)
        return (H1 - H0) / h
    end

    # Helper function for ForwardDiff fallback
    function forwarddiff_fallback(x0, v, obs_lik)
        loghessian_path = t -> loghessian(x0 + t * v, obs_lik)
        return ForwardDiff.derivative(loghessian_path, 0.0)
    end

    @testset "Poisson Likelihood" begin
        n = 10
        y = rand(1:20, n)
        obs_model = ExponentialFamily(Poisson)
        obs_lik = obs_model(y)

        x0 = randn(n)
        v = randn(n)

        dH_specialized = loghessian_directional_derivative(x0, v, obs_lik)
        dH_fallback = forwarddiff_fallback(x0, v, obs_lik)
        dH_finite = finite_diff_loghessian_derivative(x0, v, obs_lik)

        @test dH_specialized isa Diagonal
        @test size(dH_specialized) == (n, n)
        @test dH_specialized ≈ dH_fallback rtol = 1.0e-10
        @test dH_specialized ≈ dH_finite atol = 1.0e-4

        # Test with zero direction
        dH_zero = loghessian_directional_derivative(x0, zeros(n), obs_lik)
        @test all(diag(dH_zero) .≈ 0)
    end

    @testset "Bernoulli Likelihood" begin
        n = 10
        y = rand(0:1, n)
        obs_model = ExponentialFamily(Bernoulli)
        obs_lik = obs_model(y)

        x0 = randn(n)
        v = randn(n)

        dH_specialized = loghessian_directional_derivative(x0, v, obs_lik)
        dH_fallback = forwarddiff_fallback(x0, v, obs_lik)
        dH_finite = finite_diff_loghessian_derivative(x0, v, obs_lik)

        @test dH_specialized isa Diagonal
        @test size(dH_specialized) == (n, n)
        @test dH_specialized ≈ dH_fallback rtol = 1.0e-10
        @test dH_specialized ≈ dH_finite atol = 1.0e-4

        # Test with extreme values to check numerical stability
        x_extreme = [10.0, -10.0, 0.0]
        v_extreme = ones(3)
        y_extreme = [1, 0, 1]
        obs_lik_extreme = ExponentialFamily(Bernoulli)(y_extreme)

        dH_extreme = loghessian_directional_derivative(x_extreme, v_extreme, obs_lik_extreme)
        @test all(isfinite.(diag(dH_extreme)))

        # Validate against fallback even for extreme values
        dH_extreme_fallback = forwarddiff_fallback(x_extreme, v_extreme, obs_lik_extreme)
        @test dH_extreme ≈ dH_extreme_fallback rtol = 1.0e-10
    end

    @testset "Binomial Likelihood" begin
        n = 10
        n_trials = rand(5:20, n)
        y_successes = [min(rand(0:nt), nt) for nt in n_trials]
        y = BinomialObservations(y_successes, n_trials)

        obs_model = ExponentialFamily(Binomial)
        obs_lik = obs_model(y)

        x0 = randn(n)
        v = randn(n)

        dH_specialized = loghessian_directional_derivative(x0, v, obs_lik)
        dH_fallback = forwarddiff_fallback(x0, v, obs_lik)
        dH_finite = finite_diff_loghessian_derivative(x0, v, obs_lik)

        @test dH_specialized isa Diagonal
        @test size(dH_specialized) == (n, n)
        @test dH_specialized ≈ dH_fallback rtol = 1.0e-10
        @test dH_specialized ≈ dH_finite atol = 1.0e-4
    end

    @testset "Normal Likelihood" begin
        n = 10
        y = randn(n)
        σ = 1.5
        obs_model = ExponentialFamily(Normal)
        obs_lik = obs_model(y; σ = σ)

        x0 = randn(n)
        v = randn(n)

        dH_specialized = loghessian_directional_derivative(x0, v, obs_lik)
        dH_fallback = forwarddiff_fallback(x0, v, obs_lik)
        dH_finite = finite_diff_loghessian_derivative(x0, v, obs_lik)

        @test dH_specialized isa Diagonal
        @test size(dH_specialized) == (n, n)
        @test dH_specialized ≈ dH_fallback rtol = 1.0e-10
        @test dH_specialized ≈ dH_finite atol = 1.0e-4

        # For Normal, Hessian is constant so derivative should be zero
        @test all(diag(dH_specialized) .≈ 0)
    end

    # Note: AutoDiffLikelihood test removed due to nested ForwardDiff issues
    # The fallback implementation is tested via finite differences in other tests

    @testset "LinearlyTransformedLikelihood" begin
        # Test the specialized implementation for linearly transformed likelihoods
        n_latent = 8
        n_obs = 5

        # Create a sparse random design matrix (typical use case)
        Random.seed!(123)
        A = sprandn(n_obs, n_latent, 0.3)  # 30% density

        # Create base Bernoulli model
        y = rand(0:1, n_obs)
        base_model = ExponentialFamily(Bernoulli)

        # Create linearly transformed model
        obs_model = LinearlyTransformedObservationModel(base_model, A)
        obs_lik = obs_model(y)

        x0 = randn(n_latent)
        v = randn(n_latent)

        # Specialized implementation
        dH_specialized = loghessian_directional_derivative(x0, v, obs_lik)

        # Finite difference validation
        dH_finite = finite_diff_loghessian_derivative(x0, v, obs_lik)

        # ForwardDiff fallback for comparison
        loghessian_path = t -> loghessian(x0 + t * v, obs_lik)
        dH_fallback = ForwardDiff.derivative(loghessian_path, 0.0)

        @test size(dH_specialized) == (n_latent, n_latent)
        @test dH_specialized ≈ dH_fallback rtol = 1.0e-10
        @test dH_specialized ≈ dH_finite atol = 1.0e-4

        # Test with identity transformation (should match base model)
        A_identity = sparse(I(n_obs))
        obs_model_identity = LinearlyTransformedObservationModel(base_model, A_identity)
        obs_lik_identity = obs_model_identity(y)
        base_lik_direct = base_model(y)

        x0_small = randn(n_obs)
        v_small = randn(n_obs)

        dH_transformed = loghessian_directional_derivative(x0_small, v_small, obs_lik_identity)
        dH_base = loghessian_directional_derivative(x0_small, v_small, base_lik_direct)

        @test dH_transformed ≈ dH_base rtol = 1.0e-12

        # Test with Poisson base model
        y_poisson = rand(1:10, n_obs)
        base_model_poisson = ExponentialFamily(Poisson)
        obs_model_poisson = LinearlyTransformedObservationModel(base_model_poisson, A)
        obs_lik_poisson = obs_model_poisson(y_poisson)

        dH_poisson = loghessian_directional_derivative(x0, v, obs_lik_poisson)
        dH_poisson_fallback = ForwardDiff.derivative(
            t -> loghessian(x0 + t * v, obs_lik_poisson), 0.0
        )

        @test dH_poisson ≈ dH_poisson_fallback rtol = 1.0e-10
    end

    @testset "Type Stability" begin
        n = 5
        y_poisson = [1, 2, 3, 4, 5]
        obs_lik_poisson = ExponentialFamily(Poisson)(y_poisson)
        x0 = ones(n)
        v = ones(n)

        # Type stability for Poisson
        @inferred Diagonal loghessian_directional_derivative(x0, v, obs_lik_poisson)

        # Type stability for Bernoulli
        y_bernoulli = [1, 0, 1, 0, 1]
        obs_lik_bernoulli = ExponentialFamily(Bernoulli)(y_bernoulli)
        @inferred Diagonal loghessian_directional_derivative(x0, v, obs_lik_bernoulli)

        # Type stability for Normal
        y_normal = randn(n)
        obs_lik_normal = ExponentialFamily(Normal)(y_normal; σ = 1.0)
        @inferred Diagonal loghessian_directional_derivative(x0, v, obs_lik_normal)
    end

    @testset "Edge Cases" begin
        # Test with single observation
        y_single = [5]
        obs_lik = ExponentialFamily(Poisson)(y_single)
        x0_single = [1.0]
        v_single = [0.5]

        dH = loghessian_directional_derivative(x0_single, v_single, obs_lik)
        dH_fallback = forwarddiff_fallback(x0_single, v_single, obs_lik)

        @test size(dH) == (1, 1)
        @test dH ≈ dH_fallback rtol = 1.0e-10

        # Test with unit direction vector
        n = 5
        y = rand(1:10, n)
        obs_lik = ExponentialFamily(Poisson)(y)
        x0 = randn(n)
        e_i = zeros(n)
        e_i[3] = 1.0  # Unit vector in 3rd direction

        dH = loghessian_directional_derivative(x0, e_i, obs_lik)
        dH_fallback = forwarddiff_fallback(x0, e_i, obs_lik)

        @test dH ≈ dH_fallback rtol = 1.0e-10

        # Only the 3rd diagonal entry should be non-zero
        @test abs(dH[3, 3]) > 0
        @test all(abs.(diag(dH)[[1, 2, 4, 5]]) .< 1.0e-14)
    end

    @testset "Diagonal Structure Preservation" begin
        # Verify that exponential families return Diagonal matrices
        n = 8
        test_cases = [
            ("Poisson", ExponentialFamily(Poisson)(rand(1:10, n))),
            ("Bernoulli", ExponentialFamily(Bernoulli)(rand(0:1, n))),
            ("Normal", ExponentialFamily(Normal)(randn(n); σ = 1.0)),
        ]

        for (name, obs_lik) in test_cases
            x0 = randn(n)
            v = randn(n)
            dH = loghessian_directional_derivative(x0, v, obs_lik)

            @test dH isa Diagonal

            # Validate against fallback
            dH_fallback = forwarddiff_fallback(x0, v, obs_lik)
            @test dH ≈ dH_fallback rtol = 1.0e-10
        end
    end
end

@testset "Diagonal third/fourth derivatives — Gamma" begin
    Random.seed!(7)

    # Independent reference: central finite differences of the analytic loghessian
    # diagonal (GMRFs ships the closed-form Gamma loghessian, so this is AD-free).
    hess_diag(x, ol) = diag(loghessian(x, ol))
    function fd_third(x0, i, ol; δ = 1.0e-4)
        e = zeros(length(x0))
        e[i] = 1.0
        return (hess_diag(x0 + δ * e, ol)[i] - hess_diag(x0 - δ * e, ol)[i]) / (2δ)
    end
    function fd_fourth(x0, i, ol; δ = 1.0e-3)
        e = zeros(length(x0))
        e[i] = 1.0
        return (hess_diag(x0 + δ * e, ol)[i] - 2 * hess_diag(x0, ol)[i] + hess_diag(x0 - δ * e, ol)[i]) / δ^2
    end

    @testset "matches closed form + finite-diff (indices = 1:n)" begin
        n = 8
        y = rand(n) .+ 0.5            # Gamma observations must be positive
        φ = 2.7
        ol = ExponentialFamily(Gamma)(y; phi = φ)
        x0 = 0.5 .* randn(n)

        d3 = third_derivative_diagonal(ol, x0)
        d4 = fourth_derivative_diagonal(ol, x0)
        @test d3 !== nothing                 # no longer the AD fallback
        @test d4 !== nothing
        @test d3.indices == collect(1:n)
        @test d4.indices == collect(1:n)

        # closed form (log link, μ = e^η): h'''(η) = +φy e^{−η}, h''''(η) = −φy e^{−η}
        @test d3.values ≈ [φ * y[j] * exp(-x0[j]) for j in 1:n] rtol = 1.0e-12
        @test d4.values ≈ [-φ * y[j] * exp(-x0[j]) for j in 1:n] rtol = 1.0e-12

        # independent finite-difference check against GMRFs' analytic Hessian
        @test d3.values ≈ [fd_third(x0, i, ol) for i in 1:n] atol = 1.0e-4
        @test d4.values ≈ [fd_fourth(x0, i, ol) for i in 1:n] atol = 1.0e-2
    end

    @testset "honours custom indices (augmented η block)" begin
        # observations map to a sub-block of a larger latent vector, as in the
        # augmented layout where the η predictor block occupies positions 1:n_obs.
        n_latent = 12
        n_obs = 5
        idx = collect(1:n_obs)
        y = rand(n_obs) .+ 0.5
        φ = 1.3
        ol = GammaLikelihood(LogLink(), y, φ, idx)
        x0 = 0.3 .* randn(n_latent)

        d3 = third_derivative_diagonal(ol, x0)
        @test d3.indices == idx
        @test d3.values ≈ [φ * y[j] * exp(-x0[idx[j]]) for j in 1:n_obs] rtol = 1.0e-12
    end
end
