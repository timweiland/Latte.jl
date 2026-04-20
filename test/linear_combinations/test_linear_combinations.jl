using Test
using Latte
using GaussianMarkovRandomFields
using Distributions
using SparseArrays
using Statistics
using Random
using LinearAlgebra

@testset "linear_combinations" begin

    # Shared model constructors
    function make_normal_iid_model(n)
        spec = @hyperparams begin
            (σ ~ InverseGamma(2, 1), transform = log, space = natural)
        end
        function latent_func(; σ, kwargs...)
            Q = spdiagm(0 => fill(1 / σ^2, n))
            return (zeros(n), Q)
        end
        obs_model = ExponentialFamily(Normal)
        return LatentGaussianModel(spec, FunctionLatentModel(latent_func, n), obs_model)
    end

    function make_normal_ar1_model(n)
        spec = @hyperparams begin
            (σ ~ InverseGamma(2, 1), transform = log, space = natural)
            (τ ~ Gamma(2, 1), transform = log, space = natural)
        end
        function latent_func(; τ, kwargs...)
            # Tridiagonal precision (RW1 + ridge) — introduces correlations
            d = fill(2τ, n)
            d[1] = τ; d[n] = τ
            d .+= 0.01  # small ridge for positive definiteness
            off = fill(-τ, n - 1)
            Q = spdiagm(0 => d, 1 => off, -1 => off)
            return (zeros(n), Q)
        end
        obs_model = ExponentialFamily(Normal)
        return LatentGaussianModel(spec, FunctionLatentModel(latent_func, n), obs_model)
    end

    # Fit a model once for reuse
    n = 10
    model = make_normal_iid_model(n)
    Random.seed!(42)
    y = randn(n)
    result = inla(model, y; progress = false)

    @testset "Identity matrix: means match latent marginals" begin
        I_mat = Matrix(1.0I, n, n)
        lc_marginals = linear_combinations(result, I_mat)

        @test length(lc_marginals) == n
        for i in 1:n
            @test mean(lc_marginals[i]) ≈ mean(result.latent_marginals[i]) atol = 1.0e-10
        end
    end

    @testset "Sum: captures correlations (AR1 model)" begin
        # Use AR1 model which has off-diagonal entries in Q (correlated latents)
        model_ar1 = make_normal_ar1_model(n)
        Random.seed!(42)
        y_ar1 = randn(n)
        result_ar1 = inla(model_ar1, y_ar1; progress = false)

        a = ones(n)
        sum_marginal = linear_combinations(result_ar1, a)

        # Mean should equal sum of latent means
        expected_mean = sum(mean(result_ar1.latent_marginals[i]) for i in 1:n)
        @test mean(sum_marginal) ≈ expected_mean atol = 1.0e-10

        # Variance should NOT equal sum of marginal variances (correlations matter)
        sum_of_variances = sum(var(result_ar1.latent_marginals[i]) for i in 1:n)
        @test var(sum_marginal) != sum_of_variances
    end

    @testset "Contrast: difference of two variables" begin
        a = zeros(n)
        a[1] = 1.0
        a[2] = -1.0
        contrast_marginal = linear_combinations(result, a)

        # Mean should be difference of means
        expected_mean = mean(result.latent_marginals[1]) - mean(result.latent_marginals[2])
        @test mean(contrast_marginal) ≈ expected_mean atol = 1.0e-10

        # Should be a valid distribution
        @test var(contrast_marginal) > 0
        @test isfinite(std(contrast_marginal))
    end

    @testset "Matrix: multiple linear combinations" begin
        A = zeros(3, n)
        A[1, 1] = 1.0  # first variable
        A[2, :] .= 1.0 / n  # average
        A[3, 1] = 1.0; A[3, 2] = -1.0  # contrast

        marginals = linear_combinations(result, A)
        @test length(marginals) == 3
        @test all(m -> m isa WeightedMixture, marginals)

        # First row selects x[1]
        @test mean(marginals[1]) ≈ mean(result.latent_marginals[1]) atol = 1.0e-10

        # Second row is average
        expected_avg = sum(mean(result.latent_marginals[i]) for i in 1:n) / n
        @test mean(marginals[2]) ≈ expected_avg atol = 1.0e-10
    end

    @testset "Sparse A" begin
        A_sparse = sparse([1, 2], [1, 2], [1.0, 1.0], 2, n)
        marginals = linear_combinations(result, A_sparse)
        @test length(marginals) == 2
        @test mean(marginals[1]) ≈ mean(result.latent_marginals[1]) atol = 1.0e-10
        @test mean(marginals[2]) ≈ mean(result.latent_marginals[2]) atol = 1.0e-10
    end

    @testset "Single vector convenience" begin
        a = zeros(n)
        a[1] = 1.0
        marginal = linear_combinations(result, a)
        @test marginal isa WeightedMixture
        @test mean(marginal) ≈ mean(result.latent_marginals[1]) atol = 1.0e-10
    end

    @testset "Dimension mismatch error" begin
        @test_throws DimensionMismatch linear_combinations(result, ones(2, n + 1))
        @test_throws DimensionMismatch linear_combinations(result, ones(n + 1))
    end

    @testset "Edge case: n=1 model" begin
        model1 = make_normal_iid_model(1)
        Random.seed!(99)
        y1 = randn(1)
        result1 = inla(model1, y1; progress = false)

        marginal = linear_combinations(result1, [1.0])
        @test marginal isa WeightedMixture
        @test isfinite(mean(marginal))
        @test var(marginal) > 0
    end

    @testset "Integration: CCD exploration" begin
        model_ccd = make_normal_iid_model(n)
        Random.seed!(42)
        y_ccd = randn(n)
        result_ccd = inla(model_ccd, y_ccd; progress = false, exploration_strategy = CCDExplorationStrategy())

        a = ones(n)
        marginal = linear_combinations(result_ccd, a)

        expected_mean = sum(mean(result_ccd.latent_marginals[i]) for i in 1:n)
        @test mean(marginal) ≈ expected_mean atol = 1.0e-10
        @test var(marginal) > 0
    end
end
