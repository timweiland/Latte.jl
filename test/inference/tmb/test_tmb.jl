using Test
using Latte
using Distributions
using GaussianMarkovRandomFields
using LinearAlgebra
using SparseArrays
using Random
using Statistics

isdefined(@__MODULE__, :make_poisson_iid_model) ||
    include(joinpath(@__DIR__, "..", "..", "shared_test_models.jl"))

@testset "tmb(LatentGaussianModel, y)" begin
    # One shared fit (n = 10, seed 42) serves the alias / rand / reproducibility
    # testsets, which only inspect the fitted result.
    shared_n = 10
    shared_result = let
        Random.seed!(42)
        tmb(make_poisson_iid_model(shared_n), rand(Poisson(3.0), shared_n))
    end

    @testset "Result shape and protocol conformance" begin
        n = 15
        model = make_poisson_iid_model(n)
        Random.seed!(42)
        y = rand(Poisson(3.0), n)
        result = tmb(model, y)

        @test result isa Latte.InferenceResult

        # Tier 1 protocol
        lm = latent_marginals(result)
        @test lm isa Vector{<:Distribution}
        @test length(lm) == n

        hp = hyperparameter_marginals(result)
        @test hp isa Vector{<:Distribution}
        @test length(hp) == 1

        @test latent_groups(result) isa AbstractDict
        @test hyperparameter_groups(result)[:τ] == 1:1
        @test hyperparameter_marginals(result, :τ) == hp

        @test length(hyperparameter_mode(result)) == 1
        @test converged(result) isa Bool
        @test time_elapsed(result) > 0
        @test log_marginal_likelihood(result) isa Float64
    end

    @testset "Hyperparameter marginals live in natural space" begin
        # τ uses `transform = log`, so the natural-space support is
        # τ > 0. Protocol guarantees natural-space marginals.
        n = 30
        model = make_poisson_iid_model(n)
        Random.seed!(42)
        y = rand(Poisson(3.0), n)
        result = tmb(model, y)

        m = hyperparameter_marginals(result, :τ)[1]
        @test mean(m) > 0
        @test quantile(m, 0.025) > 0
        @test quantile(m, 0.975) > quantile(m, 0.025)
    end

    @testset "TMB aliases" begin
        n, result = shared_n, shared_result

        @test fixed_effects(result) === hyperparameter_marginals(result)
        @test random_effects(result) === latent_marginals(result)
        @test fixef(result) === hyperparameter_marginals(result)
        @test ranef(result) === latent_marginals(result)
    end

    @testset "rand(TMBResult, n) returns PosteriorSamples" begin
        n, result = shared_n, shared_result

        samples = rand(MersenneTwister(1), result, 20)
        @test samples isa PosteriorSamples
        @test size(samples.θ) == (20, 1)
        @test size(samples.x) == (20, n)
        @test length(samples) == 20

        # Iteration yields per-draw NamedTuples
        s = first(samples)
        @test haskey(s, :θ) && haskey(s, :x)
        @test length(s.θ) == 1 && length(s.x) == n

        # Single-draw form returns a NamedTuple directly
        s1 = rand(MersenneTwister(1), result)
        @test s1 isa NamedTuple
        @test haskey(s1, :θ) && haskey(s1, :x)

        # include_y adds posterior-predictive samples
        samples_y = rand(MersenneTwister(1), result, 5; include_y = true)
        @test size(samples_y.y) == (5, n)
        @test eltype(samples_y.y) <: Integer
    end

    @testset "Reproducibility with seeded RNG" begin
        n, result = shared_n, shared_result

        s1 = rand(MersenneTwister(99), result, 5)
        s2 = rand(MersenneTwister(99), result, 5)
        @test s1.θ == s2.θ
        @test s1.x == s2.x
    end

    @testset "Augmented LGM: θ_cov positive with default ADStrategy" begin
        # Regression test for the TMB negative-θ_cov bug on augmented
        # LGMs. Prior to Fix 1 (src/inference/tmb/inference.jl using the
        # diff_strategy dispatch), default TMB called
        # `FiniteDiff.finite_difference_hessian` directly — catastrophic
        # cancellation at small steps on the augmented objective produced a
        # negative `θ_cov`. The new default `ADStrategy()` uses FD-of-AD-
        # gradient, which is noise-robust.
        Random.seed!(2026)
        n, p, G = 40, 2, 5
        X = [ones(n) randn(n)]
        group = rand(1:G, n)
        β_true = [0.3, 0.5]
        u_true = randn(G) ./ sqrt(4.0)
        y_obs = [
            rand(Poisson(exp(X[i, :] ⋅ β_true + u_true[group[i]])))
                for i in 1:n
        ]

        spec = @hyperparams begin
            (τ_u ~ Gamma(2, 1), transform = log, space = natural)
        end
        latent_fn(; τ_u, kwargs...) = (zeros(p + G), spdiagm(0 => vcat(fill(1 / 100, p), fill(τ_u, G))))
        A = zeros(n, p + G)
        A[:, 1:p] .= X
        for i in 1:n
            A[i, p + group[i]] = 1.0
        end
        lgm = LatentGaussianModel(
            spec, FunctionLatentModel(latent_fn, p + G),
            LinearlyTransformedObservationModel(ExponentialFamily(Poisson, LogLink()), sparse(A)),
        )

        r = tmb(lgm, y_obs)
        @test r.θ_cov[1, 1] > 0
        @test isfinite(r.θ_cov[1, 1])
    end
end
