using Test
using Latte
using Distributions
using GaussianMarkovRandomFields
using LinearAlgebra
using SparseArrays
using Random
using Statistics

@testset "tmb(LatentGaussianModel, y)" begin
    function make_poisson_iid_model(n)
        spec = @hyperparams begin
            (τ ~ Gamma(2, 1), transform = log, space = natural)
        end
        function latent_func(; τ, kwargs...)
            Q = spdiagm(0 => fill(τ, n))
            return (zeros(n), Q)
        end
        obs_model = ExponentialFamily(Poisson)
        return LatentGaussianModel(spec, FunctionLatentModel(latent_func, n), obs_model)
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

    @testset "TMB aliases" begin
        n = 10
        model = make_poisson_iid_model(n)
        Random.seed!(42)
        y = rand(Poisson(3.0), n)
        result = tmb(model, y)

        @test fixed_effects(result) === hyperparameter_marginals(result)
        @test random_effects(result) === latent_marginals(result)
        @test fixef(result) === hyperparameter_marginals(result)
        @test ranef(result) === latent_marginals(result)
    end

    @testset "rand(TMBResult, n) returns PosteriorSamples" begin
        n = 10
        model = make_poisson_iid_model(n)
        Random.seed!(42)
        y = rand(Poisson(3.0), n)
        result = tmb(model, y)

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
        n = 10
        model = make_poisson_iid_model(n)
        Random.seed!(42)
        y = rand(Poisson(3.0), n)
        result = tmb(model, y)

        s1 = rand(MersenneTwister(99), result, 5)
        s2 = rand(MersenneTwister(99), result, 5)
        @test s1.θ == s2.θ
        @test s1.x == s2.x
    end
end
