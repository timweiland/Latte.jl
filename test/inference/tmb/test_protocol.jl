using Test
using Latte
using Distributions
using GaussianMarkovRandomFields
using SparseArrays
using Random
using Statistics

# Exercises the shared InferenceResult protocol across two concrete
# implementations (INLAResult + TMBResult) on the same LGM. If the abstract is
# well-shaped, the same client code should work for both.
@testset "InferenceResult protocol — INLA and TMB on the same model" begin
    n = 15
    spec = @hyperparams begin
        (τ ~ Gamma(2, 1), transform = log, space = natural)
    end
    function latent_func(; τ, kwargs...)
        Q = spdiagm(0 => fill(τ, n))
        return (zeros(n), Q)
    end
    obs_model = ExponentialFamily(Poisson)
    model = LatentGaussianModel(spec, FunctionLatentModel(latent_func, n), obs_model)

    Random.seed!(42)
    y = rand(Poisson(3.0), n)

    inla_result = inla(model, y; progress = false)
    tmb_result = tmb(model, y)

    @testset "Both results satisfy Tier 1 protocol" begin
        for r in (inla_result, tmb_result)
            @test r isa Latte.InferenceResult

            @test latent_marginals(r) isa Vector{<:Distribution}
            @test length(latent_marginals(r)) == n

            @test hyperparameter_marginals(r) isa Vector{<:Distribution}
            @test length(hyperparameter_marginals(r)) == 1

            @test haskey(hyperparameter_groups(r), :τ)
            @test hyperparameter_groups(r)[:τ] == 1:1

            @test latent_groups(r) isa AbstractDict  # empty for manually-built models

            # Name-keyed slice
            @test hyperparameter_marginals(r, :τ) == hyperparameter_marginals(r)

            @test hyperparameter_mode(r) isa Latte.NaturalHyperparameters
            @test converged(r) isa Bool
            @test time_elapsed(r) > 0
        end
    end

    @testset "Methods produce similar estimates (same model, same data)" begin
        # Modes agree: INLA finds the same mode TMB reports
        @test hyperparameter_mode(inla_result).θ ≈ hyperparameter_mode(tmb_result).θ rtol = 1.0e-4

        # Latent means agree to ~1 SE (INLA integrates over θ; TMB Laplace-at-MAP;
        # for a 1D hyperparameter with concentrated posterior, these agree closely)
        inla_means = mean.(latent_marginals(inla_result))
        tmb_means = mean.(latent_marginals(tmb_result))
        tmb_stds = std.(latent_marginals(tmb_result))
        for i in 1:n
            @test abs(inla_means[i] - tmb_means[i]) < tmb_stds[i]
        end

        # Log marginal likelihood estimates: both produce a Float64
        @test log_marginal_likelihood(inla_result) isa Float64
        @test log_marginal_likelihood(tmb_result) isa Float64
    end

    @testset "rand returns PosteriorSamples for both" begin
        for r in (inla_result, tmb_result)
            s = rand(MersenneTwister(1), r, 10)
            @test s isa PosteriorSamples
            @test size(s.x, 1) == 10
            @test size(s.x, 2) == n
            @test length(s) == 10
        end
    end
end
