using Test
using Latte
using Distributions
using GaussianMarkovRandomFields
using LinearAlgebra
using SparseArrays
using Random
using Statistics

@testset "hmc_laplace(LatentGaussianModel, y)" begin
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
        result = hmc_laplace(model, y; n_samples = 150, n_warmup = 50, rng = MersenneTwister(1))

        @test result isa Latte.InferenceResult
        @test result isa HMCLaplaceResult

        # Tier 1 protocol
        lm = latent_marginals(result)
        @test lm isa Vector{<:Distribution}
        @test length(lm) == n
        # Latent marginals are mixture-of-Gaussians across chain samples
        @test lm[1] isa MixtureModel

        hp = hyperparameter_marginals(result)
        @test hp isa Vector{<:Distribution}
        @test length(hp) == 1

        @test latent_groups(result) isa AbstractDict
        @test hyperparameter_groups(result)[:τ] == 1:1
        @test hyperparameter_marginals(result, :τ) == hp

        @test length(hyperparameter_mode(result)) == 1
        @test converged(result) isa Bool
        @test time_elapsed(result) > 0

        # log p(y): Tier 2 — HMC honestly returns nothing (no bridge sampling)
        @test log_marginal_likelihood(result) === nothing
    end

    @testset "MCMC-specific diagnostics" begin
        n = 10
        model = make_poisson_iid_model(n)
        Random.seed!(42)
        y = rand(Poisson(3.0), n)
        result = hmc_laplace(model, y; n_samples = 150, n_warmup = 50, rng = MersenneTwister(2))

        # Raw chain access
        chain = samples(result)
        @test chain isa Matrix{Float64}
        @test size(chain) == (150, 1)

        @test divergences(result) isa Int
        @test divergences(result) >= 0

        @test mean_tree_depth(result) > 0
        # With Laplace preconditioning the chain should mix in ≤ a few steps;
        # 6 is a generous upper bound well below the default NUTS cap of 10.
        @test mean_tree_depth(result) < 6

        @test 0 <= acceptance_rate(result) <= 1
        # NUTS targets 0.8 accept rate via step-size adaptation; be forgiving.
        @test acceptance_rate(result) > 0.5

        @test mean_step_size(result) > 0
    end

    @testset "rand(HMCLaplaceResult, n) returns PosteriorSamples" begin
        n = 10
        model = make_poisson_iid_model(n)
        Random.seed!(42)
        y = rand(Poisson(3.0), n)
        result = hmc_laplace(model, y; n_samples = 100, n_warmup = 50, rng = MersenneTwister(3))

        ps = rand(MersenneTwister(1), result, 30)
        @test ps isa PosteriorSamples
        @test size(ps.θ) == (30, 1)
        @test size(ps.x) == (30, n)
        @test length(ps) == 30

        s = first(ps)
        @test haskey(s, :θ) && haskey(s, :x)

        # Single-sample form
        s1 = rand(MersenneTwister(1), result)
        @test s1 isa NamedTuple
        @test haskey(s1, :θ) && haskey(s1, :x)

        # include_y adds posterior-predictive y
        ps_y = rand(MersenneTwister(1), result, 10; include_y = true)
        @test size(ps_y.y) == (10, n)
    end

    @testset "Reproducibility with seeded RNG" begin
        n = 10
        model = make_poisson_iid_model(n)
        Random.seed!(42)
        y = rand(Poisson(3.0), n)

        r1 = hmc_laplace(model, y; n_samples = 100, n_warmup = 50, rng = MersenneTwister(7))
        r2 = hmc_laplace(model, y; n_samples = 100, n_warmup = 50, rng = MersenneTwister(7))
        @test samples(r1) == samples(r2)
    end
end
