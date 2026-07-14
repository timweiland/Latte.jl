using Test
using Latte
using Distributions
using GaussianMarkovRandomFields
using SparseArrays
using Random
using Statistics

# Same model, three methods, shared protocol.
#
# This is the architectural test: if the InferenceResult abstract is well
# shaped, the same client code works uniformly over parametric (INLA, TMB)
# and sample-based (HMC-Laplace) result types.
@testset "Three-way protocol conformance — INLA, TMB, HMC-Laplace" begin
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

    inla_r = inla(model, y; progress = false)
    tmb_r = tmb(model, y)
    hmc_r = hmc_laplace(model, y; n_samples = 100, n_warmup = 50, rng = MersenneTwister(1))

    results = [("INLA", inla_r), ("TMB", tmb_r), ("HMC-Laplace", hmc_r)]

    @testset "All three satisfy Tier 1 protocol" begin
        for (name, r) in results
            @test r isa Latte.InferenceResult

            @test latent_marginals(r) isa Vector{<:Distribution}
            @test length(latent_marginals(r)) == n

            @test hyperparameter_marginals(r) isa Vector{<:Distribution}
            @test length(hyperparameter_marginals(r)) == 1

            @test hyperparameter_groups(r)[:τ] == 1:1
            @test hyperparameter_marginals(r, :τ) == hyperparameter_marginals(r)

            @test hyperparameter_mode(r) isa Latte.NaturalHyperparameters
            @test time_elapsed(r) > 0
        end
    end

    @testset "log_marginal_likelihood: parametric methods produce one; HMC doesn't" begin
        @test log_marginal_likelihood(inla_r) isa Float64
        @test log_marginal_likelihood(tmb_r) isa Float64
        @test log_marginal_likelihood(hmc_r) === nothing
    end

    @testset "All three agree on posterior means (same model, same data)" begin
        # On a well-behaved model the three methods should agree within
        # MCMC error; we allow 2 HMC-std slack for each latent site.
        inla_means = mean.(latent_marginals(inla_r))
        tmb_means = mean.(latent_marginals(tmb_r))
        hmc_means = mean.(latent_marginals(hmc_r))
        hmc_stds = std.(latent_marginals(hmc_r))

        for i in 1:n
            @test abs(inla_means[i] - hmc_means[i]) < 2 * hmc_stds[i]
            @test abs(tmb_means[i] - hmc_means[i]) < 2 * hmc_stds[i]
        end
    end

    @testset "rand returns PosteriorSamples across all three" begin
        for (name, r) in results
            ps = rand(MersenneTwister(1), r, 20)
            @test ps isa PosteriorSamples
            @test size(ps.x) == (20, n)
            @test length(ps) == 20
        end
    end
end
