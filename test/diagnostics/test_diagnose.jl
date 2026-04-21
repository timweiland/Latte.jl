using Test
using Latte
using Distributions
using GaussianMarkovRandomFields
using SparseArrays
using Random

# diagnose() works uniformly across the three Laplace-based InferenceResult
# types. Well-behaved IID Poisson model → excellent/acceptable verdict;
# exact k̂ varies with RNG but interpretation is stable.
@testset "diagnose() on InferenceResult types" begin
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

    @testset "Uniform shape across INLA, TMB, HMC-Laplace" begin
        n = 15
        model = make_poisson_iid_model(n)
        Random.seed!(42)
        y = rand(Poisson(3.0), n)

        inla_r = inla(model, y; progress = false)
        tmb_r = tmb(model, y)
        hmc_r = hmc_laplace(model, y; n_samples = 100, n_warmup = 50, rng = MersenneTwister(1))

        for r in (inla_r, tmb_r, hmc_r)
            Random.seed!(2026)
            d = diagnose(r; M = 500)
            # Shape: NamedTuple with expected fields
            @test haskey(d, :rel_ess)
            @test haskey(d, :ess)
            @test haskey(d, :pareto_k)
            @test haskey(d, :interpretation)
            @test haskey(d, :M)

            # Sanity of values
            @test 0 < d.rel_ess <= 1
            @test d.ess > 0
            @test d.interpretation in (:excellent, :acceptable, :unreliable)
            @test d.M == 500
        end
    end

    @testset "Well-behaved Poisson yields acceptable / excellent verdict" begin
        n = 20
        model = make_poisson_iid_model(n)
        Random.seed!(42)
        y = rand(Poisson(3.0), n)
        tmb_r = tmb(model, y)

        Random.seed!(2026)
        d = diagnose(tmb_r; M = 1000)
        @test d.interpretation in (:excellent, :acceptable)
        # Laplace is near-exact for a Poisson with λ ≈ 3 (well away from the
        # "many zeros" regime that would bend it); k̂ should stay < 0.7
        @test d.pareto_k < 0.7
    end

    @testset "diagnose_chain(HMCLaplaceResult) scans quantiles" begin
        n = 12
        model = make_poisson_iid_model(n)
        Random.seed!(42)
        y = rand(Poisson(3.0), n)
        hmc_r = hmc_laplace(model, y; n_samples = 150, n_warmup = 50, rng = MersenneTwister(5))

        Random.seed!(2026)
        d = diagnose_chain(hmc_r; M = 300, quantiles = (0.025, 0.5, 0.975))
        # Keys: at_map + one per quantile
        @test haskey(d, :at_map)
        @test haskey(d, :q_0_025)
        @test haskey(d, :q_0_5)
        @test haskey(d, :q_0_975)
        # Each entry is a per-point diagnostic NamedTuple
        for v in values(d)
            @test haskey(v, :rel_ess)
            @test haskey(v, :pareto_k)
            @test haskey(v, :interpretation)
        end
    end
end
