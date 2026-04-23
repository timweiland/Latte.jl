using Test
using Latte
using Latte: PosteriorStrategy, PosteriorAccumulator, materialize
using GaussianMarkovRandomFields
using Distributions
using SparseArrays
using LinearAlgebra
using Random

@testset "Accumulator strategies" begin

    @testset "Strategies are immutable configs" begin
        for S in (DICStrategy, WAICStrategy, CPOStrategy, MarginalLogLikelihoodStrategy)
            @test !ismutabletype(S)
            @test S <: PosteriorStrategy
        end
    end

    @testset "materialize produces fresh accumulator instances" begin
        @test materialize(DICStrategy()) isa DICAccumulator
        @test materialize(WAICStrategy()) isa WAICAccumulator
        @test materialize(CPOStrategy()) isa CPOAccumulator
        @test materialize(MarginalLogLikelihoodStrategy()) isa MarginalLogLikelihoodAccumulator

        # Every call is independent state
        a = materialize(DICStrategy())
        b = materialize(DICStrategy())
        @test a !== b

        a2 = materialize(WAICStrategy())
        b2 = materialize(WAICStrategy())
        @test a2 !== b2
    end

    @testset "Strategy knobs flow through to accumulator" begin
        @test materialize(WAICStrategy(n_nodes = 25)).n_nodes == 25
        cpo_acc = materialize(CPOStrategy(n_nodes = 20, compute_pit = false))
        @test cpo_acc.n_nodes == 20
        @test cpo_acc.compute_pit == false
    end

    @testset "inla accepts strategies and stores materialized accumulators" begin
        Random.seed!(11)
        n = 10
        spec = @hyperparams begin
            (τ ~ Gamma(2, 1), transform = log, space = natural)
        end
        latent_fn = (; τ, kwargs...) -> (zeros(n), spdiagm(0 => fill(τ, n)))
        obs_model = ExponentialFamily(Poisson)
        model = LatentGaussianModel(spec, FunctionLatentModel(latent_fn, n), obs_model)
        y = rand(Poisson(3.0), n)

        result = inla(
            model, y;
            progress = false,
            accumulators = (DICStrategy(), WAICStrategy(), MarginalLogLikelihoodStrategy()),
        )

        @test result.accumulators[1] isa DICAccumulator
        @test result.accumulators[2] isa WAICAccumulator
        @test result.accumulators[3] isa MarginalLogLikelihoodAccumulator
        @test isfinite(result.accumulators[1].DIC)
        @test isfinite(result.accumulators[2].WAIC)
        @test isfinite(result.accumulators[3].log_marginal_likelihood)
    end

    @testset "Reusing a strategy tuple across inla calls is safe (no shared state)" begin
        # This is the footgun that prompted the strategy layer:
        # before the refactor, a const accumulator tuple was shared between calls
        # and accumulated across runs, silently corrupting results.
        Random.seed!(42)
        n = 8
        spec = @hyperparams begin
            (τ ~ Gamma(2, 1), transform = log, space = natural)
        end
        latent_fn = (; τ, kwargs...) -> (zeros(n), spdiagm(0 => fill(τ, n)))
        obs_model = ExponentialFamily(Poisson)
        model = LatentGaussianModel(spec, FunctionLatentModel(latent_fn, n), obs_model)
        y = rand(Poisson(3.0), n)

        strategies = (DICStrategy(), WAICStrategy(), MarginalLogLikelihoodStrategy())
        r1 = inla(model, y; progress = false, accumulators = strategies)
        r2 = inla(model, y; progress = false, accumulators = strategies)

        # Independent materialized instances
        @test r1.accumulators[1] !== r2.accumulators[1]
        @test r1.accumulators[2] !== r2.accumulators[2]

        # Same data + model ⇒ matching results
        @test r1.accumulators[1].DIC ≈ r2.accumulators[1].DIC
        @test r1.accumulators[2].WAIC ≈ r2.accumulators[2].WAIC
    end

end
