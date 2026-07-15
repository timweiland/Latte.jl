using Test
using Latte
using Latte: SBCResult, sbc_coverage, sbc_quantile_position
using DynamicPPL: @model
using DynamicPPL  # @latte needs DynamicPPL in scope
using Distributions
using GaussianMarkovRandomFields: IIDModel
using Statistics
using Random

# Shared canonical models compile the inference pipeline once for the whole block.
isdefined(@__MODULE__, :sbc_pois) || include("shared_models.jl")

@testset "sbc_run (end-to-end)" begin

    @testset "smoke: tiny Poisson+IID on INLA" begin
        n = 5
        build = y -> sbc_pois(y, n)
        y_proto = Vector{Missing}(missing, n)

        r = sbc_run(
            build, y_proto;
            n_attempted = 8,
            n_posterior = 64,
            engine = :inla,
            random = (:x,),
            base_seed = UInt64(0x5b),
            progress = false,
        )

        @test r isa SBCResult
        @test r.n_attempted == 8
        @test r.n_success + r.n_failures == 8
        @test length(r.targets) == 1
        @test r.targets[1].label == :τ
        @test size(r.ranks) == (r.n_success, 1)
        @test size(r.truths) == (r.n_success, 1)
        @test all(0 .<= r.ranks .<= 64)
        @test all(isfinite, r.truths)
        @test r.status in (:valid, :completed_with_failures, :invalid)
    end

    @testset "determinism across executors" begin
        n = 4
        build = y -> sbc_pois(y, n)
        y_proto = Vector{Missing}(missing, n)

        common = (;
            n_attempted = 4, n_posterior = 32,
            engine = :inla, random = (:x,),
            base_seed = UInt64(0xbeef), progress = false,
        )
        r_seq = sbc_run(build, y_proto; common..., executor = SequentialExecutor())
        r_thr = sbc_run(build, y_proto; common..., executor = ThreadedExecutor(nworkers = 2))

        @test r_seq.ranks == r_thr.ranks
        @test r_seq.truths == r_thr.truths
        @test r_seq.n_success == r_thr.n_success
    end

    @testset "determinism: same seed ⇒ same ranks" begin
        n = 4
        build = y -> sbc_pois(y, n)
        y_proto = Vector{Missing}(missing, n)

        r1 = sbc_run(
            build, y_proto; n_attempted = 3, n_posterior = 32,
            engine = :inla, random = (:x,), base_seed = UInt64(0xdead), progress = false
        )
        r2 = sbc_run(
            build, y_proto; n_attempted = 3, n_posterior = 32,
            engine = :inla, random = (:x,), base_seed = UInt64(0xdead), progress = false
        )

        @test r1.ranks == r2.ranks
        @test r1.truths == r2.truths
    end

    @testset "works for :tmb engine" begin
        n = 4
        build = y -> sbc_pois(y, n)
        y_proto = Vector{Missing}(missing, n)
        r = sbc_run(
            build, y_proto;
            n_attempted = 6, n_posterior = 64,
            engine = :tmb, random = (:x,),
            base_seed = UInt64(0x71), progress = false,
        )
        @test r.n_attempted == 6
        @test r.engine == :tmb
        @test r.n_success + r.n_failures == 6
        @test r.n_success >= 1  # guards against "every replicate silently fails"
    end

    @testset "works for :hmc_laplace engine and is reproducible" begin
        n = 4
        build = y -> sbc_pois(y, n)
        y_proto = Vector{Missing}(missing, n)
        common = (;
            n_attempted = 3, n_posterior = 48,
            engine = :hmc_laplace, random = (:x,),
            base_seed = UInt64(0x04c3), progress = false,
            engine_kwargs = (n_samples = 100, n_warmup = 50),
        )
        r1 = sbc_run(build, y_proto; common...)
        r2 = sbc_run(build, y_proto; common...)
        @test r1.n_success >= 1
        # Per-replicate RNG means repeated runs with same seed ⇒ same ranks.
        @test r1.ranks == r2.ranks
    end

    @testset "truths in natural space" begin
        # For τ ~ PCPrior.Precision, truths should be positive (natural
        # space). If working-space leaked in, some would be negative
        # (log(τ) for small τ).
        n = 3
        build = y -> sbc_pois(y, n)
        y_proto = Vector{Missing}(missing, n)
        r = sbc_run(
            build, y_proto;
            n_attempted = 8, n_posterior = 32,
            engine = :inla, random = (:x,),
            base_seed = UInt64(0x0abc), progress = false,
        )
        @test all(r.truths .> 0)   # natural-space τ is strictly positive
    end

    @testset "summaries" begin
        n = 4
        build = y -> sbc_pois(y, n)
        y_proto = Vector{Missing}(missing, n)
        r = sbc_run(
            build, y_proto; n_attempted = 8, n_posterior = 32,
            engine = :inla, random = (:x,), base_seed = UInt64(0xface), progress = false
        )

        q = sbc_quantile_position(r, 1)
        @test length(q) == r.n_success
        @test all(0.0 .< q .< 1.0)

        cov = sbc_coverage(r, 1)
        @test 0.0 <= cov.cov_0_5 <= 1.0
        @test 0.0 <= cov.cov_0_8 <= 1.0
        @test 0.0 <= cov.cov_0_95 <= 1.0
        # Nested: wider CI always covers at least as often as narrower
        @test cov.cov_0_5 <= cov.cov_0_8 <= cov.cov_0_95
    end

    @testset "accepts a @latte LGM factory directly" begin
        @latte function smoke_latte(y, n)
            τ ~ PCPrior.Precision(1.0, α = 0.01)
            x ~ IIDModel(n)(τ = τ)
            for i in eachindex(y)
                y[i] ~ Poisson(exp(x[i]); check_args = false)
            end
        end
        n = 5

        r = sbc_run(
            y -> smoke_latte(y, n), Vector{Missing}(missing, n);
            n_attempted = 20, n_posterior = 200,
            engine = :inla, base_seed = UInt64(0x05bc_abc),
            progress = false,
        )

        @test r isa SBCResult
        @test r.n_attempted == 20
        @test r.n_success + r.n_failures == 20
        @test r.status != :invalid
        @test length(r.targets) == 1
        @test r.targets[1].label == :τ

        # τ mean quantile position should land near 0.5 for a calibrated run.
        q = sbc_quantile_position(r, 1)
        @test isapprox(Statistics.mean(q), 0.5; atol = 0.2)

        cov = sbc_coverage(r, 1)
        @test isfinite(cov.cov_0_5)
        @test isfinite(cov.cov_0_8)
        @test isfinite(cov.cov_0_95)
    end
end
