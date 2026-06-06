using Test
using Latte
using Latte: resolve_targets, Hyperparameters, DataDependentQuantity,
    _pit_quantile, _q_to_rank, hyperparameter_marginals
using Distributions
using Distributions: cdf
using GaussianMarkovRandomFields: IIDModel, ExponentialFamily
using Random

# Hand-built Gaussian IID LGM (τ free, σ fixed) — INLA gives a continuous τ
# marginal, the prerequisite for PIT ranking.
function _gauss_iid_lgm(n; σ = 0.5)
    hp = @hyperparams begin
        (τ ~ PCPrior.Precision(1.0, α = 0.01), transform = log, space = natural)
        σ = σ
    end
    return LatentGaussianModel(hp, IIDModel(n), ExponentialFamily(Normal))
end

@testset "PIT ranking" begin

    @testset "_q_to_rank maps the unit interval onto {0,…,L}" begin
        @test _q_to_rank(0.0, 99) == 0
        @test _q_to_rank(1.0, 99) == 99
        @test _q_to_rank(0.5, 99) == 50
        @test 0 <= _q_to_rank(0.27, 999) <= 999
        @test _q_to_rank(2.0, 99) == 99      # clamped
        @test _q_to_rank(-1.0, 99) == 0      # clamped
    end

    @testset "_pit_quantile = cdf(marginal, truth) for a scalar hp, nothing otherwise" begin
        n = 5
        lgm = _gauss_iid_lgm(n)
        Random.seed!(3)
        y = randn(n)
        res = inla(lgm, y; progress = false)

        d_hp = resolve_targets(Hyperparameters(), lgm)[1]   # τ
        ctx = (; result = res)
        truth_val = 1.7
        want = clamp(cdf(only(hyperparameter_marginals(res, :τ)), truth_val), 0.0, 1.0)
        @test _pit_quantile(d_hp, ctx, truth_val) ≈ want
        @test 0.0 <= _pit_quantile(d_hp, ctx, truth_val) <= 1.0

        # Derived quantities have no marginal CDF ⇒ no PIT.
        d_dd = resolve_targets(DataDependentQuantity(), lgm)[1]
        @test _pit_quantile(d_dd, ctx, 0.0) === nothing
    end

    @testset "sbc_run :auto differs from :sample for grid-based INLA" begin
        n = 5
        build = y -> _gauss_iid_lgm(n)
        common = (;
            n_attempted = 25, n_posterior = 200, engine = :inla,
            base_seed = UInt64(0x9173), progress = false,
        )
        r_auto = sbc_run(build, Vector{Missing}(missing, n); rank_method = :auto, common...)
        r_samp = sbc_run(build, Vector{Missing}(missing, n); rank_method = :sample, common...)

        @test all(0 .<= r_auto.ranks .<= r_auto.n_posterior)
        @test all(0 .<= r_samp.ranks .<= r_samp.n_posterior)
        # Same seed ⇒ same replicates; only the ranking rule differs. INLA's θ
        # samples are grid-quantized, so PIT and sample ranks diverge.
        @test r_auto.ranks != r_samp.ranks
    end

    @testset "invalid rank_method errors" begin
        n = 4
        build = y -> _gauss_iid_lgm(n)
        @test_throws ArgumentError sbc_run(
            build, Vector{Missing}(missing, n);
            n_attempted = 2, n_posterior = 16, engine = :inla,
            rank_method = :bogus, progress = false,
        )
    end
end
