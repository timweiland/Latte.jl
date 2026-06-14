using Test
using Latte
using Latte: PriorModeStart, RandomStarts, resolve_mode_starts, ModeStartStrategy
using GaussianMarkovRandomFields
using Distributions
using LinearAlgebra
using SparseArrays
using Random

# Mode-finder initialisation API: `mode_init` kwarg + multi-start +
# `mode_info` return field.

# Small reusable test model: single hp τ controlling a white-noise latent
# precision, Bernoulli observations.
function _make_simple_lgm(n = 6)
    spec = @hyperparams begin
        (τ ~ Gamma(2, 1), transform = log, space = natural)
    end
    function simple_latent(; τ, kwargs...)
        Q = spdiagm(0 => fill(τ, n))
        return (zeros(n), Q)
    end
    obs_model = ExponentialFamily(Bernoulli)
    return LatentGaussianModel(spec, FunctionLatentModel(simple_latent, n), obs_model)
end

# Two-hp model so the multi-start cases have something to spread across.
function _make_two_hp_lgm(n = 6)
    spec = @hyperparams begin
        (τ ~ Gamma(2, 1), transform = log, space = natural)
        (σ ~ Gamma(2, 1), transform = log, space = natural)
    end
    function two_hp_latent(; τ, σ, kwargs...)
        Q = spdiagm(0 => fill(τ * σ, n))
        return (zeros(n), Q)
    end
    obs_model = ExponentialFamily(Bernoulli)
    return LatentGaussianModel(spec, FunctionLatentModel(two_hp_latent, n), obs_model)
end

@testset "Mode-finder init" begin
    Random.seed!(20260514)

    @testset "PriorModeStart is the default and reproduces legacy behavior" begin
        lgm = _make_simple_lgm()
        y = [true, false, true, true, false, false]

        θ_star_default, _, _ = Latte.find_hyperparameter_mode(lgm, y)
        θ_star_explicit, _, _ = Latte.find_hyperparameter_mode(
            lgm, y; mode_init = PriorModeStart(),
        )
        @test θ_star_default.θ ≈ θ_star_explicit.θ atol = 1.0e-10
    end

    @testset "resolve_mode_starts: NamedTuple → single working start" begin
        lgm = _make_simple_lgm()
        spec = lgm.hyperparameter_spec
        starts = resolve_mode_starts((; τ = 0.7), spec)
        @test length(starts) == 1
        s = starts[1]
        @test s isa WorkingHyperparameters
        # The working-space value is log(0.7) given the log transform.
        @test s.θ[1] ≈ log(0.7) atol = 1.0e-10
    end

    @testset "resolve_mode_starts: Vector{NamedTuple} → multi-start" begin
        lgm = _make_simple_lgm()
        spec = lgm.hyperparameter_spec
        starts = resolve_mode_starts([(; τ = 0.5), (; τ = 1.5), (; τ = 3.0)], spec)
        @test length(starts) == 3
        @test starts[1].θ[1] ≈ log(0.5) atol = 1.0e-10
        @test starts[2].θ[1] ≈ log(1.5) atol = 1.0e-10
        @test starts[3].θ[1] ≈ log(3.0) atol = 1.0e-10
    end

    @testset "resolve_mode_starts: missing hp → ArgumentError" begin
        lgm = _make_two_hp_lgm()
        spec = lgm.hyperparameter_spec
        @test_throws ArgumentError resolve_mode_starts((; τ = 1.0), spec)
    end

    @testset "resolve_mode_starts: unknown hp → ArgumentError" begin
        lgm = _make_simple_lgm()
        spec = lgm.hyperparameter_spec
        @test_throws ArgumentError resolve_mode_starts((; τ = 1.0, bogus = 0.1), spec)
    end

    @testset "resolve_mode_starts: non-finite working coord → ArgumentError" begin
        lgm = _make_simple_lgm()
        spec = lgm.hyperparameter_spec
        # τ = 0 maps to log(0) = -Inf in working space.
        @test_throws ArgumentError resolve_mode_starts((; τ = 0.0), spec)
    end

    @testset "Single NamedTuple mode_init drives BFGS from that point" begin
        lgm = _make_simple_lgm()
        y = [true, false, true, true, false, false]
        # Use an obviously off-default start in natural space; BFGS should
        # still converge to ~the same mode (the model is unimodal here).
        θ_star, _, _, mode_info = Latte.find_hyperparameter_mode(
            lgm, y; mode_init = (; τ = 5.0),
        )
        @test mode_info.n_starts == 1
        @test mode_info.best_start_index == 1
        # Same mode as default within tolerance.
        θ_default, _, _ = Latte.find_hyperparameter_mode(lgm, y)
        @test θ_star.θ[1] ≈ θ_default.θ[1] atol = 0.1
    end

    @testset "Multi-start mode_init picks the best by final log-density" begin
        lgm = _make_simple_lgm()
        y = [true, false, true, true, false, false]
        θ_default, _, _ = Latte.find_hyperparameter_mode(lgm, y)

        θ_star, _, _, mode_info = Latte.find_hyperparameter_mode(
            lgm, y; mode_init = [(; τ = 0.5), (; τ = 5.0), (; τ = 0.1)],
            iterations = 200,
        )
        @test mode_info.n_starts == 3
        @test 1 <= mode_info.best_start_index <= 3
        @test length(mode_info.final_logdensities) == 3
        @test all(isfinite, mode_info.final_logdensities)
        # The winner's log-density should be at least as good as any other
        # start's (within numerical noise).
        winning_logp = mode_info.final_logdensities[mode_info.best_start_index]
        @test all(winning_logp + 1.0e-6 .>= mode_info.final_logdensities)
        # And matches the unimodal default mode.
        @test θ_star.θ[1] ≈ θ_default.θ[1] atol = 0.05
    end

    @testset "RandomStarts(n) runs n starts, reproducible under rng" begin
        lgm = _make_two_hp_lgm()
        y = [true, false, true, true, false, false]
        rng_a = MersenneTwister(42)
        rng_b = MersenneTwister(42)
        θ_a, _, _, info_a = Latte.find_hyperparameter_mode(
            lgm, y; mode_init = RandomStarts(4; rng = rng_a), iterations = 100,
        )
        θ_b, _, _, info_b = Latte.find_hyperparameter_mode(
            lgm, y; mode_init = RandomStarts(4; rng = rng_b), iterations = 100,
        )
        @test info_a.n_starts == 4
        @test info_a.final_logdensities ≈ info_b.final_logdensities atol = 1.0e-8
        @test θ_a.θ ≈ θ_b.θ atol = 1.0e-8
    end

    @testset "mode_info fields are populated for the default single start" begin
        lgm = _make_simple_lgm()
        y = [true, false, true, true, false, false]
        _, _, _, mode_info = Latte.find_hyperparameter_mode(lgm, y)
        @test mode_info.n_starts == 1
        @test mode_info.best_start_index == 1
        @test length(mode_info.final_logdensities) == 1
        @test isfinite(mode_info.final_logdensities[1])
        @test mode_info.converged isa Bool
        @test mode_info.runner_up_gap === nothing || isfinite(mode_info.runner_up_gap)
    end

    @testset "runner_up_gap reports log-density gap between best and 2nd best" begin
        lgm = _make_simple_lgm()
        y = [true, false, true, true, false, false]
        _, _, _, mode_info = Latte.find_hyperparameter_mode(
            lgm, y; mode_init = [(; τ = 0.5), (; τ = 5.0)],
        )
        @test mode_info.runner_up_gap !== nothing
        @test mode_info.runner_up_gap >= -1.0e-6   # gap is best - second-best, ≥ 0 modulo noise
    end

    @testset "mode-quality diagnostic locates the mode via θ_star, not grid order" begin
        lgm = _make_simple_lgm()
        spec = lgm.hyperparameter_spec
        # Grid stored in ascending τ with the mode in the interior (highest
        # log-density at index 3) — the layout the default 1-D exploration
        # produces. The diagnostic must not assume grid_points[1] is the mode.
        τs = [2.5, 3.4, 4.6, 6.2, 8.4]
        logd = [-1.68, -0.21, 0.28, -0.22, -1.74]
        pts = [
            Latte.GridPoint(
                    convert(Latte.WorkingHyperparameters, Latte.NaturalHyperparameters([τ], spec)),
                    ld, nothing,
                ) for (τ, ld) in zip(τs, logd)
        ]
        expl = (; grid_points = pts)   # only `.grid_points` is read

        # θ* is the interior peak: the optimizer succeeded → no warning/error.
        θ_star = pts[3].θ
        @test Latte._diagnose_mode_quality(θ_star, expl, lgm, :error, 1.0) === nothing

        # θ* stuck at an edge worse than the interior peak → genuine failure fires.
        θ_stuck = pts[1].θ
        @test_throws ErrorException Latte._diagnose_mode_quality(θ_stuck, expl, lgm, :error, 1.0)
    end
end
