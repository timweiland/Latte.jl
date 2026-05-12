using Test
using Latte
using DynamicPPL
using Distributions
using LinearAlgebra
using Random
using GaussianMarkovRandomFields
import GaussianMarkovRandomFields as GMRF
using Statistics: mean

# End-to-end coverage of the prelude-lift path: confirms the macro emits
# the lift callables, that the LGM uses the lifted obs model wrapper, that
# the materialised obs likelihood agrees with a DPPL-AD reference, and that
# `inla()` completes through the whole pipeline.

@testset "Prelude-lift end-to-end" begin
    Random.seed!(20260512)

    @testset "Composite obs (two channels with distinct nuisance σ) — lifted" begin
        @latte function lift_e2e_two_chan(y_a, y_b, A_a, A_b)
            σ_a ~ Gamma(2.0, 1.0)
            σ_b ~ Gamma(2.0, 1.0)
            # Heavy-ish hp-dependent prelude operation: precomputed scale.
            scale = exp(-(σ_a^2 + σ_b^2) / 100)
            @random β ~ MvNormal(zeros(size(A_a, 2)), 1.0)
            for i in eachindex(y_a)
                y_a[i] ~ Normal(scale * dot(A_a[i, :], β), σ_a)
            end
            for i in eachindex(y_b)
                y_b[i] ~ Normal(dot(A_b[i, :], β), σ_b)
            end
        end

        n_a, n_b, p = 5, 4, 3
        A_a = randn(n_a, p)
        A_b = randn(n_b, p)
        y_a = randn(n_a)
        y_b = randn(n_b)
        lgm = lift_e2e_two_chan(y_a, y_b, A_a, A_b)

        # Lift fires only for components that need AD; the second group's
        # rhs is linear in β so its fast-path detection succeeds. We just
        # verify the wrapper is _LiftedCompositeObsModel — both lifted and
        # mixed cases are valid as long as the wrapper is correct.
        @test lgm.observation_model isa Latte._LiftedCompositeObsModel
        meta = Latte.latte_analysis(lift_e2e_two_chan)
        @test meta.lift_meta !== nothing
        @test :scale in meta.lift_meta.capture
        @test Set(meta.lift_meta.hp_syms) == Set([:σ_a, :σ_b])

        # Materialise and call loglik at a fixed β.
        β = randn(p)
        σ_a_val, σ_b_val = 0.7, 1.3
        obs_lik = lgm.observation_model(vcat(y_a, y_b); σ_a = σ_a_val, σ_b = σ_b_val)
        ll = GMRF.loglik(β, obs_lik)

        # Analytic reference (matches what the body computes).
        scale_ref = exp(-(σ_a_val^2 + σ_b_val^2) / 100)
        expected = sum(
            logpdf(Normal(scale_ref * dot(A_a[i, :], β), σ_a_val), y_a[i])
                for i in eachindex(y_a)
        ) + sum(
            logpdf(Normal(dot(A_b[i, :], β), σ_b_val), y_b[i])
                for i in eachindex(y_b)
        )
        @test ll ≈ expected atol = 1.0e-10
    end

    @testset "Full inla() pipeline through lifted obs model" begin
        @latte function lift_e2e_inla(y_a, y_b, A_a, A_b)
            σ_a ~ Gamma(2.0, 1.0)
            σ_b ~ Gamma(2.0, 1.0)
            scale = exp(-(σ_a^2 + σ_b^2) / 100)
            @random β ~ MvNormal(zeros(size(A_a, 2)), 1.0)
            for i in eachindex(y_a)
                y_a[i] ~ Normal(scale * dot(A_a[i, :], β), σ_a)
            end
            for i in eachindex(y_b)
                y_b[i] ~ Normal(dot(A_b[i, :], β), σ_b)
            end
        end

        n_a, n_b, p = 6, 6, 2
        A_a = randn(n_a, p)
        A_b = randn(n_b, p)
        β_true = randn(p)
        y_a = A_a * β_true .+ 0.05 .* randn(n_a)
        y_b = A_b * β_true .+ 0.05 .* randn(n_b)
        lgm = lift_e2e_inla(y_a, y_b, A_a, A_b)
        @test lgm.observation_model isa Latte._LiftedCompositeObsModel

        # FiniteDiff sidesteps an unrelated outer-AD-over-hp issue that
        # affects FunctionalGPs models — irrelevant here but matches the
        # canonical user-flow for non-Dual-clean kernel libraries.
        result = inla(lgm, vcat(y_a, y_b); diff_strategy = Latte.FiniteDiffStrategy())
        @test result isa Latte.INLAResult
        @test isfinite(mean(result.hyperparameter_marginals[:σ_a]))
        @test isfinite(mean(result.hyperparameter_marginals[:σ_b]))
    end

    @testset "Signature with kwargs disables lift, falls back to DPPL" begin
        @latte function lift_kwargs_fallback(y; sigma_prior_scale = 1.0)
            σ ~ Gamma(2.0, sigma_prior_scale)
            @random β ~ MvNormal(zeros(2), 1.0)
            for i in eachindex(y)
                y[i] ~ Normal(β[i], σ)
            end
        end
        meta = Latte.latte_analysis(lift_kwargs_fallback)
        @test meta.lift_meta === nothing
        lgm = lift_kwargs_fallback(randn(2))
        # The body is fast-path-eligible so the wrapper is a fast-path obs
        # model — but critically NOT the lifted variant.
        @test !(lgm.observation_model isa Latte._LiftedCompositeObsModel)
        @test !(lgm.observation_model isa Latte._LiftedSingleObsModel)
    end

    @testset "Single-group lifted: fast-path-rejected single AD model" begin
        # Force the AD fallback by making the linear predictor non-static
        # — `r_i` depends on σ multiplicatively, which the fast-path detector
        # rejects (σ enters the mean, not just the dispersion). Single group
        # since all obs have the same family + deps. This is the only path
        # that actually exercises `_LiftedSingleObsModel`.
        @latte function lift_e2e_single_ad(y)
            σ ~ Gamma(2.0, 1.0)
            scale = log(1 + σ)            # hp-dep prelude
            @random β ~ MvNormal(zeros(3), 1.0)
            for i in eachindex(y)
                y[i] ~ Normal(scale * β[i], 0.1)
            end
        end

        y = randn(3)
        lgm = lift_e2e_single_ad(y)
        @test lgm.observation_model isa Latte._LiftedSingleObsModel
        meta = Latte.latte_analysis(lift_e2e_single_ad)
        @test meta.lift_meta !== nothing
        @test :scale in meta.lift_meta.capture

        β = [0.2, -0.4, 0.1]
        σ_val = 0.8
        obs_lik = lgm.observation_model(y; σ = σ_val)
        ll = GMRF.loglik(β, obs_lik)
        scale_ref = log(1 + σ_val)
        expected = sum(
            logpdf(Normal(scale_ref * β[i], 0.1), y[i]) for i in eachindex(y)
        )
        @test ll ≈ expected atol = 1.0e-10
    end
end
