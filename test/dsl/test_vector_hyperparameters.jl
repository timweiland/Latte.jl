using Test
using Latte
using DynamicPPL: @model
using Distributions
using LinearAlgebra
using Random
using Statistics

# Vector-valued hyperparameters through the DSL (issue #41): a non-random
# multivariate vector prior becomes one vector hyperparameter — via
# `latte_from_dppl` directly, and via `@latte` with the `@fixed` marker
# (an unmarked `MvNormal` on a scalar LHS still classifies as a latent
# fixed-effects block, unchanged).

@testset "DSL vector hyperparameters" begin
    Random.seed!(41)
    n = 6
    Σκ = [0.5 0.2; 0.2 0.5]
    y_obs = [0.9, -0.4, 1.2, -1.0, 1.7, -0.2]

    @testset "latte_from_dppl admits a multivariate vector prior as hp" begin
        @model function mv_hp(y)
            κ ~ MvNormal(zeros(2), [0.5 0.2; 0.2 0.5])
            x ~ MvNormal(zeros(6), exp(-κ[1]) * I(6))
            for i in 1:6
                y[i] ~ Normal(x[i], exp(κ[2] / 2))
            end
        end

        lgm = latte_from_dppl(mv_hp(y_obs); random = (:x,))
        @test lgm isa LatentGaussianModel
        @test keys(lgm.hyperparameter_spec.free) == (:κ,)
        @test length(lgm.hyperparameter_spec.free.κ.prior) == 2

        r = inla(lgm, y_obs)
        @test length(hyperparameter_marginals(r)) == 2
        @test hyperparameter_groups(r)[:κ] == 1:2
        @test all(isfinite, mean.(hyperparameter_marginals(r)))
    end

    @testset "@latte with @fixed marker on a vector prior" begin
        @latte function mv_latte(y)
            @fixed κ ~ MvNormal(zeros(2), [0.5 0.2; 0.2 0.5])
            x ~ MvNormal(zeros(6), exp(-κ[1]) * I(6))
            for i in 1:6
                y[i] ~ Normal(x[i], exp(κ[2] / 2))
            end
        end

        lgm = mv_latte(y_obs)
        @test lgm isa LatentGaussianModel
        @test keys(lgm.hyperparameter_spec.free) == (:κ,)
        @test length(lgm.hyperparameter_spec.free.κ.prior) == 2

        r = inla(lgm, y_obs)
        @test length(hyperparameter_marginals(r)) == 2
        @test all(isfinite, mean.(hyperparameter_marginals(r)))
    end

    @testset "Unmarked MvNormal on scalar LHS stays a latent block" begin
        # Regression guard: fixed-effects ceremony must not silently turn
        # into a hyperparameter now that the adapter admits vector priors.
        @latte function fe_model(y, X)
            β ~ MvNormal(zeros(2), 25.0 * I(2))
            τ ~ Gamma(2, 1)
            u ~ IIDModel(6)(τ = τ)
            for i in 1:6
                y[i] ~ Normal(X[i, :]' * β + u[i], 0.5)
            end
        end
        X = [ones(6) randn(MersenneTwister(2), 6)]
        lgm = fe_model(y_obs, X)
        @test keys(lgm.hyperparameter_spec.free) == (:τ,)
        @test haskey(Latte.latent_groups(lgm), :β)
    end

    @testset "Latent-only vector hp keeps the EF fast path" begin
        # The vector hp drives only the latent precision; the likelihood never
        # touches it, so the exponential-family fast path must still fire.
        @model function mv_hp_latent_only(y)
            κ ~ MvNormal(zeros(2), [0.5 0.2; 0.2 0.5])
            x ~ MvNormal(
                zeros(6),
                Diagonal(vcat(fill(exp(-κ[1]), 3), fill(exp(-κ[2]), 3)))
            )
            for i in 1:6
                y[i] ~ Poisson(exp(x[i]); check_args = false)
            end
        end
        y_pois = [1, 0, 2, 1, 3, 0]
        lgm = latte_from_dppl(mv_hp_latent_only(y_pois); random = (:x,), augment = true)
        @test lgm.observation_model isa ExponentialFamily{Poisson, LogLink}

        r = inla(lgm, y_pois)
        @test length(hyperparameter_marginals(r)) == 2
        @test all(isfinite, mean.(hyperparameter_marginals(r)))
    end

    @testset "Vector hp in the likelihood degrades to the AD obs model" begin
        # With only scalar hps this model takes the EF fast path; a vector hp
        # the likelihood depends on must degrade gracefully to the AD
        # observation model (no scalar route can carry it), not crash.
        @model function mv_hp_pois(y)
            κ ~ MvNormal(zeros(2), 0.5 * I(2))
            x ~ MvNormal(zeros(6), exp(-κ[1]) * I(6))
            for i in 1:6
                y[i] ~ Poisson(exp(x[i] + κ[2]); check_args = false)
            end
        end
        y_pois = [1, 0, 2, 1, 3, 0]
        lgm = latte_from_dppl(mv_hp_pois(y_pois); random = (:x,))
        @test lgm isa LatentGaussianModel
        @test !(lgm.observation_model isa ExponentialFamily)
        @test !(lgm.observation_model isa LinearlyTransformedObservationModel)

        r = inla(lgm, y_pois)
        @test length(hyperparameter_marginals(r)) == 2
        @test all(isfinite, mean.(hyperparameter_marginals(r)))
    end
end
