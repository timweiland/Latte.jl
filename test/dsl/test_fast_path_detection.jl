using Test
using Latte
using DynamicPPL: @model
using Distributions
using GaussianMarkovRandomFields
using LinearAlgebra
using Random

# Does `latte_from_dppl` correctly detect the fast path for supported
# families, and fall through to the AD wrapping otherwise?
#
# Each model needs ≥ 1 hyperparameter (Latte requires it). We include a
# τ_u on a random-effect covariance to give the adapter something to pin.
@testset "Fast-path detection" begin

    @testset "Poisson + LogLink fires fast path" begin
        @model function m_poisson(y, X, group)
            τ_u ~ Gamma(2, 1)
            β ~ MvNormal(zeros(size(X, 2)), 100.0 * I(size(X, 2)))
            u ~ MvNormal(zeros(maximum(group)), (1 / τ_u) * I(maximum(group)))
            for i in eachindex(y)
                y[i] ~ Poisson(exp(X[i, :] ⋅ β + u[group[i]]); check_args = false)
            end
        end
        Random.seed!(1)
        n, p, G = 30, 2, 3
        X = [ones(n) randn(n)]
        group = rand(1:G, n)
        β_true = [0.3, 0.5]
        u_true = randn(G) ./ 2
        y_obs = [rand(Poisson(exp(X[i, :] ⋅ β_true + u_true[group[i]]))) for i in 1:n]

        lgm = latte_from_dppl(m_poisson(y_obs, X, group); random = (:β, :u), augment = true)
        @test lgm.observation_model isa ExponentialFamily{Poisson, LogLink}
        @test lgm.augmentation_info !== nothing
    end

    @testset "Bernoulli + LogitLink fires fast path" begin
        @model function m_bernoulli(y, X, group)
            τ_u ~ Gamma(2, 1)
            β ~ MvNormal(zeros(size(X, 2)), 100.0 * I(size(X, 2)))
            u ~ MvNormal(zeros(maximum(group)), (1 / τ_u) * I(maximum(group)))
            for i in eachindex(y)
                p_i = 1 / (1 + exp(-(X[i, :] ⋅ β + u[group[i]])))
                y[i] ~ Bernoulli(p_i; check_args = false)
            end
        end
        Random.seed!(2)
        n, p, G = 30, 2, 3
        X = [ones(n) randn(n)]
        group = rand(1:G, n)
        β_true = [0.1, 0.4]
        u_true = randn(G) ./ 2
        y_obs = [
            rand(Bernoulli(1 / (1 + exp(-(X[i, :] ⋅ β_true + u_true[group[i]])))))
                for i in 1:n
        ]

        lgm = latte_from_dppl(m_bernoulli(y_obs, X, group); random = (:β, :u), augment = true)
        @test lgm.observation_model isa ExponentialFamily{Bernoulli, LogitLink}
    end

    @testset "Binomial + LogitLink fires fast path with per-site trials" begin
        @model function m_binomial(y, X, group, trials)
            τ_u ~ Gamma(2, 1)
            β ~ MvNormal(zeros(size(X, 2)), 100.0 * I(size(X, 2)))
            u ~ MvNormal(zeros(maximum(group)), (1 / τ_u) * I(maximum(group)))
            for i in eachindex(y)
                p_i = 1 / (1 + exp(-(X[i, :] ⋅ β + u[group[i]])))
                y[i] ~ Binomial(trials[i], p_i; check_args = false)
            end
        end
        Random.seed!(8)
        n, p, G = 30, 2, 3
        X = [ones(n) randn(n)]
        group = rand(1:G, n)
        trials = rand(5:20, n)   # heterogeneous per-site trial counts
        β_true = [0.1, 0.4]
        u_true = randn(G) ./ 2
        y_obs = [
            rand(Binomial(trials[i], 1 / (1 + exp(-(X[i, :] ⋅ β_true + u_true[group[i]])))))
                for i in 1:n
        ]

        lgm = latte_from_dppl(
            m_binomial(y_obs, X, group, trials); random = (:β, :u), augment = true,
        )
        # Augmented fast path stores the BinomialTrials wrapper (carrying the
        # per-site trial counts) as the observation model; the inner base is
        # ExponentialFamily(Binomial, LogitLink).
        @test lgm.observation_model isa Latte.BinomialTrialsObservationModel
        @test lgm.observation_model.base isa ExponentialFamily{Binomial, LogitLink}
        @test lgm.observation_model.trials == trials
    end

    @testset "force_ad_obs_model=true takes the AD path" begin
        @model function m_poisson(y, X, group)
            τ_u ~ Gamma(2, 1)
            β ~ MvNormal(zeros(size(X, 2)), 100.0 * I(size(X, 2)))
            u ~ MvNormal(zeros(maximum(group)), (1 / τ_u) * I(maximum(group)))
            for i in eachindex(y)
                y[i] ~ Poisson(exp(X[i, :] ⋅ β + u[group[i]]); check_args = false)
            end
        end
        Random.seed!(3)
        n, G = 20, 3
        X = [ones(n) randn(n)]
        group = rand(1:G, n)
        y_obs = [rand(Poisson(exp(X[i, :] ⋅ [0.3, 0.5]))) for i in 1:n]

        lgm = latte_from_dppl(
            m_poisson(y_obs, X, group);
            random = (:β, :u), force_ad_obs_model = true,
        )
        @test !(lgm.observation_model isa ExponentialFamily)
    end

    @testset "Poisson with log-exposure offset fires fast path" begin
        @model function poisson_with_exposure(y, X, log_exposure)
            τ ~ Gamma(2, 1)
            β ~ MvNormal(zeros(size(X, 2)), (1 / τ) * I(size(X, 2)))
            for i in eachindex(y)
                y[i] ~ Poisson(exp(X[i, :] ⋅ β + log_exposure[i]); check_args = false)
            end
        end
        Random.seed!(5)
        n = 30
        X = [ones(n) randn(n)]
        log_exposure = randn(n) .* 0.5
        β_true = [0.3, 0.5]
        y_obs = [
            rand(Poisson(exp(X[i, :] ⋅ β_true + log_exposure[i])))
                for i in 1:n
        ]

        lgm = latte_from_dppl(poisson_with_exposure(y_obs, X, log_exposure); random = (:β,), augment = true)
        # Fast path with a non-zero offset puts it on the LTM (η = A·x + b);
        # LGM's auto-augmentation absorbs that offset into the augmented prior
        # mean, leaving the base ExponentialFamily as the observation model.
        @test lgm.observation_model isa ExponentialFamily{Poisson, LogLink}
        @test lgm.latent_prior isa Latte.AugmentedLatentModel
        # Detected offset matches the log-exposure vector exactly
        @test lgm.latent_prior.offset ≈ log_exposure rtol = 1.0e-10
    end

    @testset "augment=false produces a non-augmented LGM for TMB speed" begin
        @model function poisson_glm(y, X)
            τ ~ Gamma(2, 1)
            β ~ MvNormal(zeros(size(X, 2)), (1 / τ) * I(size(X, 2)))
            for i in eachindex(y)
                y[i] ~ Poisson(exp(X[i, :] ⋅ β); check_args = false)
            end
        end
        Random.seed!(6)
        n, p = 30, 2
        X = [ones(n) randn(n)]
        β_true = [0.3, 0.5]
        y_obs = [rand(Poisson(exp(X[i, :] ⋅ β_true))) for i in 1:n]

        lgm_aug = latte_from_dppl(poisson_glm(y_obs, X); random = (:β,), augment = true)
        lgm_noaug = latte_from_dppl(poisson_glm(y_obs, X); random = (:β,), augment = false)

        # Augmented: latent_dim = n + p (η + x_base)
        @test length(lgm_aug.latent_prior) == n + p
        @test lgm_aug.augmentation_info !== nothing
        # Non-augmented: latent_dim = p only
        @test length(lgm_noaug.latent_prior) == p
        @test lgm_noaug.augmentation_info === nothing
        @test lgm_noaug.observation_model isa LinearlyTransformedObservationModel

        # Both should give the same TMB results (just different inner Laplace dim)
        r_aug = tmb(lgm_aug, y_obs)
        r_noaug = tmb(lgm_noaug, y_obs)
        τ_aug = convert(NamedTuple, hyperparameter_mode(r_aug)).τ
        τ_noaug = convert(NamedTuple, hyperparameter_mode(r_noaug)).τ
        @test τ_aug ≈ τ_noaug rtol = 1.0e-6
        @test log_marginal_likelihood(r_aug) ≈ log_marginal_likelihood(r_noaug) rtol = 1.0e-6
        # Base (β) posterior means match
        base_idx = lgm_aug.augmentation_info.base_latent_indices
        β_aug = mean.(latent_marginals(r_aug))[base_idx]
        β_noaug = mean.(latent_marginals(r_noaug))
        @test maximum(abs.(β_aug .- β_noaug)) < 1.0e-4
    end

    @testset "NegativeBinomial + LogLink fires fast path" begin
        @model function m_nb(y, X, group)
            r ~ Gamma(2, 1)
            τ_u ~ Gamma(2, 1)
            β ~ MvNormal(zeros(size(X, 2)), 100.0 * I(size(X, 2)))
            u ~ MvNormal(zeros(maximum(group)), (1 / τ_u) * I(maximum(group)))
            for i in eachindex(y)
                μ_i = exp(X[i, :] ⋅ β + u[group[i]])
                y[i] ~ NegativeBinomial(r, r / (r + μ_i); check_args = false)
            end
        end
        Random.seed!(20)
        n, p, G = 30, 2, 3
        X = [ones(n) randn(n)]
        group = rand(1:G, n)
        β_true = [0.3, 0.4]
        u_true = randn(G) ./ 2
        r_true = 5.0
        y_obs = [
            (
                    μ = exp(X[i, :] ⋅ β_true + u_true[group[i]]);
                    rand(NegativeBinomial(r_true, r_true / (r_true + μ)))
                )
                for i in 1:n
        ]

        lgm = latte_from_dppl(m_nb(y_obs, X, group); random = (:β, :u), augment = true)
        @test lgm.observation_model isa ExponentialFamily{NegativeBinomial, LogLink}
        @test lgm.augmentation_info !== nothing
    end

    @testset "Gamma + LogLink fires fast path" begin
        @model function m_gamma(y, X, group)
            phi ~ Gamma(2, 1)
            τ_u ~ Gamma(2, 1)
            β ~ MvNormal(zeros(size(X, 2)), 100.0 * I(size(X, 2)))
            u ~ MvNormal(zeros(maximum(group)), (1 / τ_u) * I(maximum(group)))
            for i in eachindex(y)
                μ_i = exp(X[i, :] ⋅ β + u[group[i]])
                y[i] ~ Gamma(phi, μ_i / phi; check_args = false)
            end
        end
        Random.seed!(21)
        n, p, G = 30, 2, 3
        X = [ones(n) randn(n)]
        group = rand(1:G, n)
        β_true = [0.2, 0.3]
        u_true = randn(G) ./ 2
        phi_true = 2.0
        y_obs = [
            (
                    μ = exp(X[i, :] ⋅ β_true + u_true[group[i]]);
                    rand(Gamma(phi_true, μ / phi_true))
                )
                for i in 1:n
        ]

        lgm = latte_from_dppl(m_gamma(y_obs, X, group); random = (:β, :u), augment = true)
        @test lgm.observation_model isa ExponentialFamily{Gamma, LogLink}
        @test lgm.augmentation_info !== nothing
    end

    @testset "Mixed-family likelihood falls through to AD path" begin
        # Half the sites Poisson, half Normal — fast path demands a
        # homogeneous family and must punt.
        @model function m_mixed(y, z, X, group)
            τ_u ~ Gamma(2, 1)
            β ~ MvNormal(zeros(size(X, 2)), 100.0 * I(size(X, 2)))
            u ~ MvNormal(zeros(maximum(group)), (1 / τ_u) * I(maximum(group)))
            for i in eachindex(y)
                y[i] ~ Poisson(exp(X[i, :] ⋅ β + u[group[i]]); check_args = false)
            end
            for i in eachindex(z)
                z[i] ~ Normal(X[i, :] ⋅ β + u[group[i]], 1.0)
            end
        end
        Random.seed!(4)
        n, G = 10, 3
        X = [ones(n) randn(n)]
        group = rand(1:G, n)
        y_obs = rand(Poisson(2.0), n)
        z_obs = randn(n)

        lgm = latte_from_dppl(m_mixed(y_obs, z_obs, X, group); random = (:β, :u))
        @test !(lgm.observation_model isa ExponentialFamily)
    end
end
