using Test
using Latte
using DynamicPPL: @model
using Distributions
using GaussianMarkovRandomFields
using LinearAlgebra
using Random

# Correctness cross-check: fast path must agree with the AD fallback to
# numerical precision on loglik at random x points, and the downstream
# `inla()` posterior must match within MC error.
@testset "Fast-path ↔ AD agreement" begin

    @testset "Poisson + LogLink: loglik matches AD fallback" begin
        @model function hier_poisson(y, X, group)
            τ_u ~ Gamma(2, 1)
            β ~ MvNormal(zeros(size(X, 2)), 100.0 * I)
            u ~ MvNormal(zeros(maximum(group)), (1 / τ_u) * I)
            for i in eachindex(y)
                y[i] ~ Poisson(exp(X[i, :] ⋅ β + u[group[i]]); check_args = false)
            end
        end
        Random.seed!(2026)
        n, p, G = 40, 2, 5
        X = [ones(n) randn(n)]
        group = rand(1:G, n)
        β_true = [0.3, 0.5]
        u_true = randn(G) ./ 2
        y_obs = [
            rand(Poisson(exp(X[i, :] ⋅ β_true + u_true[group[i]])))
                for i in 1:n
        ]

        lgm_fast = latte_from_dppl(hier_poisson(y_obs, X, group); random = (:β, :u))
        lgm_ad = latte_from_dppl(
            hier_poisson(y_obs, X, group);
            random = (:β, :u), force_ad_obs_model = true,
        )

        # Evaluate hyperparameter_logpdf at 5 random working-space θ; the
        # two LGMs should agree to ~1e-10 (same prior, equivalent obs model).
        spec = lgm_fast.hyperparameter_spec
        y_wrap = PoissonObservations(y_obs)

        ws_fast = make_workspace(lgm_fast.latent_prior; τ_u = 1.0)
        ws_ad = make_workspace(lgm_ad.latent_prior; τ_u = 1.0)

        Random.seed!(7)
        for _ in 1:5
            θ_vec = [randn()]
            wh = Latte.WorkingHyperparameters(θ_vec, spec)
            lp_fast = Latte.hyperparameter_logpdf(lgm_fast, wh, y_wrap; ws = ws_fast)
            lp_ad = Latte.hyperparameter_logpdf(lgm_ad, wh, y_wrap; ws = ws_ad)
            @test lp_fast ≈ lp_ad rtol = 1.0e-6
        end
    end

    @testset "inla() posterior marginals match between paths" begin
        @model function hier_poisson(y, X, group)
            τ_u ~ Gamma(2, 1)
            β ~ MvNormal(zeros(size(X, 2)), 100.0 * I)
            u ~ MvNormal(zeros(maximum(group)), (1 / τ_u) * I)
            for i in eachindex(y)
                y[i] ~ Poisson(exp(X[i, :] ⋅ β + u[group[i]]); check_args = false)
            end
        end
        Random.seed!(2026)
        n, p, G = 30, 2, 4
        X = [ones(n) randn(n)]
        group = rand(1:G, n)
        β_true = [0.3, 0.5]
        u_true = randn(G) ./ 2
        y_obs = [
            rand(Poisson(exp(X[i, :] ⋅ β_true + u_true[group[i]])))
                for i in 1:n
        ]

        lgm_fast = latte_from_dppl(hier_poisson(y_obs, X, group); random = (:β, :u))
        lgm_ad = latte_from_dppl(
            hier_poisson(y_obs, X, group);
            random = (:β, :u), force_ad_obs_model = true,
        )

        # Both paths work under FiniteDiff + GaussianMarginal. (Default
        # AutoMarginal works on fast-path but not AD-path — that's the
        # separate nested-AD issue in AutoDiffObservationModel).
        res_fast = inla(
            lgm_fast, y_obs; progress = false,
            diff_strategy = FiniteDiffStrategy(),
            latent_marginalization_method = GaussianMarginal(),
            accumulators = (),
        )
        res_ad = inla(
            lgm_ad, y_obs; progress = false,
            diff_strategy = FiniteDiffStrategy(),
            latent_marginalization_method = GaussianMarginal(),
            accumulators = (),
        )

        # Hp mode and latent means should match tightly.
        mode_fast = convert(NamedTuple, hyperparameter_mode(res_fast))
        mode_ad = convert(NamedTuple, hyperparameter_mode(res_ad))
        @test mode_fast.τ_u ≈ mode_ad.τ_u rtol = 1.0e-4

        # Fast-path LGM is auto-augmented (latent = η ⊕ x_base, n_obs + n_base),
        # AD-path LGM is not (latent = x_base only). Compare the base
        # (β, u) marginals, which have the same dimension in both cases.
        means_fast = mean.(res_fast.base_latent_marginals)
        means_ad = mean.(latent_marginals(res_ad))
        @test maximum(abs.(means_fast .- means_ad)) < 1.0e-3
    end

    @testset "Poisson+offset fast path agrees with AD path" begin
        @model function poisson_with_exposure(y, X, log_exposure)
            τ ~ Gamma(2, 1)
            β ~ MvNormal(zeros(size(X, 2)), (1 / τ) * I(size(X, 2)))
            for i in eachindex(y)
                y[i] ~ Poisson(exp(X[i, :] ⋅ β + log_exposure[i]); check_args = false)
            end
        end
        Random.seed!(11)
        n = 30
        X = [ones(n) randn(n)]
        log_exposure = randn(n) .* 0.5
        β_true = [0.3, 0.5]
        y_obs = [
            rand(Poisson(exp(X[i, :] ⋅ β_true + log_exposure[i])))
                for i in 1:n
        ]

        lgm_fast = latte_from_dppl(
            poisson_with_exposure(y_obs, X, log_exposure);
            random = (:β,),
        )
        lgm_ad = latte_from_dppl(
            poisson_with_exposure(y_obs, X, log_exposure);
            random = (:β,), force_ad_obs_model = true,
        )

        # Compare hyperparameter_logpdf at 3 random θ values; fast path
        # routes the offset through AugmentedLatentModel's mean, AD path
        # runs the DPPL likelihood directly. They should agree up to the
        # constant additive term ∑ y_i · log(exposure_i) that Latte's
        # Poisson obs model drops (same constant for both log-likelihoods
        # at a given dataset, so cancels in *differences* — i.e. gradients
        # and posterior shapes match even if absolute values differ).
        spec = lgm_fast.hyperparameter_spec
        y_wrap = PoissonObservations(y_obs)

        ws_fast = make_workspace(lgm_fast.latent_prior; τ = 1.0)
        ws_ad = make_workspace(lgm_ad.latent_prior; τ = 1.0)

        Random.seed!(12)
        vals_fast = Float64[]
        vals_ad = Float64[]
        for _ in 1:3
            θ_vec = [randn()]
            wh = Latte.WorkingHyperparameters(θ_vec, spec)
            push!(vals_fast, Latte.hyperparameter_logpdf(lgm_fast, wh, y_wrap; ws = ws_fast))
            push!(vals_ad, Latte.hyperparameter_logpdf(lgm_ad, wh, y_wrap; ws = ws_ad))
        end
        # Differences between logpdf values at the 3 θs should match
        # across the two paths (log-posterior shape, up to additive const)
        diffs_fast = vals_fast .- vals_fast[1]
        diffs_ad = vals_ad .- vals_ad[1]
        @test maximum(abs.(diffs_fast .- diffs_ad)) < 1.0e-6
    end

    @testset "Fast path works with default AutoMarginal (AD path doesn't)" begin
        # Regression test for the specific workaround we eliminated: the
        # fast path accepts default `AutoMarginal` + FiniteDiff; the AD
        # obs model does not (nested-AD bug in AutoDiffObservationModel).
        @model function m(y, X, group)
            τ_u ~ Gamma(2, 1)
            β ~ MvNormal(zeros(size(X, 2)), 100.0 * I(size(X, 2)))
            u ~ MvNormal(zeros(maximum(group)), (1 / τ_u) * I(maximum(group)))
            for i in eachindex(y)
                y[i] ~ Poisson(exp(X[i, :] ⋅ β + u[group[i]]); check_args = false)
            end
        end
        Random.seed!(2026)
        n, G = 20, 3
        X = [ones(n) randn(n)]
        group = rand(1:G, n)
        y_obs = [rand(Poisson(exp(X[i, :] ⋅ [0.3, 0.5]))) for i in 1:n]

        lgm_fast = latte_from_dppl(m(y_obs, X, group); random = (:β, :u))

        # Default AutoMarginal + FiniteDiff: should succeed on fast path
        result = inla(lgm_fast, y_obs; progress = false, diff_strategy = FiniteDiffStrategy())
        @test result isa Latte.INLAResult
    end
end
