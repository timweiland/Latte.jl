using Test
using Latte
using DynamicPPL: @model
using Distributions
using LinearAlgebra
using Random

# End-to-end validation: write a DPPL model, hand it to `latte_from_dppl`,
# then run both `inla()` and `tmb()` on the resulting LGM. Exercises the
# shared InferenceResult protocol across a DPPL-derived model.
@testset "DPPL → latte_from_dppl → inla + tmb" begin
    @model function hier_poisson(y, X, group)
        n = length(y)
        p = size(X, 2)
        G = maximum(group)
        τ_u ~ Gamma(2, 1)
        β ~ MvNormal(zeros(p), 100.0 * I(p))
        u ~ MvNormal(zeros(G), (1 / τ_u) * I(G))
        η = X * β .+ u[group]
        for i in 1:n
            y[i] ~ Poisson(exp(η[i]); check_args = false)
        end
    end

    Random.seed!(2026)
    n, p, G = 40, 2, 5
    X = [ones(n) randn(n)]
    group = rand(1:G, n)
    β_true = [0.3, 0.5]
    u_true = randn(G) ./ sqrt(4.0)
    η_true = X * β_true .+ u_true[group]
    y_obs = [rand(Poisson(exp(η))) for η in η_true]

    dppl = hier_poisson(y_obs, X, group)
    model = latte_from_dppl(dppl; random = (:β, :u))

    # FiniteDiffStrategy is still required here — the DPPL adapter's
    # latent_fn closure doesn't survive ForwardDiff of the outer objective
    # (separate nested-AD issue, tracked in tasks/). The previous
    # `GaussianMarginal` + `accumulators=()` workaround was for the
    # AutoDiffObservationModel's *own* nested-AD bug, which the fast-path
    # Poisson detection now bypasses: default AutoMarginal and the full
    # accumulator suite work once the obs model is an ExponentialFamily.
    inla_r = inla(
        model, y_obs;
        progress = false,
        diff_strategy = FiniteDiffStrategy(),
    )
    tmb_r = tmb(model, y_obs)

    # The fast path produces a LinearlyTransformedObservationModel → LGM's
    # auto-augmentation wraps the latent field as [η₁…η_n; β; u]. All length
    # checks below use `n + p + G` and base-component slicing via
    # `augmentation_info.base_latent_indices`.
    base_idx = model.augmentation_info.base_latent_indices

    @testset "Both results satisfy the protocol" begin
        for r in (inla_r, tmb_r)
            @test r isa Latte.InferenceResult
            @test length(latent_marginals(r)) == n + p + G
            @test length(hyperparameter_marginals(r)) == 1
            @test haskey(hyperparameter_groups(r), :τ_u)
            @test converged(r)
        end
    end

    @testset "Both methods agree on the MAP" begin
        inla_mode = convert(NamedTuple, hyperparameter_mode(inla_r)).τ_u
        tmb_mode = convert(NamedTuple, hyperparameter_mode(tmb_r)).τ_u
        @test inla_mode ≈ tmb_mode rtol = 1.0e-4
    end

    @testset "Latent posterior recovers truth within ~3 SE" begin
        true_x = vcat(β_true, u_true)
        for r in (inla_r, tmb_r)
            base_marginals = latent_marginals(r)[base_idx]
            means = mean.(base_marginals)
            stds = std.(base_marginals)
            # All base components should be within 3 SE of the truth (loose
            # sanity check — not a calibration test).
            for i in eachindex(true_x)
                @test abs(means[i] - true_x[i]) < 3 * stds[i]
            end
        end
    end

    @testset "rand returns PosteriorSamples for both" begin
        for r in (inla_r, tmb_r)
            s = rand(MersenneTwister(1), r, 5)
            @test s isa PosteriorSamples
            @test size(s.x, 2) == n + p + G
            @test size(s.x, 1) == 5
        end
    end
end
