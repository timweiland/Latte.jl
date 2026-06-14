using Test
using Latte
using DynamicPPL: @model
import DynamicPPL
using Distributions
using LinearAlgebra
using SparseArrays
using Random
using GaussianMarkovRandomFields

@testset "@latte macro" begin
    Random.seed!(20260508)

    @testset "1. Simple Gaussian regression — auto-detect baseline" begin
        @latte function simple_gaussian_regression(y, X)
            σ ~ Gamma(2.0, 1.0)
            β ~ MvNormal(zeros(size(X, 2)), 100.0 * I(size(X, 2)))
            for i in eachindex(y)
                y[i] ~ Normal(dot(X[i, :], β), σ)
            end
        end

        n, p = 6, 2
        X = [ones(n) randn(n)]
        y = randn(n)
        lgm = simple_gaussian_regression(y, X)

        @test lgm isa Latte.LatentGaussianModel
        @test keys(lgm.hyperparameter_spec.free) == (:σ,)
        @test :β in keys(lgm.latent_layout)
        meta = Latte.latte_analysis(simple_gaussian_regression)
        @test meta.random_syms == (:β,)
        @test meta.fixed_syms == (:σ,)
    end

    @testset "2. Two-channel Gaussian — composite obs auto-grouped" begin
        @latte function two_channel(y_phys, y_sensor, A_phys, A_sensor)
            σ_phys ~ Gamma(2.0, 1.0)
            σ_data ~ Gamma(2.0, 1.0)
            β ~ MvNormal(zeros(size(A_phys, 2)), 100.0 * I(size(A_phys, 2)))
            for i in eachindex(y_phys)
                y_phys[i] ~ Normal(dot(A_phys[i, :], β), σ_phys)
            end
            for i in eachindex(y_sensor)
                y_sensor[i] ~ Normal(dot(A_sensor[i, :], β), σ_data)
            end
        end

        n_phys, n_sensor, p = 5, 4, 3
        A_phys = randn(n_phys, p)
        A_sensor = randn(n_sensor, p)
        y_phys = randn(n_phys)
        y_sensor = randn(n_sensor)
        lgm = two_channel(y_phys, y_sensor, A_phys, A_sensor)

        @test Set(keys(lgm.hyperparameter_spec.free)) == Set([:σ_phys, :σ_data])
        composite = Latte._underlying_composite(lgm.observation_model)
        @test composite isa Latte.GaussianMarkovRandomFields.CompositeObservationModel
        @test length(composite.components) == 2
    end

    @testset "3. @random / @fixed markers override defaults" begin
        # `α` is scalar Normal — by default would be classified as `@fixed`
        # (scalar prior). Marker forces it to be a random effect.
        @latte function marker_override(y, X)
            σ ~ Gamma(2.0, 1.0)
            @random α ~ Normal(0.0, 1.0)
            β ~ MvNormal(zeros(size(X, 2)), 100.0 * I(size(X, 2)))
            for i in eachindex(y)
                y[i] ~ Normal(α + dot(X[i, :], β), σ)
            end
        end

        meta = Latte.latte_analysis(marker_override)
        @test :α in meta.random_syms
        @test :σ in meta.fixed_syms
        @test :β in meta.random_syms
        @test :α ∉ meta.fixed_syms
    end

    @testset "4. @fixed multivariate hp prior" begin
        # `θ` is multivariate — by default would be classified as `@random`.
        # Marker forces it to be a fixed effect (hp).
        @latte function fixed_mv_hp(y, X)
            @fixed θ ~ MvNormal(zeros(2), I(2))
            σ ~ Gamma(2.0, 1.0)
            β ~ MvNormal(zeros(size(X, 2)), 100.0 * I(size(X, 2)))
            for i in eachindex(y)
                y[i] ~ Normal(θ[1] + θ[2] * dot(X[i, :], β), σ)
            end
        end

        meta = Latte.latte_analysis(fixed_mv_hp)
        @test :θ in meta.fixed_syms
        @test :θ ∉ meta.random_syms
        @test :β in meta.random_syms
    end

    @testset "5. Turing handoff via dppl_model accessor" begin
        @latte function turing_handoff(y, X)
            σ ~ Gamma(2.0, 1.0)
            β ~ MvNormal(zeros(size(X, 2)), 100.0 * I(size(X, 2)))
            for i in eachindex(y)
                y[i] ~ Normal(dot(X[i, :], β), σ)
            end
        end

        n, p = 5, 2
        X = [ones(n) randn(n)]
        y = randn(n)

        # The DPPL model constructor is registered and produces a Model.
        dppl_ctor = Latte.dppl_model(turing_handoff)
        dppl = dppl_ctor(y, X)
        @test dppl isa DynamicPPL.Model

        # Logdensity evaluates without error (uses the same body the
        # Latte path uses).
        priors = DynamicPPL.extract_priors(dppl)
        @test !isempty(priors)
    end

    @testset "6. Inter-tilde arbitrary code — alias resolution" begin
        # The free-symbol pass tracks aliases through assignment chains, so
        # transitive hp dependencies via local-variable computation are still
        # detected as obs deps.
        helper(σ_local, x) = σ_local .* x

        @latte function inter_tilde(y, X)
            σ ~ Gamma(2.0, 1.0)
            β ~ MvNormal(zeros(size(X, 2)), 100.0 * I(size(X, 2)))
            scale = σ
            transformed = helper(scale, β)
            for i in eachindex(y)
                y[i] ~ Normal(dot(X[i, :], transformed), 1.0)
            end
        end

        n, p = 5, 2
        X = [ones(n) randn(n)]
        y = randn(n)
        lgm = inter_tilde(y, X)

        @test :σ in keys(lgm.hyperparameter_spec.free)
        # `transformed` aliases through `helper(scale, β)` → `scale = σ` →
        # so the obs `~` should depend on σ even though σ never appears in
        # the obs RHS textually.
        meta = Latte.latte_analysis(inter_tilde)
        obs_deps = collect(Set(Symbol.(meta.obs_records[1][2])))
        @test :σ in obs_deps
    end

    @testset "7. Ambiguous repeated obs symbol — rejection" begin
        @latte function ambiguous(y, X)
            σ_a ~ Gamma(2.0, 1.0)
            σ_b ~ Gamma(2.0, 1.0)
            β ~ MvNormal(zeros(size(X, 2)), 100.0 * I(size(X, 2)))
            for i in eachindex(y)
                y[i] ~ Normal(dot(X[i, :], β), σ_a)
            end
            for i in eachindex(y)
                y[i] ~ Normal(dot(X[i, :], β), σ_b)
            end
        end

        n, p = 5, 2
        X = [ones(n) randn(n)]
        y = randn(n)
        @test_throws ArgumentError ambiguous(y, X)
    end

    @testset "8. Hp shadowing inside let — deps don't include shadowed name" begin
        @latte function shadowed(y, X)
            σ_outer ~ Gamma(2.0, 1.0)
            β ~ MvNormal(zeros(size(X, 2)), 100.0 * I(size(X, 2)))
            for i in eachindex(y)
                let σ_outer = 2.0
                    y[i] ~ Normal(dot(X[i, :], β), σ_outer)
                end
            end
        end

        n, p = 5, 2
        X = [ones(n) randn(n)]
        y = randn(n)
        lgm = shadowed(y, X)

        meta = Latte.latte_analysis(shadowed)
        obs_deps = collect(Set(Symbol.(meta.obs_records[1][2])))
        @test :σ_outer ∉ obs_deps
    end

    @testset "9. let-binding alias — obs deps follow alias to hp" begin
        @latte function let_alias(y, X)
            σ_obs ~ Gamma(2.0, 1.0)
            β ~ MvNormal(zeros(size(X, 2)), 100.0 * I(size(X, 2)))
            let s = σ_obs
                for i in eachindex(y)
                    y[i] ~ Normal(dot(X[i, :], β), s)
                end
            end
        end

        n, p = 5, 2
        X = [ones(n) randn(n)]
        y = randn(n)
        lgm = let_alias(y, X)

        meta = Latte.latte_analysis(let_alias)
        obs_deps = collect(Set(Symbol.(meta.obs_records[1][2])))
        @test :σ_obs in obs_deps
    end

    @testset "likelihood_hessian_pattern assembly kwarg" begin
        n, p = 30, 2
        X = [ones(n) randn(n)]
        β_true = [0.3, -0.5]
        y = exp.(X * β_true .+ 0.2 .* randn(n))   # positive — LogNormal support

        # LogNormal is not a fast-path family ⇒ AutoDiffObservationModel, so the
        # likelihood Hessian pattern matters. The assembly kwarg must be consumed
        # by the @latte wrapper (not forwarded to the model) and threaded through
        # to LGM assembly.
        @latte function lnorm_reg(y, X)
            σ ~ Gamma(2.0, 1.0)
            β ~ MvNormal(zeros(size(X, 2)), 10.0 * I(size(X, 2)))
            for i in eachindex(y)
                y[i] ~ LogNormal(dot(view(X, i, :), β), σ)
            end
        end

        lgm = lnorm_reg(y, X; likelihood_hessian_pattern = :dense)
        @test lgm isa Latte.LatentGaussianModel

        @model function lnorm_reg_dppl(y, X)
            σ ~ Gamma(2.0, 1.0)
            β ~ MvNormal(zeros(size(X, 2)), 10.0 * I(size(X, 2)))
            for i in eachindex(y)
                y[i] ~ LogNormal(dot(view(X, i, :), β), σ)
            end
        end
        ref = latte_from_dppl(
            lnorm_reg_dppl(y, X); random = :β, likelihood_hessian_pattern = :dense
        )

        r1 = inla(lgm, y; progress = false)
        r2 = inla(ref, y; progress = false)
        @test mean(r1.hyperparameter_marginals.σ) ≈ mean(r2.hyperparameter_marginals.σ) rtol = 1.0e-2
    end

    @testset "8. @latte usable without `using DynamicPPL`" begin
        # A fresh module that only `using Latte` (no `using DynamicPPL`) must
        # be able to define and fit an `@latte` model — the macro expansion
        # must not require `DynamicPPL` to be bound in the user's scope.
        m = Module(:NoDPPLUser)
        Core.eval(m, :(using Latte))
        Core.eval(m, :(using Distributions))
        Core.eval(m, :(using LinearAlgebra))
        @test !isdefined(m, :DynamicPPL)

        Core.eval(
            m,
            quote
                @latte function reg_no_dppl(y, X)
                    σ ~ Gamma(2.0, 1.0)
                    β ~ MvNormal(zeros(size(X, 2)), 100.0 * I(size(X, 2)))
                    for i in eachindex(y)
                        y[i] ~ Normal(dot(X[i, :], β), σ)
                    end
                end
            end,
        )

        n, p = 8, 2
        X = [ones(n) randn(n)]
        y = randn(n)
        lgm = Base.invokelatest(getfield(m, :reg_no_dppl), y, X)
        @test lgm isa Latte.LatentGaussianModel

        r = inla(lgm, y; progress = false)
        @test r isa Latte.InferenceResult
        @test converged(r)
    end
end
