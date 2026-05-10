using Test
using Latte
using DynamicPPL: @model
import DynamicPPL
using GaussianMarkovRandomFields:
    AutoDiffObservationModel, CompositeObservationModel, CompositeLikelihood,
    CompositeObservations, loglik, loggrad, loghessian
using Distributions
using LinearAlgebra
using SparseArrays
using Random

# Two-channel Gaussian model: physics residual + sensor obs, separate σs.
# Mirrors the PDE-inverse-problem motivating example from
# `LATTE_INTEGRATION_LEARNINGS.org`. Both blocks are Normal but with distinct
# hyperparameters, so today's single-AD path forces a single shared σ.
@model function _two_channel(y_phys, y_sensor, A_phys, A_sensor)
    σ_phys ~ Gamma(2, 1)
    σ_data ~ Gamma(2, 1)
    β ~ MvNormal(zeros(size(A_phys, 2)), 100.0 * I)
    for i in eachindex(y_phys)
        y_phys[i] ~ Normal(dot(A_phys[i, :], β), σ_phys)
    end
    for i in eachindex(y_sensor)
        y_sensor[i] ~ Normal(dot(A_sensor[i, :], β), σ_data)
    end
end

@testset "obs_groups" begin
    Random.seed!(2026)
    n_phys, n_sensor, p = 6, 4, 3
    A_phys = randn(n_phys, p)
    A_sensor = randn(n_sensor, p)
    β_true = randn(p)
    y_phys = A_phys * β_true .+ 0.1 .* randn(n_phys)
    y_sensor = A_sensor * β_true .+ 0.5 .* randn(n_sensor)

    dppl = _two_channel(y_phys, y_sensor, A_phys, A_sensor)

    @testset "Default (obs_groups = nothing) preserves existing behaviour" begin
        legacy = latte_from_dppl(
            dppl; random = (:β,), force_ad_obs_model = true,
        )
        # `obs_groups = nothing` is the documented default — passing it
        # explicitly should match.
        explicit_nothing = latte_from_dppl(
            dppl; random = (:β,), force_ad_obs_model = true, obs_groups = nothing,
        )
        @test typeof(legacy.observation_model) === typeof(explicit_nothing.observation_model)
    end

    @testset "Single group covering all obs ≈ single-AD legacy path" begin
        legacy = latte_from_dppl(
            dppl; random = (:β,), force_ad_obs_model = true,
        )
        single_grp = latte_from_dppl(
            dppl; random = (:β,),
            obs_groups = [:all => (:y_phys, :y_sensor)],
        )

        x = randn(p)
        hp = (σ_phys = 0.1, σ_data = 0.5)

        # Materialize via the LGM contract: obs_model(y; θ...) — y can be
        # whatever; the wrapper substitutes the prebaked CompositeObservations.
        legacy_lik = legacy.observation_model(vcat(y_phys, y_sensor); hp...)
        composite_lik = single_grp.observation_model(nothing; hp...)

        @test loglik(x, composite_lik) ≈ loglik(x, legacy_lik) atol = 1.0e-9
        @test loggrad(x, composite_lik) ≈ loggrad(x, legacy_lik) atol = 1.0e-9
    end

    @testset "Two groups → distinct σ_phys / σ_data" begin
        model = latte_from_dppl(
            dppl; random = (:β,),
            obs_groups = [:physics => (:y_phys,), :data => (:y_sensor,)],
        )

        x = randn(p)
        # Different hp settings → different log-likelihoods.
        lik_a = model.observation_model(nothing; σ_phys = 0.1, σ_data = 5.0)
        lik_b = model.observation_model(nothing; σ_phys = 5.0, σ_data = 0.1)
        @test loglik(x, lik_a) != loglik(x, lik_b)

        # Sanity: same σ_phys/σ_data in both calls is reproducible.
        lik_same1 = model.observation_model(nothing; σ_phys = 0.7, σ_data = 0.7)
        lik_same2 = model.observation_model(nothing; σ_phys = 0.7, σ_data = 0.7)
        @test loglik(x, lik_same1) ≈ loglik(x, lik_same2)
    end

    @testset "Composite ≈ closed-form sum of per-block log-likelihoods" begin
        model = latte_from_dppl(
            dppl; random = (:β,),
            obs_groups = [:physics => (:y_phys,), :data => (:y_sensor,)],
        )

        x = randn(p)
        lik_adapter = model.observation_model(nothing; σ_phys = 0.3, σ_data = 1.2)

        ref_phys = sum(
            logpdf(Normal(dot(A_phys[i, :], x), 0.3), y_phys[i])
                for i in eachindex(y_phys)
        )
        ref_sensor = sum(
            logpdf(Normal(dot(A_sensor[i, :], x), 1.2), y_sensor[i])
                for i in eachindex(y_sensor)
        )
        @test loglik(x, lik_adapter) ≈ ref_phys + ref_sensor atol = 1.0e-9
    end

    @testset "Routes are explicit identity (not nothing)" begin
        model = latte_from_dppl(
            dppl; random = (:β,),
            obs_groups = [:physics => (:y_phys,), :data => (:y_sensor,)],
        )
        # The wrapper exposes the underlying composite for inspection.
        composite = Latte._underlying_composite(model.observation_model)
        @test composite isa CompositeObservationModel
        # Each route is a NamedTuple identity over the full hp tuple, not
        # `nothing`. Codex flagged passthrough as a weaker contract.
        for route in composite.routes
            @test route isa NamedTuple
            @test Set(keys(route)) == Set((:σ_phys, :σ_data))
            for k in keys(route)
                @test getfield(route, k) === k
            end
        end
    end

    @testset "AD differentiation through hp kwargs (Dual-eltype path)" begin
        # The accumulator's scalar type has to widen to whichever of
        # `x`-eltype or hp-kwarg-eltype carries AD partials. Differentiating
        # the loglik w.r.t. σ_phys is the cheapest stress test: it forces a
        # ForwardDiff.Dual through the hp kwarg while x stays Float64.
        import ForwardDiff
        model = latte_from_dppl(
            dppl; random = (:β,),
            obs_groups = [:physics => (:y_phys,), :data => (:y_sensor,)],
        )
        x = randn(p)
        f = θ -> begin
            lik = model.observation_model(nothing; σ_phys = θ[1], σ_data = 0.5)
            return loglik(x, lik)
        end
        # Just verify the gradient evaluates without method errors — value
        # checks are covered by the closed-form-sum test elsewhere.
        g = ForwardDiff.gradient(f, [0.3])
        @test length(g) == 1
        @test all(isfinite, g)
    end

    @testset "Validation: missing / extra / overlapping syms" begin
        # Missing: y_sensor isn't covered.
        @test_throws ArgumentError latte_from_dppl(
            dppl; random = (:β,),
            obs_groups = [:physics => (:y_phys,)],
        )
        # Extra: :nonexistent isn't an obs sym.
        @test_throws ArgumentError latte_from_dppl(
            dppl; random = (:β,),
            obs_groups = [
                :physics => (:y_phys,),
                :data => (:y_sensor, :nonexistent),
            ],
        )
        # Overlap: y_phys appears in two groups.
        @test_throws ArgumentError latte_from_dppl(
            dppl; random = (:β,),
            obs_groups = [
                :a => (:y_phys, :y_sensor),
                :b => (:y_phys,),
            ],
        )
        # Sym refers to a hyperparameter or random — not an observation.
        @test_throws ArgumentError latte_from_dppl(
            dppl; random = (:β,),
            obs_groups = [
                :physics => (:y_phys,),
                :data => (:y_sensor, :β),
            ],
        )
        # Empty group tuple.
        @test_throws ArgumentError latte_from_dppl(
            dppl; random = (:β,),
            obs_groups = [
                :empty => (),
                :all => (:y_phys, :y_sensor),
            ],
        )
        # Duplicate group name.
        @test_throws ArgumentError latte_from_dppl(
            dppl; random = (:β,),
            obs_groups = [
                :grp => (:y_phys,),
                :grp => (:y_sensor,),
            ],
        )
    end

    @testset "NamedTuple form is accepted" begin
        # Same semantics as Vector{Pair}, more Julian for static config.
        model_nt = latte_from_dppl(
            dppl; random = (:β,),
            obs_groups = (physics = (:y_phys,), data = (:y_sensor,)),
        )
        model_pairs = latte_from_dppl(
            dppl; random = (:β,),
            obs_groups = [:physics => (:y_phys,), :data => (:y_sensor,)],
        )

        x = randn(p)
        lik_nt = model_nt.observation_model(nothing; σ_phys = 0.3, σ_data = 1.2)
        lik_pairs = model_pairs.observation_model(nothing; σ_phys = 0.3, σ_data = 1.2)
        @test loglik(x, lik_nt) ≈ loglik(x, lik_pairs) atol = 1.0e-12
    end

    @testset "Heterogeneous families: Normal + Poisson groups" begin
        # The motivating Gaussian/Gaussian case is the headline, but obs_groups
        # has to handle mixed families too (e.g. continuous outcome + count
        # outcome on overlapping latents).
        @model function _mixed(y_cont, y_count, A, B)
            σ ~ Gamma(2, 1)
            β ~ MvNormal(zeros(size(A, 2)), 100.0 * I)
            for i in eachindex(y_cont)
                y_cont[i] ~ Normal(dot(A[i, :], β), σ)
            end
            for i in eachindex(y_count)
                y_count[i] ~ Poisson(exp(dot(B[i, :], β)); check_args = false)
            end
        end

        Random.seed!(2027)
        n1, n2, q = 5, 4, 3
        A = randn(n1, q)
        B = randn(n2, q) ./ 4
        β_t = randn(q) ./ 4
        y_cont = A * β_t .+ 0.2 .* randn(n1)
        y_count = [rand(Poisson(exp(dot(B[i, :], β_t)))) for i in 1:n2]

        dppl2 = _mixed(y_cont, y_count, A, B)

        model = latte_from_dppl(
            dppl2; random = (:β,),
            obs_groups = [:cont => (:y_cont,), :count => (:y_count,)],
        )

        x = randn(q)
        lik = model.observation_model(nothing; σ = 0.3)

        ref_cont = sum(
            logpdf(Normal(dot(A[i, :], x), 0.3), y_cont[i])
                for i in eachindex(y_cont)
        )
        ref_count = sum(
            logpdf(Poisson(exp(dot(B[i, :], x))), y_count[i])
                for i in eachindex(y_count)
        )
        @test loglik(x, lik) ≈ ref_cont + ref_count atol = 1.0e-9
    end

    @testset "MvNormal observation block (single ~ vector statement)" begin
        # Verifies the accumulator correctly handles vector observations.
        # A single `y ~ MvNormal(...)` fires `accumulate_observe!!` once with
        # `left::Vector` and `right::MvNormal` — `Distributions.loglikelihood`
        # returns the joint loglik, matching what we want.
        @model function _mv(y_obs, A)
            σ ~ Gamma(2, 1)
            β ~ MvNormal(zeros(size(A, 2)), 100.0 * I)
            y_obs ~ MvNormal(A * β, σ^2 * I)
        end

        Random.seed!(2028)
        n3, q = 6, 3
        A3 = randn(n3, q)
        β_t = randn(q)
        y_obs = A3 * β_t .+ 0.3 .* randn(n3)
        dppl3 = _mv(y_obs, A3)

        model = latte_from_dppl(
            dppl3; random = (:β,),
            obs_groups = [:all => (:y_obs,)],
        )
        x = randn(q)
        lik = model.observation_model(nothing; σ = 0.4)
        ref = logpdf(MvNormal(A3 * x, 0.4^2 * I), y_obs)
        @test loglik(x, lik) ≈ ref atol = 1.0e-9
    end

    @testset "logpdf-style usage with primal hp values" begin
        # End-to-end inla() through composite-obs adapters needs an IFT
        # dispatch on `gaussian_approximation(::CompositeLikelihood)` to
        # avoid nested ForwardDiff tag stacking through the outer
        # hp-gradient pass — that's tracked as upstream work. We can still
        # exercise the `gaussian_approximation` -> `loglik` -> `logpdf`
        # path with primal hp values, which is the bedrock the rest is
        # built on.
        model = latte_from_dppl(
            dppl; random = (:β,),
            obs_groups = [:physics => (:y_phys,), :data => (:y_sensor,)],
        )
        x = randn(p)
        # log_joint_density with primal hp's — exercises the same
        # observation path the inner Newton does, just without the outer
        # AD wrapper.
        θ_n = Latte.NaturalHyperparameters([0.1, 0.5], model.hyperparameter_spec)
        ll = Latte.log_joint_density(model, x, θ_n, vcat(y_phys, y_sensor))
        @test isfinite(ll)
    end
end
