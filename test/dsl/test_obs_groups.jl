using Test
using Latte
using DynamicPPL: @model
import DynamicPPL
using GaussianMarkovRandomFields:
    AutoDiffObservationModel, CompositeObservationModel, CompositeLikelihood,
    CompositeObservations, LinearlyTransformedObservationModel, ParameterizedOffset,
    ParameterizedMatrix, loglik, loggrad, loghessian
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

    @testset "Routes are NamedTuples with Symbol values (rename-only)" begin
        # AD-forced groups use the full identity route over hp_names; per-group
        # fast-path components use a smaller route covering only the family's
        # nuisance kwargs. Both are rename-only NamedTuples (no value transforms).
        model = latte_from_dppl(
            dppl; random = (:β,), force_ad_obs_model = true,
            obs_groups = [:physics => (:y_phys,), :data => (:y_sensor,)],
        )
        composite = Latte._underlying_composite(model.observation_model)
        @test composite isa CompositeObservationModel
        for route in composite.routes
            @test route isa NamedTuple
            for v in values(route)
                @test v isa Symbol
                @test v in (:σ_phys, :σ_data)
            end
        end
        # AD-forced path: each component declares the full hp tuple.
        for route in composite.routes
            @test Set(keys(route)) == Set((:σ_phys, :σ_data))
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

    @testset "Per-group fast-path detection" begin
        using GaussianMarkovRandomFields: LinearlyTransformedObservationModel,
            ExponentialFamily

        @testset "All-Normal groups → all fast components, none AD" begin
            model = latte_from_dppl(
                dppl; random = (:β,),
                obs_groups = [:physics => (:y_phys,), :data => (:y_sensor,)],
            )
            composite = Latte._underlying_composite(model.observation_model)
            @test length(composite.components) == 2
            for comp in composite.components
                @test !(comp isa AutoDiffObservationModel)
                @test comp isa LinearlyTransformedObservationModel
            end
        end

        @testset "Per-component routes are scoped to family kwargs" begin
            model = latte_from_dppl(
                dppl; random = (:β,),
                obs_groups = [:physics => (:y_phys,), :data => (:y_sensor,)],
            )
            composite = Latte._underlying_composite(model.observation_model)
            # Each fast Normal component routes only `:σ` from the outer hp set.
            r1, r2 = composite.routes
            @test keys(r1) == (:σ,) && r1.σ === :σ_phys
            @test keys(r2) == (:σ,) && r2.σ === :σ_data
        end

        @testset "Fast vs forced-AD composite: identical loglik & gradient" begin
            ad = latte_from_dppl(
                dppl; random = (:β,), force_ad_obs_model = true,
                obs_groups = [:physics => (:y_phys,), :data => (:y_sensor,)],
            )
            fast = latte_from_dppl(
                dppl; random = (:β,),
                obs_groups = [:physics => (:y_phys,), :data => (:y_sensor,)],
            )
            x = randn(p)
            hp = (σ_phys = 0.3, σ_data = 1.2)
            ad_lik = ad.observation_model(nothing; hp...)
            fast_lik = fast.observation_model(nothing; hp...)
            @test loglik(x, ad_lik) ≈ loglik(x, fast_lik) atol = 1.0e-9
            @test loggrad(x, ad_lik) ≈ loggrad(x, fast_lik) atol = 1.0e-9
        end

        @testset "Mixed Normal + Poisson families: both groups fast" begin
            @model function _mixed_fams(y_cont, y_count, A, B)
                σ ~ Gamma(2, 1)
                β ~ MvNormal(zeros(size(A, 2)), 100.0 * I)
                for i in eachindex(y_cont)
                    y_cont[i] ~ Normal(dot(A[i, :], β), σ)
                end
                for i in eachindex(y_count)
                    y_count[i] ~ Poisson(exp(dot(B[i, :], β)); check_args = false)
                end
            end

            Random.seed!(2029)
            n1, n2, q = 5, 4, 3
            A = randn(n1, q)
            B = randn(n2, q) ./ 4
            β_t = randn(q) ./ 4
            y_cont = A * β_t .+ 0.2 .* randn(n1)
            y_count = [rand(Poisson(exp(dot(B[i, :], β_t)))) for i in 1:n2]

            dppl_m = _mixed_fams(y_cont, y_count, A, B)
            model = latte_from_dppl(
                dppl_m; random = (:β,),
                obs_groups = [:cont => (:y_cont,), :count => (:y_count,)],
            )
            composite = Latte._underlying_composite(model.observation_model)
            for comp in composite.components
                @test !(comp isa AutoDiffObservationModel)
                @test comp isa LinearlyTransformedObservationModel
            end
            # Poisson route is empty (no nuisance kwargs); Normal routes σ.
            r_cont, r_count = composite.routes
            @test keys(r_cont) == (:σ,) && r_cont.σ === :σ
            @test keys(r_count) == ()
        end

        @testset "Heterogeneous group → AD fallback for that group only" begin
            # One group mixes Normal + Poisson; that group must fall back to AD,
            # the homogeneous group can still go fast.
            @model function _hetero_grp(y_n, y_p, y_g, A, B, C)
                σ ~ Gamma(2, 1)
                β ~ MvNormal(zeros(size(A, 2)), 100.0 * I)
                for i in eachindex(y_n)
                    y_n[i] ~ Normal(dot(A[i, :], β), σ)
                end
                for i in eachindex(y_p)
                    y_p[i] ~ Poisson(exp(dot(B[i, :], β)); check_args = false)
                end
                # Third channel — pure Normal, lives alone.
                for i in eachindex(y_g)
                    y_g[i] ~ Normal(dot(C[i, :], β), σ)
                end
            end

            Random.seed!(2030)
            n1, n2, n3, q = 4, 3, 5, 3
            A = randn(n1, q); B = randn(n2, q) ./ 4; C = randn(n3, q)
            β_t = randn(q) ./ 4
            y_n = A * β_t .+ 0.2 .* randn(n1)
            y_p = [rand(Poisson(exp(dot(B[i, :], β_t)))) for i in 1:n2]
            y_g = C * β_t .+ 0.2 .* randn(n3)

            dppl_h = _hetero_grp(y_n, y_p, y_g, A, B, C)
            model = latte_from_dppl(
                dppl_h; random = (:β,),
                obs_groups = [:mixed => (:y_n, :y_p), :pure => (:y_g,)],
            )
            composite = Latte._underlying_composite(model.observation_model)
            # First group (mixed Normal+Poisson) → AD; second (pure Normal) → fast.
            @test composite.components[1] isa AutoDiffObservationModel
            @test composite.components[2] isa LinearlyTransformedObservationModel
        end

        @testset "Hardcoded nuisance kwarg → fast component (no AD)" begin
            # `y_phys[i] ~ Normal(μ, 1e-3)` literal σ — should still fast-path,
            # baking the constant into the component instead of forcing AD.
            @model function _hard_sigma(y_phys, y_data, A_phys, A_data)
                σ_data ~ Gamma(2, 1)
                β ~ MvNormal(zeros(size(A_phys, 2)), 100.0 * I)
                for i in eachindex(y_phys)
                    y_phys[i] ~ Normal(dot(A_phys[i, :], β), 1.0e-3)
                end
                for i in eachindex(y_data)
                    y_data[i] ~ Normal(dot(A_data[i, :], β), σ_data)
                end
            end

            Random.seed!(2032)
            n1, n2, q = 6, 5, 3
            A_p = randn(n1, q); A_d = randn(n2, q)
            β_t = randn(q)
            y_p = A_p * β_t .+ 1.0e-3 .* randn(n1)
            y_d = A_d * β_t .+ 0.4 .* randn(n2)

            dppl_h = _hard_sigma(y_p, y_d, A_p, A_d)
            model = latte_from_dppl(
                dppl_h; random = (:β,),
                obs_groups = [:phys => (:y_phys,), :data => (:y_data,)],
            )
            composite = Latte._underlying_composite(model.observation_model)
            for comp in composite.components
                @test !(comp isa AutoDiffObservationModel)
            end
            # The :phys group has no hp-driven kwargs, so its route is empty.
            r_phys, r_data = composite.routes
            @test keys(r_phys) == ()
            @test keys(r_data) == (:σ,) && r_data.σ === :σ_data
            # Materialise and compare to the closed-form sum: σ_phys is fixed
            # at 1e-3 inside the component, σ_data flows from the kwarg.
            x = randn(q)
            lik = model.observation_model(nothing; σ_data = 0.4)
            ref_p = sum(
                logpdf(Normal(dot(A_p[i, :], x), 1.0e-3), y_p[i])
                    for i in eachindex(y_p)
            )
            ref_d = sum(
                logpdf(Normal(dot(A_d[i, :], x), 0.4), y_d[i])
                    for i in eachindex(y_d)
            )
            @test loglik(x, lik) ≈ ref_p + ref_d rtol = 1.0e-12
        end

        @testset "Single-group hardcoded σ → fast (no obs-side hps)" begin
            # τ drives the β prior precision so we have an outer hp for
            # INLA, but the obs-side σ is hardcoded — the obs route should
            # be empty and the component should still be fast.
            @model function _all_fixed(y, A)
                τ ~ Gamma(2, 1)
                β ~ MvNormal(zeros(size(A, 2)), (1 / τ) * I(size(A, 2)))
                for i in eachindex(y)
                    y[i] ~ Normal(dot(A[i, :], β), 0.5)
                end
            end
            Random.seed!(2033)
            n, q = 7, 3
            A = randn(n, q); β_t = randn(q)
            y = A * β_t .+ 0.5 .* randn(n)
            dppl_a = _all_fixed(y, A)
            model = latte_from_dppl(
                dppl_a; random = (:β,),
                obs_groups = [:all => (:y,)],
            )
            composite = Latte._underlying_composite(model.observation_model)
            @test !(composite.components[1] isa AutoDiffObservationModel)
            @test composite.routes[1] == NamedTuple()
            x = randn(q)
            lik = model.observation_model(nothing; τ = 0.01)
            ref = sum(
                logpdf(Normal(dot(A[i, :], x), 0.5), y[i])
                    for i in eachindex(y)
            )
            @test loglik(x, lik) ≈ ref atol = 1.0e-9
        end

        @testset "hp-dependent design matrix → LTM ParameterizedMatrix (composite path)" begin
            # PDE-inverse shape: η = -κ·L·u + u — linear in u, but the design
            # matrix entries depend on κ (its sparsity pattern does not). On the
            # composite path this is captured by the LTM's ParameterizedMatrix,
            # which threads the κ-Dual through the per-component IFT — so the κ
            # posterior is informed, not collapsed to its prior.
            @model function _hp_in_A(y, L)
                κ ~ Gamma(2, 0.1)
                u ~ MvNormal(zeros(size(L, 1)), 100.0 * I(size(L, 1)))
                for i in eachindex(y)
                    y[i] ~ Normal(-κ * dot(L[i, :], u) + u[i], 0.01)
                end
            end
            Random.seed!(2035)
            n = 6
            L = randn(n, n)
            u_t = randn(n)
            y = (-0.1 .* (L * u_t) .+ u_t) .+ 0.01 .* randn(n)
            dppl_hpA = _hp_in_A(y, L)
            model = latte_from_dppl(
                dppl_hpA; random = (:u,),
                obs_groups = [:all => (:y,)],
            )
            composite = Latte._underlying_composite(model.observation_model)
            @test composite.components[1] isa LinearlyTransformedObservationModel
            @test composite.components[1].design_matrix isa ParameterizedMatrix
        end

        @testset "hp-dependent offset b(θ) → LTM offset (fast path, not AD)" begin
            # η = u + ω where ω is an outer hp shift. A is constant (= I) and
            # only the offset b(θ) = ω depends on θ. Unlike a θ-dependent A,
            # this is captured by the LTM's ParameterizedOffset, so the
            # component stays on the fast path — its θ-offset gradient is
            # forward-mode-exact through the composite IFT.
            @model function _hp_in_b(y)
                ω ~ Gamma(2, 1)
                u ~ MvNormal(zeros(length(y)), 100.0 * I(length(y)))
                for i in eachindex(y)
                    y[i] ~ Normal(u[i] + ω, 0.1)
                end
            end
            Random.seed!(2036)
            y = randn(5)
            dppl_b = _hp_in_b(y)
            model = latte_from_dppl(
                dppl_b; random = (:u,),
                obs_groups = [:all => (:y,)],
            )
            composite = Latte._underlying_composite(model.observation_model)
            @test composite.components[1] isa LinearlyTransformedObservationModel
            @test composite.components[1].offset isa ParameterizedOffset
        end

        @testset "Latent-dependent σ → AD fallback (not misclassified as fixed)" begin
            # σ = exp(α) where α is a latent — at probe x=0, σ=1, so
            # under hp-only perturbation the fast-path detector would
            # call it a constant. The latent-invariance check must catch
            # the dependence and reject the group.
            @model function _latent_sigma(y, A)
                τ ~ Gamma(2, 1)
                α ~ Normal(0, 1)
                β ~ MvNormal(zeros(size(A, 2)), (1 / τ) * I(size(A, 2)))
                for i in eachindex(y)
                    y[i] ~ Normal(dot(A[i, :], β), exp(α))
                end
            end
            Random.seed!(2034)
            n, q = 5, 3
            A = randn(n, q); β_t = randn(q)
            y = A * β_t .+ 0.3 .* randn(n)
            dppl_l = _latent_sigma(y, A)
            model = latte_from_dppl(
                dppl_l; random = (:α, :β),
                obs_groups = [:all => (:y,)],
            )
            composite = Latte._underlying_composite(model.observation_model)
            @test composite.components[1] isa AutoDiffObservationModel
        end

        @testset "Emission-order y consistency under interleaved sites" begin
            # Model with interleaved y_a/y_b sites — A rows for the fast-path
            # component must align with the y values in DPPL emission order,
            # not with `_collect_group_y` concatenation order.
            @model function _interleaved(y_a, y_b, A, B)
                σ ~ Gamma(2, 1)
                β ~ MvNormal(zeros(size(A, 2)), 100.0 * I)
                # Interleave: a, b, a, b, ...
                n = min(length(y_a), length(y_b))
                for i in 1:n
                    y_a[i] ~ Normal(dot(A[i, :], β), σ)
                    y_b[i] ~ Normal(dot(B[i, :], β), σ)
                end
            end

            Random.seed!(2031)
            n, q = 5, 3
            A = randn(n, q); B = randn(n, q)
            β_t = randn(q)
            y_a = A * β_t .+ 0.1 .* randn(n)
            y_b = B * β_t .+ 0.1 .* randn(n)

            dppl_i = _interleaved(y_a, y_b, A, B)
            ad = latte_from_dppl(
                dppl_i; random = (:β,), force_ad_obs_model = true,
                obs_groups = [:a => (:y_a,), :b => (:y_b,)],
            )
            fast = latte_from_dppl(
                dppl_i; random = (:β,),
                obs_groups = [:a => (:y_a,), :b => (:y_b,)],
            )
            x = randn(q)
            hp = (σ = 0.3,)
            ll_ad = loglik(x, ad.observation_model(nothing; hp...))
            ll_fast = loglik(x, fast.observation_model(nothing; hp...))
            @test ll_ad ≈ ll_fast atol = 1.0e-9
        end
    end
end

# Two constant-noise linear-Gaussian channels with *different* fixed σ share
# family and hp deps, so grouping by (family, deps) alone would merge them into
# one heteroskedastic group — which the rename-only fast-path route can't
# represent, needlessly dropping the group to AutoDiff. The grouping key also
# carries the σ expression, so the channels stay separate LTM components.
@testset "constant-σ Normal channels with different σ stay separate LTM groups" begin
    n, p = 6, 2
    Random.seed!(21)
    A = randn(n, p)
    β0 = [0.5, -0.3]
    y_a = A * β0 .+ 0.01 .* randn(n)   # tight channel
    y_b = A * β0 .+ 0.1 .* randn(n)   # loose channel
    y_c = A * β0 .+ 0.05 .* randn(n)   # hp-noise channel (forces the composite path)
    y = vcat(y_a, y_b, y_c)

    @latte function het_split(y_a, y_b, y_c, A, p)
        σ_c ~ truncated(Normal(0.1, 0.05); lower = 0.01)
        β ~ MvNormal(zeros(p), 100.0 * I(p))
        for i in eachindex(y_a)
            y_a[i] ~ Normal(dot(A[i, :], β), 0.01)
        end
        for i in eachindex(y_b)
            y_b[i] ~ Normal(dot(A[i, :], β), 0.1)
        end
        for i in eachindex(y_c)
            y_c[i] ~ Normal(dot(A[i, :], β), σ_c)
        end
    end

    lgm = het_split(y_a, y_b, y_c, A, p)
    comp = Latte._underlying_composite(lgm.observation_model)
    @test comp !== nothing
    @test length(comp.components) == 3
    s = string(typeof(lgm.observation_model))
    @test !occursin("AutoDiff", s)
    @test !occursin("NonlinearLeastSquares", s)
    @test occursin("LinearlyTransformed", s)

    # Exactness: every component is a plain linear-Gaussian likelihood, so the
    # posterior matches the exact single-AD reference.
    res = inla(het_split(y_a, y_b, y_c, A, p), y; latent_marginalization_method = GaussianMarginal(), progress = false)
    dppl = Latte.dppl_model(het_split)(y_a, y_b, y_c, A, p)
    lgm_ad = Latte.latte_from_dppl(dppl; random = :β, force_ad_obs_model = true)
    @test occursin("AutoDiff", string(typeof(lgm_ad.observation_model)))
    res_ad = inla(lgm_ad, y; latent_marginalization_method = GaussianMarginal(), progress = false)
    lm, lm_ad = latent_marginals(res), latent_marginals(res_ad)
    @test maximum(abs, mean.(lm) .- mean.(lm_ad)) < 1.0e-3
    @test maximum(abs, std.(lm) .- std.(lm_ad)) < 1.0e-3
    @test isapprox(
        mean(res.hyperparameter_marginals[:σ_c]),
        mean(res_ad.hyperparameter_marginals[:σ_c]); atol = 0.02,
    )
end

# Homoskedastic control: matching σ expressions still merge into one group —
# the σ-key split only separates channels whose noise genuinely differs.
@testset "constant-σ Normal channels with matching σ still merge into one group" begin
    n, p = 6, 2
    Random.seed!(22)
    A = randn(n, p)
    β0 = [0.5, -0.3]
    y_a = A * β0 .+ 0.05 .* randn(n)
    y_b = A * β0 .+ 0.05 .* randn(n)
    y_c = A * β0 .+ 0.05 .* randn(n)

    @latte function hom_merge_g(y_a, y_b, y_c, A, p)
        σ_c ~ truncated(Normal(0.1, 0.05); lower = 0.01)
        β ~ MvNormal(zeros(p), 100.0 * I(p))
        for i in eachindex(y_a)
            y_a[i] ~ Normal(dot(A[i, :], β), 0.05)
        end
        for i in eachindex(y_b)
            y_b[i] ~ Normal(dot(A[i, :], β), 0.05)
        end
        for i in eachindex(y_c)
            y_c[i] ~ Normal(dot(A[i, :], β), σ_c)
        end
    end

    lgm = hom_merge_g(y_a, y_b, y_c, A, p)
    comp = Latte._underlying_composite(lgm.observation_model)
    @test comp !== nothing
    @test length(comp.components) == 2   # merged constant-σ pair + hp channel
    s = string(typeof(lgm.observation_model))
    @test !occursin("AutoDiff", s)
    @test occursin("LinearlyTransformed", s)
end

# A pair of constant-σ channels with nothing else in the model stays on the
# single-obs path (today's AD fallback there is gradient-capable); the σ-key
# split only refines models that are already composite.
@testset "σ-key split does not flip a single-group model to composite" begin
    n = 5
    Random.seed!(23)
    ya, yb = 0.2 .* randn(n), 0.2 .* randn(n)
    @latte function two_chan_only(y_a, y_b, n)
        τ ~ truncated(Normal(1.0, 0.5); lower = 0.1)
        x ~ IIDModel(n)(τ = τ)
        for i in eachindex(y_a)
            y_a[i] ~ Normal(x[i], 0.01)
        end
        for i in eachindex(y_b)
            y_b[i] ~ Normal(x[i], 0.1)
        end
    end
    lgm = two_chan_only(ya, yb, n)
    @test Latte._underlying_composite(lgm.observation_model) === nothing
    @test occursin("AutoDiff", string(typeof(lgm.observation_model)))
end
