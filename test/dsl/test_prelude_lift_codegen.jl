using Test
using Latte
using Latte: _analyze_lift_eligibility, _generate_lift_callables, LiftPlan
using Distributions

# Codegen tests: generate prelude/obs/pointwise bodies from a LiftPlan,
# eval them into a sandbox module, and exercise the resulting functions.

@testset "Prelude-lift codegen" begin

    # Sandbox module with `Distributions` in scope (mirrors `using Latte`
    # in a user module — Latte's emitted code references `Distributions.*`
    # so the user module must have it accessible).
    @eval module _LiftCodegenSandbox
    using Distributions
    end

    function _eval_into_sandbox(defs)
        Base.eval(_LiftCodegenSandbox, defs.prelude_def)
        Base.eval(_LiftCodegenSandbox, defs.obs_body_def)
        Base.eval(_LiftCodegenSandbox, defs.pointwise_def)
        return (
            prelude = getfield(_LiftCodegenSandbox, defs.prelude_fname),
            obs_body = getfield(_LiftCodegenSandbox, defs.obs_body_fname),
            pointwise = getfield(_LiftCodegenSandbox, defs.pointwise_fname),
        )
    end

    @testset "Tiny linear-Normal model: prelude returns expected captures" begin
        body = quote
            σ ~ Gamma(2, 1)
            scale_factor = 2.0
            offset = 0.5
            @random β ~ MvNormal(zeros(3), 1.0)
            for i in eachindex(y)
                y[i] ~ Normal(scale_factor * β[i] + offset, σ)
            end
        end
        plan = _analyze_lift_eligibility(body, (:y,))
        @test plan isa LiftPlan
        @test Set(plan.capture) == Set([:scale_factor, :offset])

        defs = _generate_lift_callables(plan, :tiny_model, (:y,))
        fns = _eval_into_sandbox(defs)

        args_nt = (; y = [0.1, 0.2, 0.3])
        hp_nt = (; σ = 1.0)
        prelude_state = fns.prelude(args_nt, hp_nt)
        @test prelude_state isa NamedTuple
        @test prelude_state.scale_factor == 2.0
        @test prelude_state.offset == 0.5
    end

    @testset "Tiny linear-Normal model: obs body matches analytic logpdf" begin
        body = quote
            σ ~ Gamma(2, 1)
            scale_factor = 2.0
            offset = 0.5
            @random β ~ MvNormal(zeros(3), 1.0)
            for i in eachindex(y)
                y[i] ~ Normal(scale_factor * β[i] + offset, σ)
            end
        end
        plan = _analyze_lift_eligibility(body, (:y,))
        defs = _generate_lift_callables(plan, :tiny_model_v2, (:y,))
        fns = _eval_into_sandbox(defs)

        y_data = [0.7, -0.3, 1.2]
        β = [0.1, 0.2, 0.3]
        σ_val = 1.5

        args_nt = (; y = y_data)
        hp_nt = (; σ = σ_val)
        prelude_state = fns.prelude(args_nt, hp_nt)
        # Payload carries args, prelude state, group_syms, offsets, is_scalar.
        offsets = (; β = 1:3)
        is_scalar = (; β = false)
        payload = (
            args = args_nt, prelude_state = prelude_state,
            group_syms = (:y,), offsets = offsets, is_scalar = is_scalar,
        )
        flat_x = β
        logp = fns.obs_body(flat_x; y = payload, σ = σ_val)

        expected = sum(
            logpdf(Normal(2.0 * β[i] + 0.5, σ_val), y_data[i])
                for i in eachindex(y_data)
        )
        @test logp ≈ expected
    end

    @testset "Pointwise body returns one contribution per syntactic obs site" begin
        body = quote
            σ ~ Gamma(2, 1)
            offset = 0.5
            @random β ~ MvNormal(zeros(2), 1.0)
            for i in eachindex(y)
                y[i] ~ Normal(β[i] + offset, σ)
            end
        end
        plan = _analyze_lift_eligibility(body, (:y,))
        defs = _generate_lift_callables(plan, :tiny_pw, (:y,))
        fns = _eval_into_sandbox(defs)

        y_data = [0.5, -0.5]
        β = [0.1, 0.2]
        σ_val = 1.0

        args_nt = (; y = y_data)
        hp_nt = (; σ = σ_val)
        prelude_state = fns.prelude(args_nt, hp_nt)
        offsets = (; β = 1:2)
        is_scalar = (; β = false)
        payload = (
            args = args_nt, prelude_state = prelude_state,
            group_syms = (:y,), offsets = offsets, is_scalar = is_scalar,
        )
        contribs = fns.pointwise(β; y = payload, σ = σ_val)
        @test length(contribs) == length(y_data)
        for i in eachindex(y_data)
            @test contribs[i] ≈ logpdf(Normal(β[i] + 0.5, σ_val), y_data[i])
        end
    end

    @testset "Multi-component obs body filters by group_syms" begin
        body = quote
            σ_a ~ Gamma(2, 1)
            σ_b ~ Gamma(2, 1)
            @random β ~ MvNormal(zeros(2), 1.0)
            for i in eachindex(y_a)
                y_a[i] ~ Normal(β[i], σ_a)
            end
            for i in eachindex(y_b)
                y_b[i] ~ Normal(β[i], σ_b)
            end
        end
        plan = _analyze_lift_eligibility(body, (:y_a, :y_b))
        defs = _generate_lift_callables(plan, :two_channel, (:y_a, :y_b))
        fns = _eval_into_sandbox(defs)

        y_a = [0.1, 0.2]
        y_b = [-0.1, -0.2]
        β = [0.5, -0.5]
        σ_a, σ_b = 1.0, 2.0

        args_nt = (; y_a = y_a, y_b = y_b)
        hp_nt = (; σ_a = σ_a, σ_b = σ_b)
        prelude_state = fns.prelude(args_nt, hp_nt)
        offsets = (; β = 1:2)
        is_scalar = (; β = false)

        # Only group :y_a
        payload_a = (
            args = args_nt, prelude_state = prelude_state,
            group_syms = (:y_a,), offsets = offsets, is_scalar = is_scalar,
        )
        logp_a = fns.obs_body(β; y = payload_a, σ_a = σ_a, σ_b = σ_b)
        expected_a = sum(logpdf(Normal(β[i], σ_a), y_a[i]) for i in eachindex(y_a))
        @test logp_a ≈ expected_a

        # Only group :y_b
        payload_b = (
            args = args_nt, prelude_state = prelude_state,
            group_syms = (:y_b,), offsets = offsets, is_scalar = is_scalar,
        )
        logp_b = fns.obs_body(β; y = payload_b, σ_a = σ_a, σ_b = σ_b)
        expected_b = sum(logpdf(Normal(β[i], σ_b), y_b[i]) for i in eachindex(y_b))
        @test logp_b ≈ expected_b

        # Both groups
        payload_both = (
            args = args_nt, prelude_state = prelude_state,
            group_syms = (:y_a, :y_b), offsets = offsets, is_scalar = is_scalar,
        )
        logp_both = fns.obs_body(β; y = payload_both, σ_a = σ_a, σ_b = σ_b)
        @test logp_both ≈ expected_a + expected_b
    end

    @testset "Capture aliasing: post-body reads prelude var via post-local alias" begin
        # `g_state` is in prelude. Post-body has `local_alias = g_state.x`.
        # Capture must include `g_state` because `local_alias` references it.
        body = quote
            σ ~ Gamma(2, 1)
            g_state = (x = [10.0, 20.0],)
            @random β ~ MvNormal(zeros(2), 1.0)
            local_alias = g_state.x
            for i in eachindex(y)
                y[i] ~ Normal(β[i] + local_alias[i], σ)
            end
        end
        plan = _analyze_lift_eligibility(body, (:y,))
        @test plan isa LiftPlan
        @test :g_state in plan.capture

        defs = _generate_lift_callables(plan, :alias_check, (:y,))
        fns = _eval_into_sandbox(defs)

        y_data = [1.0, 2.0]
        β = [0.0, 0.0]
        σ_val = 1.0
        args_nt = (; y = y_data)
        hp_nt = (; σ = σ_val)
        prelude_state = fns.prelude(args_nt, hp_nt)
        @test prelude_state.g_state.x == [10.0, 20.0]

        payload = (
            args = args_nt, prelude_state = prelude_state,
            group_syms = (:y,), offsets = (; β = 1:2),
            is_scalar = (; β = false),
        )
        logp = fns.obs_body(β; y = payload, σ = σ_val)
        expected = sum(
            logpdf(Normal(β[i] + [10.0, 20.0][i], σ_val), y_data[i])
                for i in eachindex(y_data)
        )
        @test logp ≈ expected
    end

    @testset "Scalar random effect: __flat_x is sliced via first(__offsets.<rsym>)" begin
        body = quote
            σ ~ Gamma(2, 1)
            @random α ~ Normal(0.0, 1.0)
            for i in eachindex(y)
                y[i] ~ Normal(α, σ)
            end
        end
        plan = _analyze_lift_eligibility(body, (:y,))
        @test plan isa LiftPlan
        @test plan.random_syms == [:α]

        defs = _generate_lift_callables(plan, :scalar_rand, (:y,))
        fns = _eval_into_sandbox(defs)

        y_data = [0.3, -0.2, 0.5]
        α_val = 0.7
        σ_val = 1.2
        flat_x = [α_val]    # length-1 storage for the scalar
        args_nt = (; y = y_data)
        hp_nt = (; σ = σ_val)
        prelude_state = fns.prelude(args_nt, hp_nt)
        payload = (
            args = args_nt, prelude_state = prelude_state,
            group_syms = (:y,),
            offsets = (; α = 1:1),
            is_scalar = (; α = true),
        )
        logp = fns.obs_body(flat_x; y = payload, σ = σ_val)
        expected = sum(logpdf(Normal(α_val, σ_val), y_data[i]) for i in eachindex(y_data))
        @test logp ≈ expected
    end

    @testset "AD compatibility: obs body propagates ForwardDiff Duals" begin
        using ForwardDiff: Dual, value
        body = quote
            σ ~ Gamma(2, 1)
            @random β ~ MvNormal(zeros(2), 1.0)
            for i in eachindex(y)
                y[i] ~ Normal(β[i], σ)
            end
        end
        plan = _analyze_lift_eligibility(body, (:y,))
        defs = _generate_lift_callables(plan, :ad_check, (:y,))
        fns = _eval_into_sandbox(defs)

        y_data = [0.4, -0.3]
        σ_val = 1.0
        β_dual = [Dual(0.1, 1.0, 0.0), Dual(0.2, 0.0, 1.0)]
        args_nt = (; y = y_data)
        hp_nt = (; σ = σ_val)
        prelude_state = fns.prelude(args_nt, hp_nt)
        payload = (
            args = args_nt, prelude_state = prelude_state,
            group_syms = (:y,), offsets = (; β = 1:2),
            is_scalar = (; β = false),
        )
        logp_dual = fns.obs_body(β_dual; y = payload, σ = σ_val)
        @test logp_dual isa Dual
        expected_value = sum(
            logpdf(Normal(value(β_dual[i]), σ_val), y_data[i])
                for i in eachindex(y_data)
        )
        @test value(logp_dual) ≈ expected_value
    end
end
