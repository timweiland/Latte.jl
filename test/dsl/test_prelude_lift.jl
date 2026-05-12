using Test
using Latte
using Latte: _analyze_lift_eligibility, LiftPlan

# Eligibility analyzer for the `@latte` prelude-lift optimization.
# Tests the AST-level analysis only — codegen and end-to-end equivalence
# live in test_prelude_lift_codegen.jl.

@testset "Prelude-lift eligibility analyzer" begin

    @testset "accepts canonical PDE-style body" begin
        body = quote
            κ ~ Gamma(2.0, 0.1)
            σ_data ~ truncated(Normal(0.02, 0.005); lower = 1.0e-3)
            ℓ_x ~ truncated(Normal(0.25, 0.1); lower = 0.05)
            k = HalfIntegerMaternKernel(3, [ℓ_x])
            f_gp = GP(k)
            fg = FunctionalGaussian(f_gp; u = δ(X_c))
            g = vecchia(fg)
            @random x ~ g
            blocks = nameview(g, x)
            for i in eachindex(y_phys)
                y_phys[i] ~ Normal(blocks.u[i], σ_data)
            end
        end
        plan = _analyze_lift_eligibility(body, (:y_phys, :X_c))
        @test plan isa LiftPlan
        @test :g in plan.capture
        @test :κ in plan.hp_syms
        @test :σ_data in plan.hp_syms
        @test :ℓ_x in plan.hp_syms
        @test :x in plan.random_syms
    end

    @testset "rejects: body has no @random site" begin
        body = quote
            σ ~ Gamma(2, 1)
            for i in eachindex(y)
                y[i] ~ Normal(0.0, σ)
            end
        end
        @test _analyze_lift_eligibility(body, (:y,)) === nothing
    end

    @testset "rejects: first @random is inside a for loop" begin
        body = quote
            σ ~ Gamma(2, 1)
            for i in 1:3
                @random x[i] ~ Normal(0, σ)
            end
            for i in eachindex(y)
                y[i] ~ Normal(0.0, σ)
            end
        end
        @test _analyze_lift_eligibility(body, (:y,)) === nothing
    end

    @testset "rejects: first @random is inside a let block" begin
        body = quote
            σ ~ Gamma(2, 1)
            let
                @random x ~ Normal(0.0, σ)
            end
            for i in eachindex(y)
                y[i] ~ Normal(0.0, σ)
            end
        end
        @test _analyze_lift_eligibility(body, (:y,)) === nothing
    end

    @testset "rejects: first @random is inside an if branch" begin
        body = quote
            σ ~ Gamma(2, 1)
            if true
                @random x ~ Normal(0.0, σ)
            end
            for i in eachindex(y)
                y[i] ~ Normal(0.0, σ)
            end
        end
        @test _analyze_lift_eligibility(body, (:y,)) === nothing
    end

    @testset "rejects: observation site before first @random" begin
        body = quote
            σ ~ Gamma(2, 1)
            y_extra[1] ~ Normal(0.0, σ)
            @random x ~ Normal(0.0, σ)
            for i in eachindex(y)
                y[i] ~ Normal(x, σ)
            end
        end
        @test _analyze_lift_eligibility(body, (:y, :y_extra)) === nothing
    end

    @testset "rejects: hp `~` after first @random" begin
        body = quote
            σ ~ Gamma(2, 1)
            @random x ~ Normal(0.0, σ)
            τ ~ Gamma(2, 1)
            for i in eachindex(y)
                y[i] ~ Normal(x, τ)
            end
        end
        @test _analyze_lift_eligibility(body, (:y,)) === nothing
    end

    @testset "rejects: hp `~` inside a for loop in prelude" begin
        body = quote
            κ ~ Gamma(2, 1)
            for j in 1:3
                ℓ_j ~ truncated(Normal(0.25, 0.1); lower = 0.05)
            end
            @random x ~ Normal(0.0, 1.0)
            for i in eachindex(y)
                y[i] ~ Normal(x, 1.0)
            end
        end
        @test _analyze_lift_eligibility(body, (:y,)) === nothing
    end

    @testset "rejects: body is wrapped in a single top-level let" begin
        body = quote
            let
                σ ~ Gamma(2, 1)
                @random x ~ Normal(0.0, σ)
                for i in eachindex(y)
                    y[i] ~ Normal(x, σ)
                end
            end
        end
        @test _analyze_lift_eligibility(body, (:y,)) === nothing
    end

    @testset "tight capture: only vars actually read post-random are captured" begin
        body = quote
            σ ~ Gamma(2, 1)
            unused_intermediate = 42.0
            used_intermediate = 17.0
            @random x ~ Normal(0.0, σ)
            for i in eachindex(y)
                y[i] ~ Normal(x + used_intermediate, σ)
            end
        end
        plan = _analyze_lift_eligibility(body, (:y,))
        @test plan isa LiftPlan
        @test :used_intermediate in plan.capture
        @test :unused_intermediate ∉ plan.capture
    end

    @testset "post-body sees captured prelude state via alias chains" begin
        # `blocks = nameview(g, x)` — `blocks` is post-local. The body
        # still reads `g`, which is in prelude. Tight capture must include
        # `g` because `blocks`'s RHS (post-local) references it.
        body = quote
            σ ~ Gamma(2, 1)
            g = build_gp()
            @random x ~ g
            blocks = nameview(g, x)
            for i in eachindex(y)
                y[i] ~ Normal(blocks.u[i], σ)
            end
        end
        plan = _analyze_lift_eligibility(body, (:y,))
        @test plan isa LiftPlan
        @test :g in plan.capture
    end

    @testset "multiple @random sites accepted if all top-level after prelude" begin
        body = quote
            σ ~ Gamma(2, 1)
            g1 = build()
            g2 = build()
            @random u ~ g1
            @random v ~ g2
            for i in eachindex(y)
                y[i] ~ Normal(u + v, σ)
            end
        end
        plan = _analyze_lift_eligibility(body, (:y,))
        @test plan isa LiftPlan
        @test Set(plan.random_syms) == Set([:u, :v])
        @test :g1 in plan.capture
        @test :g2 in plan.capture
    end

    @testset "lift is disabled when the @latte signature has kwargs" begin
        # An `@latte function foo(y; opt=1)` body might pass `opt` through
        # in the prelude. The current codegen doesn't thread kwargs into the
        # prelude function, so we reject lift in the macro and fall back to
        # the standard DPPL path. This testset covers that the analyzer
        # itself doesn't crash on such bodies (it shouldn't be called for
        # them — the macro guards earlier — but defensive coverage).
        body = quote
            σ ~ Gamma(2, 1)
            opt_local = 1
            @random β ~ MvNormal(zeros(2), 1.0)
            for i in eachindex(y)
                y[i] ~ Normal(β[i] * opt_local, σ)
            end
        end
        plan = _analyze_lift_eligibility(body, (:y,))
        @test plan isa LiftPlan
        @test :opt_local in plan.capture
    end

    @testset "hp_syms only contains symbols actually present as priors" begin
        body = quote
            σ ~ Gamma(2, 1)
            g = build()
            @random x ~ g
            for i in eachindex(y)
                y[i] ~ Normal(x, σ)
            end
        end
        plan = _analyze_lift_eligibility(body, (:y,))
        @test plan isa LiftPlan
        @test plan.hp_syms == [:σ]
    end
end
