using Test
using Latte
using Distributions, Random, SparseArrays
import GaussianMarkovRandomFields as G

# The recognition guard in `build_latent_model`: when handed a structured-prior spec it builds the
# factor-graph `StructuredLatentPrior`, verifies it reproduces the monolithic `AutoDiffLatentPrior`
# at probe points, and uses it only on a match. A bad spec (throws / wrong type / wrong values)
# must fall back to the monolithic prior, so the structured path can never change recognition.

@testset "Structured prior guard" begin
    nA, nY = 4, 10

    @latte function sam_guard(logC, nA, nY)
        log_σN ~ Normal(-2.0, 0.5)
        log_σF ~ Normal(-2.0, 0.5)
        log_σc ~ Normal(-2.0, 0.5)
        σN = exp(log_σN); σF = exp(log_σF); σc = exp(log_σc)
        logN = Matrix{Real}(undef, nA, nY)
        logF = Matrix{Real}(undef, nA, nY)
        for a in 1:nA
            logN[a, 1] ~ Normal(8.0, 0.5)
            logF[a, 1] ~ Normal(-1.5, 0.5)
        end
        for y in 2:nY
            for a in 1:nA
                logF[a, y] ~ Normal(logF[a, y - 1], σF)
            end
            logN[1, y] ~ Normal(logN[1, y - 1], σN)
            for a in 2:nA
                logN[a, y] ~ Normal(logN[a - 1, y - 1] - exp(logF[a - 1, y - 1]) - 0.2, σN)
            end
        end
        for y in 1:nY, a in 1:nA
            Z = exp(logF[a, y]) + 0.2
            logC[(y - 1) * nA + a] ~ Normal(logN[a, y] + logF[a, y] - log(Z) + log1p(-exp(-Z)), σc)
        end
    end

    latent_syms = (:logN, :logF)
    hp_names = (:log_σN, :log_σF, :log_σc)
    prelude = [:(σN = exp(log_σN)), :(σF = exp(log_σF)), :(σc = exp(log_σc))]

    correct_body = quote
        for a in 1:nA
            logN[a, 1] ~ Normal(8.0, 0.5)
            logF[a, 1] ~ Normal(-1.5, 0.5)
        end
        for y in 2:nY
            for a in 1:nA
                logF[a, y] ~ Normal(logF[a, y - 1], σF)
            end
            logN[1, y] ~ Normal(logN[1, y - 1], σN)
            for a in 2:nA
                logN[a, y] ~ Normal(logN[a - 1, y - 1] - exp(logF[a - 1, y - 1]) - 0.2, σN)
            end
        end
    end

    # Same structure, but the logN initial mean is shifted (8.0 → 9.0): valid StructuredLatentPrior,
    # numerically *wrong* — its gradient `h` differs from the monolithic prior's.
    wrong_body = quote
        for a in 1:nA
            logN[a, 1] ~ Normal(9.0, 0.5)
            logF[a, 1] ~ Normal(-1.5, 0.5)
        end
        for y in 2:nY
            for a in 1:nA
                logF[a, y] ~ Normal(logF[a, y - 1], σF)
            end
            logN[1, y] ~ Normal(logN[1, y - 1], σN)
            for a in 2:nA
                logN[a, y] ~ Normal(logN[a - 1, y - 1] - exp(logF[a - 1, y - 1]) - 0.2, σN)
            end
        end
    end

    correct_builder = Base.eval(
        @__MODULE__,
        Latte._emit_structured_prior_builder(correct_body, (:nA, :nY), hp_names, prelude, latent_syms),
    )
    wrong_builder = Base.eval(
        @__MODULE__,
        Latte._emit_structured_prior_builder(wrong_body, (:nA, :nY), hp_names, prelude, latent_syms),
    )

    layout = Dict(:logN => (0, (nA, nY)), :logF => (nA * nY, (nA, nY)))
    layout_builder = (a, b) -> layout
    correct_spec = (builder = correct_builder, layout_builder = layout_builder, posarg_vals = (nA, nY))

    Random.seed!(20260618)
    logC = 8.0 .- 1.5 .+ 0.1 .* randn(nA * nY)
    dppl = Latte._LATTE_DPPL_CONSTRUCTORS[sam_guard](logC, nA, nY)

    # No spec → today's monolithic behaviour.
    mono, path = Latte.build_latent_model(dppl, latent_syms, hp_names)
    @test path === :sparse_nongaussian
    @test !(mono isa G.StructuredLatentPrior)
    @test mono isa G.NonGaussianLatentPrior

    # Valid spec → the structured prior is accepted.
    structured, path_s = Latte.build_latent_model(dppl, latent_syms, hp_names; structured_spec = correct_spec)
    @test path_s === :sparse_nongaussian
    @test structured isa G.StructuredLatentPrior

    # The accepted structured prior must agree with the monolithic one.
    hp = (log_σN = -2.0, log_σF = -2.0, log_σc = -2.0)
    x0 = vcat(8.0 .+ 0.1 .* sin.(1:(nA * nY)), -1.5 .+ 0.1 .* cos.(1:(nA * nY)))
    lqm = G.local_quadratic(mono, x0; hp...)
    lqs = G.local_quadratic(structured, x0; hp...)
    @test maximum(abs.(lqm.h .- lqs.h)) < 1.0e-8

    # Fallback cases: each must return the monolithic prior, not a StructuredLatentPrior.
    throwing_spec = (builder = (l, n, p, a, b) -> error("boom"), layout_builder = layout_builder, posarg_vals = (nA, nY))
    fb1, _ = Latte.build_latent_model(dppl, latent_syms, hp_names; structured_spec = throwing_spec)
    @test !(fb1 isa G.StructuredLatentPrior)

    wrongtype_spec = (builder = (l, n, p, a, b) -> 42, layout_builder = layout_builder, posarg_vals = (nA, nY))
    fb2, _ = Latte.build_latent_model(dppl, latent_syms, hp_names; structured_spec = wrongtype_spec)
    @test !(fb2 isa G.StructuredLatentPrior)

    # A layout_builder that throws must also fall back (codegen miss → monolithic, not a crash).
    throwing_layout_spec = (builder = correct_builder, layout_builder = (a, b) -> error("nope"), posarg_vals = (nA, nY))
    fb_layout, _ = Latte.build_latent_model(dppl, latent_syms, hp_names; structured_spec = throwing_layout_spec)
    @test !(fb_layout isa G.StructuredLatentPrior)

    # A prior that BUILDS fine but throws when `local_quadratic` differentiates its factor closure
    # (e.g. a loop variable left in a non-index position → an unbound symbol in the closure). The
    # guard must catch this at probe time too, not crash model construction.
    evil_closure = (vals, θ) -> vals[1] * NONEXISTENT_GUARD_SYMBOL
    evil_group = G.LatentFactorGroup([(i,) for i in 1:(2 * nA * nY)], evil_closure)
    evil_builder = (l, n, p, a, b) -> G.StructuredLatentPrior(n, (evil_group,), p; hyperparams = hp_names)
    evil_spec = (builder = evil_builder, layout_builder = layout_builder, posarg_vals = (nA, nY))
    fb_probe, _ = Latte.build_latent_model(dppl, latent_syms, hp_names; structured_spec = evil_spec)
    @test !(fb_probe isa G.StructuredLatentPrior)
    @test fb_probe isa G.NonGaussianLatentPrior

    wrong_spec = (builder = wrong_builder, layout_builder = layout_builder, posarg_vals = (nA, nY))
    fb3, _ = Latte.build_latent_model(dppl, latent_syms, hp_names; structured_spec = wrong_spec)
    @test !(fb3 isa G.StructuredLatentPrior)
    @test fb3 isa G.NonGaussianLatentPrior
end
