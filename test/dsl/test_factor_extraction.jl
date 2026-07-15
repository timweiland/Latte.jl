using Test
using Latte
using Distributions, Random, SparseArrays
import GaussianMarkovRandomFields as G

# The loop-preserving factor extractor (`_emit_structured_prior_builder`) must recover the
# non-Gaussian latent prior of a state-space `@latte` model as a factor-graph builder whose
# `StructuredLatentPrior` is numerically identical to the monolithic `AutoDiffLatentPrior`.
# This pins the codegen (increment 1): walk the body, build the builder, compare `local_quadratic`.

isdefined(@__MODULE__, :shared_sam) || include("shared_models.jl")

@testset "Factor extraction codegen" begin
    nA, nY = 4, 10

    # The age-structured SAM (shared_sam, see shared_models.jl): two coupled nonlinear latent
    # fields (logN, logF) with a survival recursion and a Baranov catch observation.
    # The model body, verbatim, as an Expr — the macro hook (increment 3) hands this in; here we
    # supply it directly to exercise the codegen in isolation.
    body = quote
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

    # The walker must pick up exactly the 5 latent `~` sites — skipping the 3 hyperparameter
    # priors (scalar LHS) and the Baranov observation (`logC` is not a latent).
    templates = Latte._walk_factor_templates(body, latent_syms)
    @test length(templates) == 5

    builder_expr = Latte._emit_structured_prior_builder(
        body, (:nA, :nY), hp_names, prelude, latent_syms,
    )
    @test builder_expr !== nothing
    builder = Base.eval(@__MODULE__, builder_expr)

    # Monolithic prior straight from the macro.
    Random.seed!(20260618)
    logC = 8.0 .- 1.5 .+ 0.1 .* randn(nA * nY)
    lgm = shared_sam(logC, nA, nY)
    mono = lgm.latent_prior
    @test mono isa G.NonGaussianLatentPrior

    hp = (log_σN = -2.0, log_σF = -2.0, log_σc = -2.0)
    x0 = vcat(8.0 .+ 0.1 .* sin.(1:(nA * nY)), -1.5 .+ 0.1 .* cos.(1:(nA * nY)))

    patm = G.local_quadratic(mono, x0; hp...).Q
    pattern = SparseMatrixCSC(patm.m, patm.n, copy(patm.colptr), copy(patm.rowval), ones(Bool, nnz(patm)))
    layout = Dict(:logN => (0, (nA, nY)), :logF => (nA * nY, (nA, nY)))

    structured = Base.invokelatest(builder, layout, 2 * nA * nY, pattern, nA, nY)

    lqm = G.local_quadratic(mono, x0; hp...)
    lqs = G.local_quadratic(structured, x0; hp...)

    @test maximum(abs.(Matrix(lqm.Q) .- Matrix(lqs.Q))) < 1.0e-10
    @test maximum(abs.(lqm.h .- lqs.h)) < 1.0e-8
    @test abs(lqm.logp_ref - lqs.logp_ref) < 1.0e-8

    # And at a second, distinct latent point (the precision is value-dependent here).
    x1 = vcat(8.0 .+ 0.2 .* cos.(1:(nA * nY)), -1.5 .- 0.1 .* sin.(1:(nA * nY)))
    lqm1 = G.local_quadratic(mono, x1; hp...)
    lqs1 = G.local_quadratic(structured, x1; hp...)
    @test maximum(abs.(Matrix(lqm1.Q) .- Matrix(lqs1.Q))) < 1.0e-10
    @test maximum(abs.(lqm1.h .- lqs1.h)) < 1.0e-8
end
