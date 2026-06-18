using Test
using Latte
using Distributions, Random, SparseArrays
import GaussianMarkovRandomFields as G

# End-to-end: the tutorial-form state-space `@latte` model (natural mortality as a body-local
# constant `M = 0.2`, a Baranov catch likelihood with a per-iteration local `Z`) must automatically
# build BOTH a factor-graph `StructuredLatentPrior` and a `StructuredObservationModel`. Each engages
# only via its guard, so `isa Structured*` is itself the assertion that the extracted version
# reproduced the monolithic one. The prior is additionally checked numerically against the monolith.

@testset "Structured prior + obs via @latte macro" begin
    nA, nY = 4, 10

    @latte function sam_macro(logC, nA, nY)
        log_σN ~ Normal(-2.0, 0.5)
        log_σF ~ Normal(-2.0, 0.5)
        log_σc ~ Normal(-2.0, 0.5)
        σN = exp(log_σN); σF = exp(log_σF); σc = exp(log_σc)
        M = 0.2
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
                logN[a, y] ~ Normal(logN[a - 1, y - 1] - exp(logF[a - 1, y - 1]) - M, σN)
            end
        end
        for y in 1:nY, a in 1:nA
            Z = exp(logF[a, y]) + M
            logC[(y - 1) * nA + a] ~ Normal(logN[a, y] + logF[a, y] - log(Z) + log1p(-exp(-Z)), σc)
        end
    end

    Random.seed!(20260618)
    logC = 8.0 .- 1.5 .+ 0.1 .* randn(nA * nY)

    # Both factor graphs must auto-engage (each only if its guard confirmed a match with the monolith).
    lgm = sam_macro(logC, nA, nY)
    @test lgm.latent_prior isa G.StructuredLatentPrior
    @test lgm.observation_model isa G.StructuredObservationModel

    # Explicit numeric check of the prior against the monolithic baseline (constant `M` carried into
    # the closures correctly).
    dppl = Latte._LATTE_DPPL_CONSTRUCTORS[sam_macro](logC, nA, nY)
    mono, path = Latte.build_latent_model(dppl, (:logN, :logF), (:log_σN, :log_σF, :log_σc))
    @test path === :sparse_nongaussian
    @test !(mono isa G.StructuredLatentPrior)

    hp = (log_σN = -2.0, log_σF = -2.0, log_σc = -2.0)
    for x in (
            vcat(8.0 .+ 0.1 .* sin.(1:(nA * nY)), -1.5 .+ 0.1 .* cos.(1:(nA * nY))),
            vcat(8.0 .+ 0.2 .* cos.(1:(nA * nY)), -1.5 .- 0.1 .* sin.(1:(nA * nY))),
        )
        lqs = G.local_quadratic(lgm.latent_prior, x; hp...)
        lqm = G.local_quadratic(mono, x; hp...)
        @test maximum(abs.(Matrix(lqs.Q) .- Matrix(lqm.Q))) < 1.0e-10
        @test maximum(abs.(lqs.h .- lqm.h)) < 1.0e-8
        @test abs(lqs.logp_ref - lqm.logp_ref) < 1.0e-8
    end
end
