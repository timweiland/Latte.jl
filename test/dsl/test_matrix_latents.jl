using Test
using Latte
using GaussianMarkovRandomFields: NonGaussianLatentPrior
using Distributions, Random
using Statistics: mean

# Matrix-indexed latents (e.g. `logN[a, y]`) must flow through the whole pipeline, including
# the pointwise observation path (WAIC / CPO / DIC accumulators). DPPL's `_default_vnt` already
# lays a matrix latent out correctly for the main likelihood; the pointwise path seeds its
# VarInfo from the SAME layout via `InitFromVector`, so a matrix-indexed latent reconstructs
# instead of failing with "No value for logF[1, 2]". This test pins that: a matrix-indexed
# nonlinear SAM must agree with its flat-vector equivalent.

@testset "Matrix-indexed latents" begin
    nA, nY = 2, 3
    K = nA * nY
    am(k) = ((k - 1) % nA) + 1
    ym(k) = ((k - 1) ÷ nA) + 1
    fl(a, y) = (y - 1) * nA + a

    # Two coupled nonlinear latent fields (logN, logF) with a survival recursion that couples
    # logN to exp(logF) and a Baranov catch observation — written with MATRIX indexing.
    @latte function sam_mat(logC, nA, nY)
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
        for k in eachindex(logC)
            a = am(k); y = ym(k)
            Z = exp(logF[a, y]) + 0.2
            predC = logN[a, y] + logF[a, y] - log(Z) + log1p(-exp(-Z))
            logC[k] ~ Normal(predC, σc)
        end
    end

    # The identical model with FLAT (vector) latents — the established workaround.
    @latte function sam_flat(logC, nA, nY)
        log_σN ~ Normal(-2.0, 0.5)
        log_σF ~ Normal(-2.0, 0.5)
        log_σc ~ Normal(-2.0, 0.5)
        σN = exp(log_σN); σF = exp(log_σF); σc = exp(log_σc)
        n = nA * nY
        logN = Vector{Real}(undef, n)
        logF = Vector{Real}(undef, n)
        for a in 1:nA
            logN[a] ~ Normal(8.0, 0.5)
            logF[a] ~ Normal(-1.5, 0.5)
        end
        for y in 2:nY
            for a in 1:nA
                logF[(y - 1) * nA + a] ~ Normal(logF[(y - 2) * nA + a], σF)
            end
            logN[(y - 1) * nA + 1] ~ Normal(logN[(y - 2) * nA + 1], σN)
            for a in 2:nA
                jp = (y - 2) * nA + (a - 1)
                logN[(y - 1) * nA + a] ~ Normal(logN[jp] - exp(logF[jp]) - 0.2, σN)
            end
        end
        for k in eachindex(logC)
            Z = exp(logF[k]) + 0.2
            predC = logN[k] + logF[k] - log(Z) + log1p(-exp(-Z))
            logC[k] ~ Normal(predC, σc)
        end
    end

    # Simulate one dataset (the two model forms share identical column-major flat layout, so
    # the same logC feeds both).
    Random.seed!(20260617)
    logN_t = zeros(K); logF_t = zeros(K)
    for a in 1:nA
        logN_t[a] = 8.0 + 0.1 * randn()
        logF_t[a] = -1.5 + 0.1 * randn()
    end
    for y in 2:nY
        for a in 1:nA
            logF_t[fl(a, y)] = logF_t[fl(a, y - 1)] + 0.1 * randn()
        end
        logN_t[fl(1, y)] = logN_t[fl(1, y - 1)] + 0.1 * randn()
        for a in 2:nA
            jp = fl(a - 1, y - 1)
            logN_t[fl(a, y)] = logN_t[jp] - exp(logF_t[jp]) - 0.2 + 0.1 * randn()
        end
    end
    logC = [
        let Z = exp(logF_t[k]) + 0.2
                logN_t[k] + logF_t[k] - log(Z) + log1p(-exp(-Z)) + 0.1 * randn()
        end for k in 1:K
    ]

    @testset "recognition + end-to-end (exercises the pointwise path)" begin
        lgm = sam_mat(logC, nA, nY)
        @test lgm.latent_prior isa NonGaussianLatentPrior
        @test length(lgm.latent_prior) == 2 * K

        # Default accumulators (WAIC / CPO / DIC) drive the pointwise path that used to throw
        # "No value for logF[1, 2]" on a matrix-indexed latent. It must run and stay finite.
        r = inla(lgm, logC; progress = false)
        @test all(isfinite, mean.(latent_marginals(r, :logN)))
        @test all(isfinite, mean.(latent_marginals(r, :logF)))
    end

    @testset "matrix layout agrees with the flat-vector workaround" begin
        r_mat = inla(sam_mat(logC, nA, nY), logC; progress = false)
        r_flat = inla(sam_flat(logC, nA, nY), logC; progress = false)
        for s in (:logN, :logF)
            m_mat = vec(collect(mean.(latent_marginals(r_mat, s))))
            m_flat = vec(collect(mean.(latent_marginals(r_flat, s))))
            @test m_mat ≈ m_flat atol = 1.0e-4
        end
    end
end
