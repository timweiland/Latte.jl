using Test
using Latte
using GaussianMarkovRandomFields: NonGaussianLatentPrior
using Distributions
using Turing
using Random
using Statistics

# End-to-end validation of the non-Gaussian state-space path against an MCMC gold standard.
# A small SAM (numbers- and fishing-mortality-at-age with a nonlinear survival recursion and a
# Baranov catch likelihood) is fit by `inla` and `tmb`, and the latent posterior is compared to
# NUTS run on the same model via the `@latte → dppl_model` handoff. NUTS is cheap at this size,
# so it runs live (no cached reference).

@testset "End-to-End: SAM vs NUTS" begin
    nA, nY = 3, 5
    n = nA * nY
    M = 0.2
    fl(a, y) = (y - 1) * nA + a

    @latte function sam(logC, nA, nY)
        log_σN ~ Normal(-2.0, 0.5)
        log_σF ~ Normal(-2.0, 0.5)
        log_σc ~ Normal(-2.0, 0.5)
        σN = exp(log_σN)
        σF = exp(log_σF)
        σc = exp(log_σc)
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
                logN[(y - 1) * nA + a] ~ Normal(logN[jp] - exp(logF[jp]) - M, σN)
            end
        end
        for k in 1:n
            Z = exp(logF[k]) + M
            logC[k] ~ Normal(logN[k] + logF[k] - log(Z) + log1p(-exp(-Z)), σc)
        end
    end

    ## Simulate one dataset from the generative process.
    Random.seed!(20260617)
    logN_t = zeros(n)
    logF_t = zeros(n)
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
            logN_t[fl(a, y)] = logN_t[jp] - exp(logF_t[jp]) - M + 0.1 * randn()
        end
    end
    logC = [
        let Z = exp(logF_t[k]) + M
                logN_t[k] + logF_t[k] - log(Z) + log1p(-exp(-Z)) + 0.1 * randn()
        end for k in 1:n
    ]

    lgm = sam(logC, nA, nY)
    @test lgm.latent_prior isa NonGaussianLatentPrior

    result = inla(lgm, logC; progress = false, accumulators = (MarginalLogLikelihoodStrategy(),))
    result_tmb = tmb(lgm, logC)

    mN_inla = mean.(latent_marginals(result, :logN))
    mF_inla = mean.(latent_marginals(result, :logF))
    mN_tmb = mean.(latent_marginals(result_tmb, :logN))
    mF_tmb = mean.(latent_marginals(result_tmb, :logF))

    ## NUTS gold standard on the same model via the handoff.
    dppl = Latte.dppl_model(sam)(logC, nA, nY)
    nuts_chain = sample(MersenneTwister(11), dppl, NUTS(500, 0.8), 800; progress = false)
    mN_nuts = [mean(nuts_chain[Symbol("logN[$k]")]) for k in 1:n]
    mF_nuts = [mean(nuts_chain[Symbol("logF[$k]")]) for k in 1:n]

    ## NUTS converges cleanly on this model (the survival recursion is well-conditioned here).
    @test sum(nuts_chain[:numerical_error]) == 0

    ## Iterated-Laplace INLA recovers the latent posterior means to within MCMC-level error.
    @test maximum(abs.(mN_inla .- mN_nuts)) < 0.15
    @test maximum(abs.(mF_inla .- mF_nuts)) < 0.15
    ## TMB (MAP + inner Laplace) lands in the same place.
    @test maximum(abs.(mN_tmb .- mN_nuts)) < 0.15
    @test maximum(abs.(mF_tmb .- mF_nuts)) < 0.15

    ## Hyperparameter medians agree with NUTS within a factor (weakly identified at this size).
    for (k, sym) in enumerate((:log_σN, :log_σF, :log_σc))
        σ_inla = exp(median(hyperparameter_marginals(result, sym)[1]))
        σ_nuts = exp(median(nuts_chain[sym]))
        @test 1 / 4 < σ_inla / σ_nuts < 4
    end
end
