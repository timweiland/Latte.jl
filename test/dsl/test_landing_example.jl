using Test
using Latte
using GaussianMarkovRandomFields
using Distributions
using DynamicPPL
using LinearAlgebra
using SparseArrays
using Random

# Drift guard for the landing-page hero example (docs/src/components/Landing.vue).
# The model body below must stay character-for-character in sync with the hero
# snippet. If the @latte macro, the recognized Poisson-log fast path, or the
# Besag/PCPrior constructors drift, this fails — so the headline code on the
# website can never silently go stale or stop running.
@testset "Landing hero: disease mapping (Besag), all three engines" begin
    @latte function disease(y, E, W)
        β ~ MvNormal(zeros(1), 100.0 * I(1))
        τ ~ PCPrior.Precision(1.0, α = 0.01)
        u ~ BesagModel(W; normalize_var = Val{true}())(τ = τ)
        for i in eachindex(y)
            y[i] ~ Poisson(E[i] * exp(β[1] + u[i]))
        end
    end

    # Small synthetic areal dataset: a chain-adjacency graph of n regions.
    Random.seed!(2026)
    n = 16
    W = spzeros(Float64, n, n)
    for i in 1:(n - 1)
        W[i, i + 1] = 1.0
        W[i + 1, i] = 1.0
    end
    E = fill(50.0, n)
    u_true = 0.3 .* randn(n)
    u_true .-= sum(u_true) / n              # center (Besag is sum-to-zero)
    y = [rand(Poisson(E[i] * exp(0.1 + u_true[i]))) for i in 1:n]

    lgm = disease(y, E, W)

    # The hero's headline claim: the same model runs under all three engines.
    inla_r = inla(lgm, y; progress = false)
    @test inla_r isa Latte.InferenceResult
    @test converged(inla_r)

    tmb_r = tmb(lgm, y)
    @test tmb_r isa Latte.InferenceResult

    hmc_r = hmc_laplace(lgm, y)
    @test hmc_r isa Latte.InferenceResult
end
