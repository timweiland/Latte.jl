using Test
using Latte
using DynamicPPL
using Distributions
using GaussianMarkovRandomFields: AR1Model, IIDModel
using LinearAlgebra
using Random
using Statistics

# Regression: an AR(1) latent component exposed two boundary bugs.
#   Bug A — the AR1Correlation PC prior on ρ defined no `mode`, so the
#           mode-finder's initial guess crashed (the inla / find-mode path).
#   Bug B — tmb built its workspace from a blanket sentinel of 1.0 for every
#           hyperparameter, but ρ=1.0 is the AR(1) stationarity boundary, so
#           tmb (and hmc_laplace, which warm-starts from tmb) crashed.
@testset "AR(1) correlation: inla / tmb / hmc_laplace run end-to-end" begin
    @latte function ar1_iid(y, n)
        τ_ar ~ PCPrior.Precision(1.0, α = 0.01)
        ρ ~ PCPrior.AR1Correlation(0.7; α = 0.1, positive_only = true)
        τ_iid ~ PCPrior.Precision(1.0, α = 0.01)
        f ~ AR1Model(n)(τ = τ_ar, ρ = ρ)
        u ~ IIDModel(n)(τ = τ_iid)
        for i in eachindex(y)
            y[i] ~ Poisson(exp(f[i] + u[i]); check_args = false)
        end
    end

    Random.seed!(20260606)
    n = 30
    ρ_true, σ_ar = 0.85, 0.6
    f = zeros(n)
    f[1] = randn() * σ_ar / sqrt(1 - ρ_true^2)
    for i in 2:n
        f[i] = ρ_true * f[i - 1] + randn() * σ_ar
    end
    η = f .+ randn(n) * 0.15 .- mean(f) .+ 0.8
    y = rand.(Poisson.(exp.(η)))
    lgm = ar1_iid(y, n)

    # Bug A: the mode-finder init must not crash on the AR1Correlation prior.
    r_inla = inla(lgm, y; progress = false)
    @test 0 < quantile(hyperparameter_marginals(r_inla, :ρ)[1], 0.5) < 1

    # Bug B: tmb / hmc_laplace must not seed their workspace at ρ=1.0.
    r_tmb = tmb(lgm, y; diff_strategy = FiniteDiffStrategy())
    @test 0 < quantile(hyperparameter_marginals(r_tmb, :ρ)[1], 0.5) < 1

    r_hmc = hmc_laplace(
        lgm, y; n_samples = 200, n_warmup = 200,
        diff_strategy = FiniteDiffStrategy(), rng = MersenneTwister(1),
    )
    @test 0 < median(vec(chain(r_hmc)[:ρ].data)) < 1
end
