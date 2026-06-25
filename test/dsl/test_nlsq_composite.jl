using Test
using Latte
using GaussianMarkovRandomFields: IIDModel
using Distributions
using Statistics
using Random

# NLS as one component of a composite observation model. A nonlinear-in-x Normal
# block with a hyperparameter-dependent mean and a linear Normal block have
# different hyperparameter dependencies, so they split into two obs groups: the
# nonlinear one is dispatched to NonlinearLeastSquares (its hyperparameters routed
# through the composite passthrough), the linear one stays on the
# exponential-family path.
@testset "composite NLS: nonlinear hp-dependent group + linear group" begin
    n = 6
    Random.seed!(7)
    y1 = exp.(0.2 .* randn(n)) .+ 0.1 .* randn(n)
    y2 = 0.2 .* randn(n) .+ 0.2 .* randn(n)
    y = vcat(y1, y2)

    @latte function comp_nls(y1, y2, n)
        α ~ truncated(Normal(1.0, 0.5); lower = 0.1)
        τ ~ truncated(Normal(1.0, 0.5); lower = 0.1)
        x ~ IIDModel(n)(τ = τ)
        for i in eachindex(y1)
            y1[i] ~ Normal(exp(α * x[i]), 0.1)   # nonlinear + hp-dependent mean → NLS
        end
        for i in eachindex(y2)
            y2[i] ~ Normal(x[i], 0.2)            # linear → exponential-family
        end
    end

    lgm = comp_nls(y1, y2, n)
    @test occursin("NonlinearLeastSquares", string(typeof(lgm.observation_model)))

    res = inla(comp_nls(y1, y2, n), y; latent_marginalization_method = GaussianMarginal(), progress = false)
    lm = latent_marginals(res)
    @test all(m -> isfinite(mean(m)) && isfinite(std(m)), lm)

    # The opt-out reaches the composite group path: nls = false forces the exact
    # AD obs Hessian for the nonlinear group.
    ad_lgm = comp_nls(y1, y2, n; nls = false)
    @test !occursin("NonlinearLeastSquares", string(typeof(ad_lgm.observation_model)))
    @test occursin("AutoDiff", string(typeof(ad_lgm.observation_model)))

    # The mean hyperparameter α is genuinely routed into the residual (not frozen
    # at the probe value): the Gauss–Newton and exact-AD paths share the mode, so
    # latent means and the α posterior agree closely (the GN approximation shows
    # only in the latent variances).
    res_ad = inla(comp_nls(y1, y2, n; nls = false), y; latent_marginalization_method = GaussianMarginal(), progress = false)
    lm_ad = latent_marginals(res_ad)
    @test maximum(abs, mean.(lm) .- mean.(lm_ad)) < 5.0e-3
    @test isapprox(mean(res.hyperparameter_marginals[:α]), mean(res_ad.hyperparameter_marginals[:α]); atol = 0.05)
    # α is informed by the data (a frozen residual would leave it at the prior mode).
    @test mean(res.hyperparameter_marginals[:α]) > 1.05
end
