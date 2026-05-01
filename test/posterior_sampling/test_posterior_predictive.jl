using Test
using Latte
using DynamicPPL: @model
using Distributions
using GaussianMarkovRandomFields: IIDModel
using Random
using Statistics

@testset "posterior_predictive" begin

    # Reusable Poisson model — IID latent, unconstrained.
    @model function pois_model(y, n)
        τ ~ PCPrior.Precision(1.0, α = 0.01)
        x ~ IIDModel(n)(τ = τ)
        for i in eachindex(y)
            y[i] ~ Poisson(exp(x[i]); check_args = false)
        end
    end

    Random.seed!(42)
    n = 20
    true_x = randn(n) .* 0.3 .+ 0.5
    y = rand.(Poisson.(exp.(true_x)))

    lgm = latte_from_dppl(pois_model(y, n); random = (:x,))
    result = inla(lgm, y; progress = false)

    @testset "posterior_predictive returns n_draws × n_obs matrix" begin
        y_rep = posterior_predictive(result, 500)
        @test size(y_rep) == (500, n)
        @test eltype(y_rep) <: Integer  # Poisson → Int draws
    end

    @testset "ppc_stat evaluates statistic on observed and replicated" begin
        y_rep = posterior_predictive(result, 300)
        T_obs, T_rep = ppc_stat(mean, y, y_rep)
        @test T_obs == mean(y)
        @test length(T_rep) == 300
        @test all(isfinite, T_rep)
    end

    @testset "bayesian_pvalue is in [0, 1] and reasonable for well-fit stat" begin
        y_rep = posterior_predictive(result, 500)
        # The mean of a well-fit Poisson model's predictive should straddle
        # the observed mean — Bayesian p-value not too close to 0 or 1.
        p = bayesian_pvalue(mean, y, y_rep)
        @test 0 <= p <= 1
        # Loose bound: Monte-Carlo + posterior-shape variation can put a
        # well-fit p-value within ~0.05 of either tail.
        @test 0.05 < p < 0.95

        # And the helper accepts arbitrary callables.
        p_max = bayesian_pvalue(maximum, y, y_rep)
        @test 0 <= p_max <= 1
    end

    @testset "works for TMBResult too" begin
        tmb_result = tmb(lgm, y)
        y_rep = posterior_predictive(tmb_result, 200)
        @test size(y_rep) == (200, n)
    end

    @testset "prediction via missing observations: y_rep matches observed count" begin
        # Drop a few observations via missing; ensure predictive draws only
        # have columns for the observed subset, not for the missing positions.
        y_missing = poisson_observations(
            counts = Union{Int, Missing}[i in [3, 9, 15] ? missing : y[i] for i in 1:n],
        )
        n_observed = n - 3
        result_pred = inla(lgm, y_missing; progress = false)
        y_rep = posterior_predictive(result_pred, 200)
        @test size(y_rep) == (200, n_observed)
    end
end
