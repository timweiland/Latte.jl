using Test
using Latte
using DynamicPPL
using Distributions
using GaussianMarkovRandomFields: IIDModel
using Turing
using Random
using Statistics

# The same @model used by `inla()` / `tmb()` / `hmc_laplace()` can also be
# fed directly into Turing for MCMC. This is a handoff smoke test: no new
# API — just verify that a DPPL model built out of Latte primitives
# samples cleanly under NUTS and that the posterior on the latent field
# agrees with INLA within MC error.
#
# Caveat: constrained GMRF priors (Besag with sum-to-zero, rank-deficient
# RWs) don't sample cleanly under HMC without a reparameterisation onto
# the constraint manifold — that's tracked separately. Here we stick to
# unconstrained priors (IIDModel, MaternModel, etc.).
@testset "Turing NUTS handoff" begin

    @testset "IIDModel + Poisson likelihood" begin
        Random.seed!(20260424)

        @model function iid_poisson(y, n)
            τ ~ PCPrior.Precision(1.0, α = 0.01)
            x ~ IIDModel(n)(τ = τ)
            for i in eachindex(y)
                y[i] ~ Poisson(exp(x[i]); check_args = false)
            end
        end

        n = 15
        true_x = randn(n) .* 0.4 .+ 0.8
        y = rand.(Poisson.(exp.(true_x)))
        model = iid_poisson(y, n)

        chain = sample(model, NUTS(), 2000; progress = false)
        x_nuts = [mean(chain[Symbol("x[$i]")]) for i in 1:n]

        lgm = latte_from_dppl(model; random = (:x,))
        result = inla(lgm, y; progress = false)
        x_inla = [mean(m) for m in result.latent_marginals[1:n]]

        # Latent posterior means should agree within ~0.15 under this
        # data size and sample count. (τ marginal is heavy-tailed; its
        # mean isn't a reliable cross-check, so we don't assert on it.)
        @test maximum(abs.(x_nuts .- x_inla)) < 0.15

        # Medians of τ should be within a factor of ~3 of each other
        τ_nuts_med = median(chain[:τ])
        τ_inla_med = median(result.hyperparameter_marginals[1])
        @test 1 / 3 < τ_nuts_med / τ_inla_med < 3
    end

    @testset "@latte dppl_model couples hyperparameters to the latent" begin
        # `Latte.dppl_model` of an @latte model with a *recognized* latent
        # (IIDModel/AR1Model/…) must be a faithful generative model. A regression
        # guard: the recognized-latent probe used to drop the latent's
        # hyperparameters (replacing it with a θ-independent MvNormal(0,I)), so
        # Turing's NUTS saw no coupling and the hyperparameter collapsed onto its
        # prior.

        ## Unit: the probe must carry the hyperparameters into the prior.
        m = IIDModel(8)
        x_probe = randn(8)
        @test logpdf(Latte._recognized_latent_probe(m, (τ = 0.5,)), x_probe) !=
            logpdf(Latte._recognized_latent_probe(m, (τ = 50.0,)), x_probe)

        ## Behavioural: Turing on the @latte handoff agrees with INLA, and the
        ## hyperparameter is updated away from its prior.
        Random.seed!(20260424)
        @latte function iid_poisson_latte(y, n)
            τ ~ PCPrior.Precision(1.0, α = 0.01)
            x ~ IIDModel(n)(τ = τ)
            for i in eachindex(y)
                y[i] ~ Poisson(exp(x[i]); check_args = false)
            end
        end

        n = 20
        true_x = randn(n) .* 0.6 .+ 0.8
        y = rand.(Poisson.(exp.(true_x)))

        dppl = Latte.dppl_model(iid_poisson_latte)(y, n)
        chain = sample(dppl, NUTS(), 1500; progress = false)
        τ_nuts_med = median(chain[:τ])

        result = inla(iid_poisson_latte(y, n), y; progress = false)
        τ_inla_med = median(hyperparameter_marginals(result, :τ)[1])
        τ_prior_med = median(rand(PCPrior.Precision(1.0, α = 0.01), 50_000))

        @test 1 / 3 < τ_nuts_med / τ_inla_med < 3                  # agrees with INLA
        @test abs(log(τ_nuts_med) - log(τ_prior_med)) > log(2)     # not the prior
    end
end
