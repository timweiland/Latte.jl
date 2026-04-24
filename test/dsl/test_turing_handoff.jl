using Test
using Latte
using DynamicPPL: @model
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
end
