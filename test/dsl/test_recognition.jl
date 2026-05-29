using Test
using Latte
using DynamicPPL: @model
import DynamicPPL
using Distributions
using LinearAlgebra
using Random
using GaussianMarkovRandomFields

# Feature 1: the `@latte` macro recognizes concrete `LatentModel` subtypes on
# random `~` sites (e.g. `x ~ RW1Model(n)(; τ=τ)`) and preserves them as a
# `RoutedLatentModel` instead of type-erasing into a cached (μ, Q) latent.
#
# Recognition is macro-only: the bare `latte_from_dppl(@model(...); random=...)`
# path keeps today's DAG / sparse-AD behavior and serves as the numerical
# reference here.
@testset "Concrete LatentModel recognition" begin
    Random.seed!(20260529)

    @testset "RW1 + Poisson — macro recognizes RW1Model" begin
        n = 50
        x_true = 1.0 .+ cumsum(randn(n)) .* 0.4
        y_obs = [rand(Poisson(exp(xi); check_args = false)) for xi in x_true]

        @latte function rw_poisson(y)
            τ ~ Gamma(2.0, 1.0)
            x ~ RW1Model(length(y))(; τ = τ)
            for i in eachindex(y)
                y[i] ~ Poisson(exp(x[i]); check_args = false)
            end
        end

        lgm = rw_poisson(y_obs)
        @test lgm isa Latte.LatentGaussianModel

        # Recognition: latent prior is a RoutedLatentModel wrapping the
        # concrete RW1Model, NOT a type-erased cached latent. (Auto-augmentation
        # for the Poisson fast path wraps the base latent, so unwrap it.)
        base = lgm.latent_prior isa Latte.AugmentedLatentModel ?
            lgm.latent_prior.base_model : lgm.latent_prior
        @test base isa Latte.RoutedLatentModel
        @test base.inner isa RWModel{1}

        # Numerical agreement with the bare DAG path.
        @model function rw_poisson_dppl(y)
            τ ~ Gamma(2.0, 1.0)
            x ~ RW1Model(length(y))(; τ = τ)
            for i in eachindex(y)
                y[i] ~ Poisson(exp(x[i]); check_args = false)
            end
        end
        ref = latte_from_dppl(rw_poisson_dppl(y_obs); random = (:x,))

        inla_rec = inla(lgm, y_obs; progress = false)
        inla_ref = inla(ref, y_obs; progress = false)

        mode_rec = convert(NamedTuple, hyperparameter_mode(inla_rec)).τ
        mode_ref = convert(NamedTuple, hyperparameter_mode(inla_ref)).τ
        @test mode_rec ≈ mode_ref rtol = 1.0e-3

        base_rec = latent_marginals(inla_rec)[lgm.augmentation_info.base_latent_indices]
        base_ref = latent_marginals(inla_ref)[ref.augmentation_info.base_latent_indices]
        @test length(base_rec) == n
        @test length(base_ref) == n
        @test mean.(base_rec) ≈ mean.(base_ref) rtol = 1.0e-3
        @test std.(base_rec) ≈ std.(base_ref) rtol = 1.0e-3
    end
end
