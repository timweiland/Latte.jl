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

# A user-defined `LatentModel` subtype whose constructor name is in NO
# Latte allow-list — exercises shape-based recognition of arbitrary subtypes.
# Delegates the numeric contract to an inner `IIDModel` but reports its own
# `model_name`, so it is a genuinely distinct type.
struct MyCustomLatent{M} <: GaussianMarkovRandomFields.LatentModel
    inner::M
end
MyCustomLatent(n::Int) = MyCustomLatent(IIDModel(n))
Base.length(m::MyCustomLatent) = length(m.inner)
GaussianMarkovRandomFields.hyperparameters(m::MyCustomLatent) =
    GaussianMarkovRandomFields.hyperparameters(m.inner)
GaussianMarkovRandomFields.precision_matrix(m::MyCustomLatent; kw...) =
    GaussianMarkovRandomFields.precision_matrix(m.inner; kw...)
Distributions.mean(m::MyCustomLatent; kw...) = Distributions.mean(m.inner; kw...)
GaussianMarkovRandomFields.constraints(m::MyCustomLatent; kw...) =
    GaussianMarkovRandomFields.constraints(m.inner; kw...)
GaussianMarkovRandomFields.model_name(::MyCustomLatent) = :mycustom
(m::MyCustomLatent)(; kw...) = m.inner(; kw...)
(m::MyCustomLatent)(ws::GaussianMarkovRandomFields.GMRFWorkspace; kw...) = m.inner(ws; kw...)

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

        # `latent_components` exposes the recognized concrete prior keyed by
        # latent symbol; the DAG path is type-erased so it returns `nothing`.
        comps = latent_components(lgm)
        @test comps isa AbstractDict{Symbol, <:GaussianMarkovRandomFields.LatentModel}
        @test collect(keys(comps)) == [:x]
        @test comps[:x] isa RWModel{1}
        @test latent_components(ref) === nothing

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

    @testset "Multi-component — IID + RW1 composed via CombinedModel" begin
        n = 40
        p = 3
        X = randn(n, p)
        β_true = randn(p) .* 0.5
        x_true = cumsum(randn(n)) .* 0.3
        η_true = X * β_true .+ x_true
        y_obs = [rand(Poisson(exp(η); check_args = false)) for η in η_true]

        @latte function gam(y, X)
            τ_β ~ Gamma(2.0, 1.0)
            τ_x ~ Gamma(2.0, 1.0)
            β ~ IIDModel(size(X, 2))(; τ = τ_β)
            x ~ RW1Model(length(y))(; τ = τ_x)
            for i in eachindex(y)
                y[i] ~ Poisson(exp(dot(view(X, i, :), β) + x[i]); check_args = false)
            end
        end

        lgm = gam(y_obs, X)
        @test lgm isa Latte.LatentGaussianModel

        # Multi-component recognition: the latent is a RoutedLatentModel
        # wrapping an upstream CombinedModel whose components are the concrete
        # IIDModel and RW1Model, in body order.
        base = lgm.latent_prior isa Latte.AugmentedLatentModel ?
            lgm.latent_prior.base_model : lgm.latent_prior
        @test base isa Latte.RoutedLatentModel
        @test base.inner isa CombinedModel
        @test length(base.inner.components) == 2
        @test base.inner.components[1] isa IIDModel
        @test base.inner.components[2] isa RWModel{1}

        @model function gam_dppl(y, X)
            τ_β ~ Gamma(2.0, 1.0)
            τ_x ~ Gamma(2.0, 1.0)
            β ~ IIDModel(size(X, 2))(; τ = τ_β)
            x ~ RW1Model(length(y))(; τ = τ_x)
            for i in eachindex(y)
                y[i] ~ Poisson(exp(dot(view(X, i, :), β) + x[i]); check_args = false)
            end
        end
        ref = latte_from_dppl(gam_dppl(y_obs, X); random = (:β, :x))

        # Multi-component: ordered mapping in body order, each entry the
        # concrete inner model unwrapped from the CombinedModel.
        comps = latent_components(lgm)
        @test collect(keys(comps)) == [:β, :x]
        @test comps[:β] isa IIDModel
        @test comps[:x] isa RWModel{1}
        @test latent_components(ref) === nothing

        inla_rec = inla(lgm, y_obs; progress = false)
        inla_ref = inla(ref, y_obs; progress = false)

        mode_rec = convert(NamedTuple, hyperparameter_mode(inla_rec))
        mode_ref = convert(NamedTuple, hyperparameter_mode(inla_ref))
        @test mode_rec.τ_β ≈ mode_ref.τ_β rtol = 1.0e-3
        @test mode_rec.τ_x ≈ mode_ref.τ_x rtol = 1.0e-3

        base_rec = latent_marginals(inla_rec)[lgm.augmentation_info.base_latent_indices]
        base_ref = latent_marginals(inla_ref)[ref.augmentation_info.base_latent_indices]
        @test length(base_rec) == n + p
        @test mean.(base_rec) ≈ mean.(base_ref) rtol = 1.0e-3
        @test std.(base_rec) ≈ std.(base_ref) rtol = 1.0e-3
    end

    @testset "Arbitrary subtype — user-defined LatentModel recognized" begin
        n = 40
        z_true = randn(n) .* 0.5
        y_obs = [rand(Poisson(exp(zi); check_args = false)) for zi in z_true]

        @latte function custom_model(y)
            τ ~ Gamma(2.0, 1.0)
            z ~ MyCustomLatent(length(y))(; τ = τ)
            for i in eachindex(y)
                y[i] ~ Poisson(exp(z[i]); check_args = false)
            end
        end

        lgm = custom_model(y_obs)
        @test lgm isa Latte.LatentGaussianModel

        base = lgm.latent_prior isa Latte.AugmentedLatentModel ?
            lgm.latent_prior.base_model : lgm.latent_prior
        @test base isa Latte.RoutedLatentModel
        @test base.inner isa MyCustomLatent

        @model function custom_dppl(y)
            τ ~ Gamma(2.0, 1.0)
            z ~ MyCustomLatent(length(y))(; τ = τ)
            for i in eachindex(y)
                y[i] ~ Poisson(exp(z[i]); check_args = false)
            end
        end
        ref = latte_from_dppl(custom_dppl(y_obs); random = (:z,))

        comps = latent_components(lgm)
        @test collect(keys(comps)) == [:z]
        @test comps[:z] isa MyCustomLatent
        @test latent_components(ref) === nothing

        inla_rec = inla(lgm, y_obs; progress = false)
        inla_ref = inla(ref, y_obs; progress = false)

        mode_rec = convert(NamedTuple, hyperparameter_mode(inla_rec)).τ
        mode_ref = convert(NamedTuple, hyperparameter_mode(inla_ref)).τ
        @test mode_rec ≈ mode_ref rtol = 1.0e-3

        base_rec = latent_marginals(inla_rec)[lgm.augmentation_info.base_latent_indices]
        base_ref = latent_marginals(inla_ref)[ref.augmentation_info.base_latent_indices]
        @test mean.(base_rec) ≈ mean.(base_ref) rtol = 1.0e-3
        @test std.(base_rec) ≈ std.(base_ref) rtol = 1.0e-3
    end
end
