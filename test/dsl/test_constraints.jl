using Test
using Latte
using DynamicPPL: @model
using Distributions
using GaussianMarkovRandomFields
using LinearAlgebra
using Random
import GaussianMarkovRandomFields: constraints

# Hard linear equality constraints on a latent Gaussian component (typical
# example: a sum-to-zero constraint on an IID random-effect block, used
# for identifiability with an intercept).
#
# Latte's `IIDModel(n, constraint=:sumtozero)(τ=τ)` materialises a
# `ConstrainedGMRF`, which is `AbstractMvNormal`-compatible and so can
# appear directly as a prior in a `@model`:
#
#     u ~ IIDModel(n, constraint=:sumtozero)(τ = τ)
#
# `latte_from_dppl` needs to detect that and forward the `(A, e)` pair all
# the way to the `LatentGaussianModel`'s latent prior so downstream
# inference (mode finding / GA) sees and respects the constraint.
@testset "DPPL adapter: constrained atomic Gaussian priors" begin

    @testset "sum-to-zero constraint surfaces on the LGM's latent prior" begin
        @model function m(y, n_iid)
            τ ~ Gamma(2, 1)
            u ~ IIDModel(n_iid, constraint = :sumtozero)(τ = τ)
            for i in eachindex(y)
                y[i] ~ Poisson(exp(u[i]); check_args = false)
            end
        end

        Random.seed!(11)
        n = 8
        y_obs = rand(Poisson(1.0), n)

        lgm = latte_from_dppl(m(y_obs, n); random = (:u,))

        # The augmented latent prior must carry the constraint. Augmentation
        # prepends `n` η-positions before the base latent, so the constraint
        # matrix gets zero-padded on the left.
        cns = constraints(lgm.latent_prior; τ = 1.0)
        @test cns !== nothing
        A_c, e_c = cns
        @test size(A_c, 1) == 1
        @test size(A_c, 2) == n + n         # η-block (n) + base u (n)
        @test all(iszero, A_c[:, 1:n])      # η-block zero-padded
        @test A_c[:, (n + 1):end] == ones(1, n)  # sum-to-zero over u
        @test e_c == [0.0]
    end

    @testset "Gaussian approximation at fixed θ respects sum-to-zero" begin
        # The load-bearing property: given a DPPL-built LGM with a
        # constrained base latent, the Laplace/Gaussian approximation at
        # any θ must produce a posterior mode that satisfies the
        # constraint. Downstream INLA machinery (grid exploration,
        # per-variable Laplace marginals) is tested elsewhere and has its
        # own subtleties around joint vs. per-marginal means.
        @model function m(y, n_iid)
            τ ~ Gamma(2, 1)
            u ~ IIDModel(n_iid, constraint = :sumtozero)(τ = τ)
            for i in eachindex(y)
                y[i] ~ Poisson(exp(u[i]); check_args = false)
            end
        end

        Random.seed!(12)
        n = 40
        u_true = randn(n) .* 0.6
        u_true .-= sum(u_true) / n
        y_obs = rand.(Poisson.(exp.(u_true)))

        lgm = latte_from_dppl(m(y_obs, n); random = (:u,))

        # Materialise prior + likelihood at τ=1 and run the GA directly.
        y_norm = Latte._normalize_observations(y_obs, lgm.observation_model)
        obs_lik = lgm.observation_model(y_norm; τ = 1.0)
        latent = lgm.latent_prior(; τ = 1.0)
        ga = gaussian_approximation(latent, obs_lik)

        # The augmented latent has η (length n) followed by u (length n).
        # The constraint sits on the u block.
        base_idx = lgm.augmentation_info.base_latent_indices
        u_mode = mean(ga)[base_idx]
        @test abs(sum(u_mode)) < 1.0e-6
    end
end
