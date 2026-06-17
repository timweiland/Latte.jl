using Test
using Latte
using GaussianMarkovRandomFields: NonGaussianLatentPrior, AutoDiffLatentPrior, local_quadratic
using Distributions
using DynamicPPL
using LinearAlgebra
using SparseArrays
using Random
import ForwardDiff
import GaussianMarkovRandomFields as GMRFs

# Recognition of a NONLINEAR latent coupling. A genuinely non-Gaussian prior (its Hessian
# depends on the latent) is recognised as a `NonGaussianLatentPrior` (GMRFs `AutoDiffLatentPrior`)
# and fit by iterated Laplace — NOT linearised once at x=0. This supersedes the earlier
# linearise-once `:sparse_ad` fallback (`CachedSparseADLatentModel`), which silently treated such
# priors as Gaussian; the value-level gate in `build_latent_model` now routes them to the
# iterated-Laplace path, and the previous outer-AD-over-hyperparameters limitation is gone.

@testset "Non-linear prior coupling" begin
    Random.seed!(20260513)

    @testset "Non-linear edge → iterated-Laplace (AutoDiffLatentPrior) path" begin
        # `v[i]` depends nonlinearly on `u[i]` → the prior Hessian is value-dependent, so the
        # joint is non-Gaussian and is recognised as such.
        @latte function nonlinear_edge(y)
            σ ~ Gamma(2.0, 1.0)
            @random u ~ MvNormal(zeros(2), 1.0)
            @random v ~ MvNormal(u .^ 2, 1.0)
            for i in eachindex(y)
                y[i] ~ Normal(v[i], σ)
            end
        end

        y = [0.1, 0.2]
        lgm = nonlinear_edge(y)
        @test lgm.latent_prior isa NonGaussianLatentPrior

        # At x = 0 the prior log-density is logp = -0.5(‖u‖² + ‖v - u²‖²) (+ const), so the local
        # quadratic there has Q = -∇²logp|₀ = I(4) and gradient 0 (⇒ natural coefficient h = 0).
        # This matches what the old linearise-once path computed *at the mode-zero point*; the
        # difference is that the new path re-linearises per Newton iterate away from zero.
        lq = local_quadratic(lgm.latent_prior, zeros(4); σ = 0.8)
        @test Matrix(lq.Q) ≈ Matrix(I, 4, 4) atol = 1.0e-10
        @test lq.h ≈ zeros(4) atol = 1.0e-10
    end

    @testset "Non-linear prior works under outer AD over hyperparameters" begin
        # Previously this model hit a SparseConnectivityTracer / ForwardDiff.Dual clash on the
        # linearise-once path and could only be fit with `FiniteDiffStrategy`. The iterated-Laplace
        # path differentiates the marginal likelihood through the Implicit Function Theorem, so the
        # default `ADStrategy` (ForwardDiff over the hyperparameters) now runs end to end.
        @latte function nonlinear_edge_dual(y)
            τ ~ Gamma(2.0, 1.0)
            σ ~ Gamma(2.0, 1.0)
            @random u ~ MvNormal(zeros(2), 1.0 / sqrt(τ))
            @random v ~ MvNormal(u .^ 2, 1.0)
            for i in eachindex(y)
                y[i] ~ Normal(v[i], σ)
            end
        end

        y = [0.1, 0.2]
        lgm = nonlinear_edge_dual(y)
        @test lgm.latent_prior isa NonGaussianLatentPrior

        # Default ADStrategy (outer ForwardDiff over τ, σ) — must run, not throw.
        result = inla(lgm, y; progress = false)
        @test all(isfinite, mean.(latent_marginals(result)))
    end
end
