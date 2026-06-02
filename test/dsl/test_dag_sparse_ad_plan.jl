using Test
using Latte
using Latte: CachedSparseADLatentModel
using Distributions
using DynamicPPL
using LinearAlgebra
using SparseArrays
using Random
import ForwardDiff
import GaussianMarkovRandomFields as GMRFs

# Phase 2 of the joint-precision caching work: pattern-cache the sparse-AD
# fallback path (`_build_joint_sparse_ad_latent`). The path is triggered
# when the analyzer can't conclude all-atomic-Gaussian-with-linear-edges
# (e.g. non-linear DAG edge). It used to allocate a fresh sparse Hessian
# on every call; the cached version fills a pre-allocated buffer.

@testset "Sparse-AD plan" begin
    Random.seed!(20260513)

    @testset "Non-linear edge forces sparse-AD path" begin
        # `v[i]` depends nonlinearly on `u[i]` → `extract_linear_map`
        # rejects, falling through to the sparse-AD path.
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
        base = lgm.latent_prior isa CachedSparseADLatentModel ?
            lgm.latent_prior : lgm.latent_prior.base_model
        @test base isa CachedSparseADLatentModel

        # At x = 0 the prior log-density is
        #   logp = -0.5(||u||² + ||v - u²||²) (+ const)
        # ∇²logp w.r.t. (u; v) at 0 = -I(4), so Q = I(4) and μ = 0.
        σ_val = 0.8
        μ_new = mean(base; σ = σ_val)
        Q_new = GMRFs.precision_matrix(base; σ = σ_val)
        @test μ_new ≈ zeros(4) atol = 1.0e-10
        @test Matrix(Q_new) ≈ Matrix(I, 4, 4) atol = 1.0e-10
    end

    @testset "Sparse-AD path under outer AD over hp (known limitation)" begin
        # The sparse-AD backend uses SparseConnectivityTracer for sparsity
        # detection. Under nested ForwardDiff over hp, SCT's tracer hits
        # method ambiguities with ForwardDiff.Dual. This is a pre-existing
        # limitation independent of the assembly-plan refactor; users with
        # outer-AD-incompatible kernel libraries pass
        # `diff_strategy = FiniteDiffStrategy()` to avoid this code path.
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
        base = lgm.latent_prior isa CachedSparseADLatentModel ?
            lgm.latent_prior : lgm.latent_prior.base_model
        @test base isa CachedSparseADLatentModel

        # Float64 call works.
        Q_primal = GMRFs.precision_matrix(base; τ = 2.0, σ = 0.7)
        @test eltype(Q_primal) === Float64

        # Dual hp throws (SCT/ForwardDiff method ambiguity). Track as
        # @test_throws so we notice if upstream ever fixes this.
        τ_dual = ForwardDiff.Dual{:tag}(2.0, 1.0)
        @test_throws Exception GMRFs.precision_matrix(base; τ = τ_dual, σ = 0.7)
    end
end
