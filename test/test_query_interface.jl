using Test
using Latte
using GaussianMarkovRandomFields
using LinearAlgebra
using SparseArrays
using Distributions
using Random

# Contract tests for Latte's pluggable posterior-query interface (Phase 3).
#
# A non-GMRF inference backend supplies a posterior `q` by subtyping
# `Distributions.AbstractMvNormal` (for mean/var/std/logpdf/logdetcov/rand) and
# implementing exactly these three covariance generics:
#   selected_covariance(q)        -- selected inverse on the factor pattern
#   conditional_column(q, i)      -- full covariance column Q⁻¹[:, i]
#   lincomb_variance(q, a)        -- aᵀ Q⁻¹ a
# The default methods back the sparse-GMRF case. Here we pin their numerical
# contract against ground-truth dense linear algebra on a small unconstrained
# posterior built the way the engine builds it (workspace prior → GA).

@testset "Posterior query interface" begin
    Random.seed!(1234)
    n = 8

    m = IIDModel(n)
    τ = 1.5
    ws = make_workspace(m; τ = τ)
    prior = m(ws; τ = τ)                       # WorkspaceGMRF prior at θ
    y = randn(n)
    obs_lik = ExponentialFamily(Normal)(y; σ = 0.7)
    q = gaussian_approximation(prior, obs_lik) # WorkspaceGMRF posterior (unconstrained)

    # The query-only Gaussian contract IS Distributions.AbstractMvNormal.
    @test q isa Distributions.AbstractMvNormal

    Σ = inv(Matrix(precision_matrix(q)))       # ground-truth posterior covariance

    @testset "selected_covariance" begin
        S = selected_covariance(q)
        @test size(S) == (n, n)
        # Diagonal == marginal variances == var(q) == diag(Q⁻¹).
        @test diag(S) ≈ diag(Σ) rtol = 1.0e-8
        @test diag(S) ≈ var(q) rtol = 1.0e-8
        # Transitional alias preserves the value.
        @test Matrix(selinv_mat(q)) ≈ Matrix(S) rtol = 1.0e-12
    end

    @testset "conditional_column" begin
        for i in (1, 4, n)
            col = conditional_column(q, i)
            @test col ≈ Σ[:, i] rtol = 1.0e-7
        end
    end

    @testset "lincomb_variance" begin
        for _ in 1:3
            a = randn(n)
            @test lincomb_variance(q, a) ≈ dot(a, Σ * a) rtol = 1.0e-7
        end
    end
end
