using Test
using Latte
using Latte: selected_covariance, selinv_mat
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

# ── Rung 0 conformance: a plain Distributions.MvNormal as a non-GMRF backend ──
# Validates tasks/validate-pluggable-interface-nongmrf-backend.org: a
# precision-free dense Gaussian satisfies the engine's query contract, and the
# marginalization *consumers* drive it through ONLY that contract (no precision
# or workspace reach-through). No bespoke type — MvNormal is already
# <: AbstractMvNormal and gives mean/var/std/cov/logpdf/logdetcov/rand; the three
# Latte generics come from the dense AbstractMvNormal fallbacks.

# Non-quadratic per-coordinate AD likelihood (Poisson-like; h''' ≠ 0) so the SLA
# fallback branch does real work.
struct _ADPois{T <: Real} <: ContinuousUnivariateDistribution
    η::T
end
Distributions.logpdf(d::_ADPois, z::Real) = z * d.η - exp(d.η)
Distributions.minimum(::_ADPois) = -Inf
Distributions.maximum(::_ADPois) = Inf
Distributions.insupport(::_ADPois, ::Real) = true

@testset "Non-GMRF backend (MvNormal) conformance" begin
    Random.seed!(99)
    n = 6
    M = randn(n, n)
    Σ = Symmetric(M * M' + n * I)          # dense SPD covariance, no precision rep
    μ = randn(n)
    q = MvNormal(μ, Σ)                       # <: AbstractMvNormal, NOT a GMRF
    Σm = Matrix(Σ)

    @test q isa Distributions.AbstractMvNormal
    @test !(q isa GaussianMarkovRandomFields.AbstractGMRF)

    @testset "query surface" begin
        @test mean(q) == μ
        @test var(q) ≈ diag(Σm)
        @test std(q) ≈ sqrt.(diag(Σm))
        @test Distributions.logdetcov(q) ≈ logdet(Σm)
        @test Matrix(selected_covariance(q)) ≈ Σm
        for i in (1, 3, n)
            @test conditional_column(q, i) ≈ Σm[:, i]
        end
        b = randn(n)
        @test lincomb_variance(q, b) ≈ dot(b, Σm * b)
    end

    @testset "marginalization consumers drive MvNormal through the contract" begin
        idx = collect(1:n)
        y = randn(n)
        gauss_obs = ExponentialFamily(Normal)(y; σ = 1.0)

        # GaussianMarginal: exact — per-index Normal(μ_i, σ_i).
        gm = marginalize(q, gauss_obs, 0.0, GaussianMarginal(), idx)
        @test length(gm.marginals) == n
        σq = std(q)
        for j in 1:n
            @test mean(gm.marginals[j]) ≈ μ[j]
            @test std(gm.marginals[j]) ≈ σq[j]
        end

        # SimplifiedLaplace, diagonal-likelihood fast path (Normal ⇒ var + column).
        sla_g = marginalize(q, gauss_obs, 0.0, SimplifiedLaplace(), idx; augmentation_info = nothing)
        @test length(sla_g.marginals) == n
        @test all(m -> m isa SkewNormal && isfinite(mean(m)) && isfinite(std(m)), sla_g.marginals)

        # SimplifiedLaplace, non-diagonal AD fallback (⇒ _compute_tr over selected_covariance).
        yc = [1.0, 0.0, 2.0, 1.0, 0.0, 3.0]
        ll = (x; kwargs...) -> sum(logpdf(_ADPois(x[i]), yc[i]) for i in eachindex(yc))
        pw = (x; kwargs...) -> [logpdf(_ADPois(x[i]), yc[i]) for i in eachindex(yc)]
        ad_obs = GaussianMarkovRandomFields.AutoDiffObservationModel(
            ll; n_latent = n, pointwise_loglik_func = pw,
        )(yc)
        sla_ad = marginalize(q, ad_obs, 0.0, SimplifiedLaplace(), idx; augmentation_info = nothing)
        @test length(sla_ad.marginals) == n
        @test all(m -> m isa SkewNormal && isfinite(mean(m)) && isfinite(std(m)), sla_ad.marginals)
    end
end
