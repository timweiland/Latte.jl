using Test
using Latte
using Latte: vbc_correction, _is_vbc_correctable, _vbc_predictor_moments, _marginalize_impl, reported_moments
using GaussianMarkovRandomFields
using GaussianMarkovRandomFields: PoissonObservations
using Distributions
using LinearAlgebra
using SparseArrays
using Random

# ── helpers ──────────────────────────────────────────────────────────────────

# n×m design where each obs loads on two adjacent latents (gives M off-diagonal
# structure so the correction is non-trivial).
function _design(n, m)
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    for i in 1:n
        j = ((i - 1) % m) + 1
        k = (j % m) + 1
        push!(rows, i, i)
        push!(cols, j, k)
        push!(vals, 1.0, 0.5)
    end
    return sparse(rows, cols, vals, n, m)
end

# A small compact LGM: latent ψ (m-dim, correlated GMRF prior), η = A·ψ, and a
# per-obs likelihood built by `build_obs(A, η_true) -> (obs_lik, y_raw)`. Returns
# the materialized obs likelihood + the GA (exactly the production types) and the
# raw response vector for the reference computations.
function _build_lgm(build_obs; m = 6, n = 10, seed = 1, ρ = 0.8)
    Random.seed!(seed)
    Q = spdiagm(0 => fill(2.0, m), -1 => fill(-ρ, m - 1), 1 => fill(-ρ, m - 1))
    prior_gmrf = GMRF(zeros(m), Q)
    A = _design(n, m)
    x_true = rand(prior_gmrf)
    obs_lik, y_raw = build_obs(A, A * x_true)
    ga = gaussian_approximation(prior_gmrf, obs_lik)
    return (; prior_gmrf, A, obs_lik, ga, y_raw, m, n)
end

function _poisson_build(offset = 0.0)
    return (A, η) -> begin
        counts = [rand(Poisson(exp(clamp(η[i] + offset, -3.0, 6.0)))) for i in eachindex(η)]
        om = LinearlyTransformedObservationModel(ExponentialFamily(Poisson), A)
        return (om(PoissonObservations(counts)), Float64.(counts))
    end
end

function _normal_build(σ = 0.3)
    return (A, η) -> begin
        y = η .+ σ .* randn(length(η))
        om = LinearlyTransformedObservationModel(ExponentialFamily(Normal), A)
        return (om(y; σ = σ), y)
    end
end

function _bernoulli_build()
    return (A, η) -> begin
        bits = [rand() < 1 / (1 + exp(-η[i])) ? 1 : 0 for i in eachindex(η)]
        om = LinearlyTransformedObservationModel(ExponentialFamily(Bernoulli), A)
        return (om(bits), Float64.(bits))
    end
end

# Independent (kernel-free) reconstruction of the GA covariance for a Poisson
# log-link model: Q_X = Q_π + Aᵀ diag(e^{η₀}) A, the exact GA Hessian at the
# mode. Uses NONE of the kernel's primitives (conditional_column /
# linear_predictor_marginals), so agreement validates the whole plumbing.
function _poisson_reference(lgm, I)
    ψ0 = collect(mean(lgm.ga))
    η0 = lgm.A * ψ0
    D = exp.(η0)                                   # Poisson working weights at the mode
    Qπ = sparse(GaussianMarkovRandomFields.precision_matrix(lgm.prior_gmrf))
    Q_X = Matrix(Qπ) + Matrix(lgm.A)' * Diagonal(D) * Matrix(lgm.A)
    Σ_X = inv(Q_X)
    M = Σ_X[:, I]
    S = sqrt.(diag(lgm.A * Σ_X * lgm.A'))
    return (; ψ0, η0, Qπ, Σ_X, M, S)
end

# Newton iterate on the exact (closed-form) Poisson VBC objective g(λ) in the
# reduced p-dim coordinate λ. `iters=0` → a single Newton step (= the kernel).
function _poisson_g_min(ref, A, y; iters = 0)
    M_A = A * ref.M
    QπM = ref.Qπ * ref.M
    λ = zeros(size(ref.M, 2))
    for _ in 0:iters
        ψ1 = ref.ψ0 + ref.M * λ
        η1 = A * ψ1
        Eexp = exp.(η1 .+ ref.S .^ 2 ./ 2)
        grad = M_A' * (Eexp .- y) .+ QπM' * ψ1
        H = M_A' * Diagonal(Eexp) * M_A .+ ref.M' * QπM
        λ = λ .- (Symmetric(Matrix(H)) \ Vector(grad))
    end
    return λ
end

_poisson_g(ref, A, y, λ) = let ψ1 = ref.ψ0 + ref.M * λ, η1 = A * ψ1
    sum(exp.(η1 .+ ref.S .^ 2 ./ 2) .- y .* η1) + 0.5 * dot(ψ1, ref.Qπ * ψ1)
end

# ── tests ────────────────────────────────────────────────────────────────────

@testset "VBC kernel (Phase 1)" begin

    @testset "Poisson μ* matches first-principles Newton step" begin
        lgm = _build_lgm(_poisson_build(); seed = 7)
        I = [1, 2, 3]
        μ_star, λ = vbc_correction(lgm.ga, lgm.obs_lik, lgm.prior_gmrf, I)

        ref = _poisson_reference(lgm, I)
        λ_ref = _poisson_g_min(ref, lgm.A, lgm.y_raw; iters = 0)       # single Newton step
        μ_ref = ref.ψ0 + ref.M * λ_ref

        @test length(μ_star) == lgm.m
        @test λ ≈ λ_ref atol = 1.0e-7
        @test μ_star ≈ μ_ref atol = 1.0e-7
        # the GA's own precision agrees with the first-principles reconstruction
        @test Matrix(GaussianMarkovRandomFields.precision_matrix(lgm.ga)) ≈
            inv(ref.Σ_X) atol = 1.0e-5
        # the correction is genuinely nonzero (Poisson is skewed)
        @test norm(μ_star - ref.ψ0) > 1.0e-3
    end

    @testset "Poisson μ* is the Newton step of the convex objective g" begin
        lgm = _build_lgm(_poisson_build(); seed = 11)
        I = [1, 2, 3, 4]
        _, λ = vbc_correction(lgm.ga, lgm.obs_lik, lgm.prior_gmrf, I)

        ref = _poisson_reference(lgm, I)
        λ_conv = _poisson_g_min(ref, lgm.A, lgm.y_raw; iters = 15)

        g0 = _poisson_g(ref, lgm.A, lgm.y_raw, zeros(length(I)))
        gk = _poisson_g(ref, lgm.A, lgm.y_raw, λ)
        gc = _poisson_g(ref, lgm.A, lgm.y_raw, λ_conv)

        @test gk < g0                                  # VBC decreases the objective
        @test gc <= gk + 1.0e-9                         # converged is the minimum
        # one step lands close to the converged minimizer (it IS the 1st Newton step)
        @test norm(ref.M * (λ - λ_conv)) < 0.3 * norm(ref.M * λ_conv)
    end

    @testset "higher data precision → smaller correction" begin
        # offset shifts the Poisson rate up ⇒ much larger counts ⇒ a tighter GA
        # (smaller selected-inverse) ⇒ a smaller mean correction.
        low = _build_lgm(_poisson_build(0.0); seed = 3, ρ = 0.5)
        high = _build_lgm(_poisson_build(4.0); seed = 3, ρ = 0.5)
        I = [1, 2, 3]
        μ_low, _ = vbc_correction(low.ga, low.obs_lik, low.prior_gmrf, I)
        μ_high, _ = vbc_correction(high.ga, high.obs_lik, high.prior_gmrf, I)
        @test norm(μ_high - collect(mean(high.ga))) < norm(μ_low - collect(mean(low.ga)))
    end

    @testset "Gaussian likelihood → exact no-op" begin
        lgm = _build_lgm(_normal_build(0.3); seed = 5)
        @test _is_vbc_correctable(lgm.obs_lik) == false
        μ_star, λ = vbc_correction(lgm.ga, lgm.obs_lik, lgm.prior_gmrf, [1, 2, 3])
        @test μ_star ≈ collect(mean(lgm.ga)) atol = 1.0e-12
        @test all(iszero, λ)
    end

    @testset "Bernoulli via Gauss–Hermite → finite, convergent in n_gh" begin
        lgm = _build_lgm(_bernoulli_build(); seed = 9)
        @test _is_vbc_correctable(lgm.obs_lik) == true
        μ7, λ7 = vbc_correction(lgm.ga, lgm.obs_lik, lgm.prior_gmrf, [1, 2, 3]; n_gh = 7)
        @test all(isfinite, μ7)
        @test length(μ7) == lgm.m
        # quadrature self-consistency: 7 vs 21 nodes give essentially the same λ
        _, λ21 = vbc_correction(lgm.ga, lgm.obs_lik, lgm.prior_gmrf, [1, 2, 3]; n_gh = 21)
        @test λ7 ≈ λ21 atol = 1.0e-4
    end

    @testset "mean_override path: corrected mean, variance untouched" begin
        lgm = _build_lgm(_poisson_build(); seed = 13)
        I = [1, 2, 3]
        idx = [1, 2, 4, 6]
        μ_star, _ = vbc_correction(lgm.ga, lgm.obs_lik, lgm.prior_gmrf, I)
        σ = std(lgm.ga)

        m_over = _marginalize_impl(
            lgm.ga, lgm.obs_lik, 0.0, VBCMarginal(I), idx, lgm.prior_gmrf;
            mean_override = μ_star,
        )
        @test all(m isa Normal for m in m_over)
        for (k, i) in enumerate(idx)
            @test mean(m_over[k]) ≈ μ_star[i] atol = 1.0e-12
            @test std(m_over[k]) ≈ σ[i] atol = 1.0e-12     # variance is the real GA's
        end

        # internal path (no override) reproduces the same means
        m_int = _marginalize_impl(
            lgm.ga, lgm.obs_lik, 0.0, VBCMarginal(I), idx, lgm.prior_gmrf,
        )
        for k in eachindex(idx)
            @test mean(m_int[k]) ≈ mean(m_over[k]) atol = 1.0e-9
            @test std(m_int[k]) ≈ std(m_over[k]) atol = 1.0e-12
        end
    end

    @testset "public marginalize returns corrected Normal marginals" begin
        lgm = _build_lgm(_poisson_build(); seed = 17)
        I = [1, 2, 3]
        idx = [1, 3, 5]
        res = marginalize(lgm.ga, lgm.obs_lik, 0.0, VBCMarginal(I), idx; prior_gmrf = lgm.prior_gmrf)
        μ_star, _ = vbc_correction(lgm.ga, lgm.obs_lik, lgm.prior_gmrf, I)
        σ = std(lgm.ga)
        @test all(m isa Normal for m in res.marginals)
        for (k, i) in enumerate(idx)
            @test mean(res.marginals[k]) ≈ μ_star[i] atol = 1.0e-9
            @test std(res.marginals[k]) ≈ σ[i] atol = 1.0e-12
        end
    end

    @testset "augmented model is rejected" begin
        lgm = _build_lgm(_poisson_build(); seed = 19)
        @test_throws ArgumentError _marginalize_impl(
            lgm.ga, lgm.obs_lik, 0.0, VBCMarginal([1, 2]), [1, 2], lgm.prior_gmrf;
            augmentation_info = (linear_predictor_indices = 1:2,),
        )
    end

    @testset "reported_moments reduces to the standardized mean shift" begin
        # VBC keeps σ_baseline and reports the marginal's (corrected) mean.
        μ_marg, σ_rep = reported_moments(VBCMarginal([1]), 1.5, 2.0, Normal(3.0, 7.0))
        @test μ_marg ≈ 3.0
        @test σ_rep ≈ 2.0
    end
end
