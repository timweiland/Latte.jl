using Test
using Latte
using Latte: vbc_correction, _is_vbc_correctable, _vbc_predictor_moments, _marginalize_impl, reported_moments, latent_index_set_for_vbc
using GaussianMarkovRandomFields
using GaussianMarkovRandomFields: PoissonObservations
using Distributions
using LinearAlgebra
using SparseArrays
using OrderedCollections: OrderedDict
using Random

# A minimal layout-bearing model double for the index-set policy (the resolver is
# duck-typed on `latent_groups`).
struct _VBCMockModel
    groups::OrderedDict{Symbol, UnitRange{Int}}
end
Latte.latent_groups(m::_VBCMockModel) = m.groups

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

    @testset "single-hub (p=1) correction is well-formed" begin
        # p=1 nearly tripped a scalarization bug (reduce(hcat) collapses to a
        # vector). M must stay m×1 so Hλ is a 1×1 matrix.
        lgm = _build_lgm(_poisson_build(); seed = 7)
        μ_star, λ = vbc_correction(lgm.ga, lgm.obs_lik, lgm.prior_gmrf, [1])
        @test length(λ) == 1
        @test all(isfinite, μ_star)
        ref = _poisson_reference(lgm, [1])
        λ_ref = _poisson_g_min(ref, lgm.A, lgm.y_raw; iters = 0)
        @test λ ≈ λ_ref atol = 1.0e-7
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

    @testset "constrained GA: μ* preserves the sum-to-zero constraint" begin
        # The Woodbury-corrected conditional_column keeps M's columns in the
        # constraint tangent, so μ* = μ0 + Mλ stays sum-to-zero. (The lsc form
        # would silently drop this on a constrained WorkspaceGMRF.)
        Random.seed!(31)
        m = 6
        n = 12
        Q = spdiagm(0 => fill(2.0, m), -1 => fill(-0.8, m - 1), 1 => fill(-0.8, m - 1))
        A_c = ones(1, m)
        prior_gmrf = ConstrainedGMRF(GMRF(zeros(m), Q), A_c, [0.0])
        A = _design(n, m)
        x_true = rand(prior_gmrf)
        counts = [rand(Poisson(exp(clamp((A * x_true)[i], -3.0, 3.0)))) for i in 1:n]
        om = LinearlyTransformedObservationModel(ExponentialFamily(Poisson), A)
        obs_lik = om(PoissonObservations(counts))
        ga = gaussian_approximation(prior_gmrf, obs_lik)

        μ0 = collect(mean(ga))
        @test abs(sum(μ0)) < 1.0e-6                  # GA mode respects the constraint
        μ_star, _ = vbc_correction(ga, obs_lik, prior_gmrf, [1, 2, 3])
        @test abs(sum(μ_star)) < 1.0e-6              # μ* still respects it
        @test norm(μ_star - μ0) > 1.0e-4             # and it actually moved
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

@testset "VBC dispatch & index resolution (Phase 2)" begin

    @testset "public marginalize threads mean_override to VBC" begin
        lgm = _build_lgm(_poisson_build(); seed = 23)
        I = [1, 2, 3]
        idx = [1, 2, 5]
        μ_star, _ = vbc_correction(lgm.ga, lgm.obs_lik, lgm.prior_gmrf, I)
        σ = std(lgm.ga)

        res = marginalize(
            lgm.ga, lgm.obs_lik, 0.0, VBCMarginal(I), idx;
            prior_gmrf = lgm.prior_gmrf, mean_override = μ_star,
        )
        for (k, i) in enumerate(idx)
            @test mean(res.marginals[k]) ≈ μ_star[i] atol = 1.0e-12
            @test std(res.marginals[k]) ≈ σ[i] atol = 1.0e-12
        end

        # mean_override is used verbatim (bypasses recomputation)
        bogus = collect(mean(lgm.ga)) .+ 1.0
        res2 = marginalize(
            lgm.ga, lgm.obs_lik, 0.0, VBCMarginal(I), idx;
            prior_gmrf = lgm.prior_gmrf, mean_override = bogus,
        )
        for (k, i) in enumerate(idx)
            @test mean(res2.marginals[k]) ≈ bogus[i] atol = 1.0e-12
        end
    end

    @testset "mean_override is ignored by non-VBC methods (regression)" begin
        lgm = _build_lgm(_poisson_build(); seed = 29)
        idx = [1, 2, 3]
        bogus = collect(mean(lgm.ga)) .+ 5.0

        a = marginalize(lgm.ga, lgm.obs_lik, 0.0, SimplifiedLaplace(), idx; prior_gmrf = lgm.prior_gmrf)
        b = marginalize(
            lgm.ga, lgm.obs_lik, 0.0, SimplifiedLaplace(), idx;
            prior_gmrf = lgm.prior_gmrf, mean_override = bogus,
        )
        for k in eachindex(idx)
            @test mean(a.marginals[k]) ≈ mean(b.marginals[k]) atol = 1.0e-12
            @test std(a.marginals[k]) ≈ std(b.marginals[k]) atol = 1.0e-12
        end

        g1 = marginalize(lgm.ga, lgm.obs_lik, 0.0, GaussianMarginal(), idx)
        g2 = marginalize(lgm.ga, lgm.obs_lik, 0.0, GaussianMarginal(), idx; mean_override = bogus)
        for k in eachindex(idx)
            @test mean(g1.marginals[k]) ≈ mean(g2.marginals[k]) atol = 1.0e-12
        end
    end

    @testset "latent_index_set_for_vbc — AutoVBCIndexSet policy" begin
        groups = OrderedDict(:intercept => 1:1, :β => 2:3, :u => 4:10, :field => 11:410)
        m = _VBCMockModel(groups)
        # short_dim=8: intercept(1), β(2), u(7) kept; field(400) excluded
        @test latent_index_set_for_vbc(m, AutoVBCIndexSet(short_dim = 8)) == collect(1:10)
        # widening short_dim pulls in the big field block too
        @test latent_index_set_for_vbc(m, AutoVBCIndexSet(short_dim = 500)) == collect(1:410)
        # explicit vector passes through unchanged (order preserved)
        @test latent_index_set_for_vbc(m, [5, 2, 8]) == [5, 2, 8]
    end

    @testset "latent_index_set_for_vbc — error paths" begin
        empty_m = _VBCMockModel(OrderedDict{Symbol, UnitRange{Int}}())
        @test_throws ArgumentError latent_index_set_for_vbc(empty_m, AutoVBCIndexSet())
        all_big = _VBCMockModel(OrderedDict(:f1 => 1:20, :f2 => 21:50))
        @test_throws ArgumentError latent_index_set_for_vbc(all_big, AutoVBCIndexSet(short_dim = 8))
    end
end

@testset "VBC end-to-end through inla (Phase 3)" begin
    # A compact Poisson LTM model run through the full inla pipeline: the per-θ
    # hook computes μ* at each grid point and threads it into the marginals.
    Random.seed!(101)
    m = 6
    n = 18
    spec = @hyperparams begin
        (τ ~ Gamma(2, 1), transform = log, space = natural)
    end
    # Dense SPD prior so its pattern ⊇ AᵀA (the compact path's precondition for
    # linear_predictor_marginals / selected inverse; A here couples 6↔1).
    Qbase = let Qr = Matrix(2.0I, m, m)
        for i in 1:m, j in (i + 1):m
            Qr[i, j] = Qr[j, i] = -0.1
        end
        sparse(Qr)
    end
    latent_func = (; τ, kwargs...) -> (zeros(m), τ .* Qbase)
    A = _design(n, m)
    obs_model = LinearlyTransformedObservationModel(ExponentialFamily(Poisson), A)
    model = LatentGaussianModel(spec, FunctionLatentModel(latent_func, m), obs_model; augment_latent = false)

    x_true = randn(m) .* 0.7
    y = [rand(Poisson(exp(clamp((A * x_true)[i], -2.0, 3.0)))) for i in 1:n]

    res_g = inla(model, y; progress = false, latent_marginalization_method = GaussianMarginal())
    res_vbc = inla(model, y; progress = false, latent_marginalization_method = VBCMarginal([1, 2, 3, 4, 5, 6]))

    mg = [mean(res_g.latent_marginals[i]) for i in 1:m]
    mv = [mean(res_vbc.latent_marginals[i]) for i in 1:m]
    @test all(isfinite, mv)
    @test all(std(res_vbc.latent_marginals[i]) > 0 for i in 1:m)
    # VBC shifts the posterior mean off the Gaussian (GA-mode) marginal
    @test norm(mv - mg) > 1.0e-3

    # Phase 4: linear_combinations uses the corrected mean μ*. A unit-vector
    # lincomb must reproduce that latent's (corrected) marginal mean, and the
    # VBC lincomb must differ from the Gaussian-mode lincomb.
    e3 = [i == 3 ? 1.0 : 0.0 for i in 1:m]
    lc_vbc = linear_combinations(res_vbc, e3)
    lc_g = linear_combinations(res_g, e3)
    @test mean(lc_vbc) ≈ mean(res_vbc.latent_marginals[3]) atol = 1.0e-5
    @test mean(lc_g) ≈ mean(res_g.latent_marginals[3]) atol = 1.0e-5
    @test abs(mean(lc_vbc) - mean(lc_g)) > 1.0e-4

    # Phase 4: posterior samples are centered at μ* (consistent with the
    # corrected marginals), not the GA mode.
    Random.seed!(202)
    samp = rand(res_vbc, 6000)
    xbar = [sum(@view samp.x[:, i]) / size(samp.x, 1) for i in 1:m]
    @test xbar ≈ mv atol = 0.04

    # Phase 4: compact observation_marginals computed on demand (lazy), with the
    # VBC-corrected predictor mean propagating to fitted values.
    om_vbc = observation_marginals(res_vbc)
    om_g = observation_marginals(res_g)
    @test length(om_vbc) == n
    @test all(isfinite(mean(o)) for o in om_vbc)
    @test all(mean(o) > 0 for o in om_vbc)          # Poisson fitted rates are positive
    @test norm([mean(o) for o in om_vbc] - [mean(o) for o in om_g]) > 1.0e-4
end
