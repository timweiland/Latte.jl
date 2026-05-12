using Test
using Latte
using Latte: accumulate!, finalize!, _waic_pointwise_integrals
using Random
using GaussianMarkovRandomFields
using Distributions
using SparseArrays
using LinearAlgebra
using Statistics

include("test_helpers.jl")

@testset "WAICAccumulator" begin

    @testset "Integrated predictive loglik: Normal analytic" begin
        # For Normal + IdentityLink:
        # integrated_ll: log N(y; μ, √(σ²_obs + v))
        # expected_log_ll: -½log(2π) - log(σ) - [(y-μ)² + v]/(2σ²)
        n = 5
        μ = randn(n)
        v = rand(n) .+ 0.1  # marginal variances
        Q = Diagonal(1.0 ./ v)
        ga = GMRF(μ, Q)

        σ_obs = 1.5
        y = randn(n)
        obs_lik = ExponentialFamily(Normal)(y; σ = σ_obs)

        integrated_ll, expected_log_ll = _waic_pointwise_integrals(ga, obs_lik)

        # Compare integrated_ll against analytic formula
        expected_int = [logpdf(Normal(μ[i], sqrt(σ_obs^2 + v[i])), y[i]) for i in 1:n]
        @test integrated_ll ≈ expected_int

        # Compare expected_log_ll against analytic formula
        expected_ell = [
            -0.5 * log(2π) - log(σ_obs) - ((y[i] - μ[i])^2 + v[i]) / (2 * σ_obs^2)
                for i in 1:n
        ]
        @test expected_log_ll ≈ expected_ell

        # Jensen's inequality: E[log p] ≤ log E[p], so expected_log_ll ≤ integrated_ll
        @test all(expected_log_ll .<= integrated_ll .+ 1.0e-12)
    end

    @testset "Pretty printing" begin
        acc = WAICAccumulator()
        acc.WAIC = 123.45
        acc.p_WAIC = 4.56
        acc.lppd = -59.45

        str = sprint(show, MIME("text/plain"), acc)
        @test occursin("WAIC", str)
        @test occursin("123.45", str)
        @test occursin("4.56", str)
    end

    @testset "Integration: Normal IID model" begin
        n = 20
        spec = @hyperparams begin
            (σ ~ InverseGamma(2, 1), transform = log, space = natural)
        end
        function latent_func_normal(; σ, kwargs...)
            Q = spdiagm(0 => fill(1 / σ^2, n))
            return (zeros(n), Q)
        end
        obs_model = ExponentialFamily(Normal)
        model = LatentGaussianModel(spec, FunctionLatentModel(latent_func_normal, n), obs_model)

        y = randn(n)

        result = inla(
            model, y;
            progress = false,
            accumulators = (DICStrategy(), MarginalLogLikelihoodStrategy(), WAICStrategy()),
        )
        waic_acc = result.accumulators[3]

        # WAIC should be finite
        @test isfinite(waic_acc.WAIC)
        @test isfinite(waic_acc.lppd)
        @test isfinite(waic_acc.p_WAIC)

        # p_WAIC should be non-negative and meaningfully > 0
        @test waic_acc.p_WAIC >= 0
        # With p_WAIC1 formula, this should capture latent uncertainty (several effective params)
        @test waic_acc.p_WAIC > 1.0

        # lppd should be negative (log of probabilities)
        @test waic_acc.lppd < 0
    end

    @testset "Integration: Poisson model" begin
        n = 15
        spec = @hyperparams begin
            (τ ~ Gamma(2, 1), transform = log, space = natural)
        end
        function latent_func_poisson(; τ, kwargs...)
            Q = spdiagm(0 => fill(τ, n))
            return (zeros(n), Q)
        end
        obs_model = ExponentialFamily(Poisson)
        model = LatentGaussianModel(spec, FunctionLatentModel(latent_func_poisson, n), obs_model)

        y = rand(Poisson(3.0), n)

        result = inla(
            model, y;
            progress = false,
            accumulators = (DICStrategy(), MarginalLogLikelihoodStrategy(), WAICStrategy()),
        )
        waic_acc = result.accumulators[3]

        @test isfinite(waic_acc.WAIC)
        @test waic_acc.p_WAIC >= 0
        @test waic_acc.lppd < 0
    end

    @testset "n_nodes parameter" begin
        acc = WAICAccumulator(; n_nodes = 25)
        @test acc.n_nodes == 25
    end

    @testset "Multi-coefficient LTL: η = A·x with row sparsity > 1" begin
        # Regression test for the bug where element-wise quadrature over
        # the latent silently used the wrong marginal for obs that depend
        # on linear combinations of latent entries. Pre-fix:
        # `(μ, σ) = (mean(ga), std(ga))` → effective variance per row was
        # the rank-1 surrogate `(Σ_k a_ik σ_k)²`, not `(A Σ A')_ii`.
        # Post-fix: `linear_predictor_marginals` returns the correct η
        # marginals and the closed-form Normal-IdentityLink formula
        # collapses to `log N(y_i; (A μ)_i, σ² + (A Σ A')_ii)`.
        Random.seed!(42)
        n_latent = 6
        # Multi-coefficient A: each row has 3-4 nonzeros, rows differ.
        n_obs = 4
        A = sparse(
            [
                1.0 0.5 0.0 0.2 0.0 0.0
                0.0 0.7 -0.3 0.0 0.4 0.0
                0.1 0.0 0.5 0.6 0.0 -0.2
                0.0 0.0 0.4 -0.1 0.3 0.5
            ]
        )
        # Build a fully dense SPD Q so selected-inversion returns every
        # `Σ_{jk}` entry, satisfying `linear_predictor_marginals`'s
        # documented `Q ⊇ A'A` precondition without having to plumb a
        # specific sparsity union (the test's job is the accumulator
        # math, not pattern bookkeeping).
        Q_raw = Matrix(1.5 * I, n_latent, n_latent)
        for i in 1:n_latent, j in (i + 1):n_latent
            Q_raw[i, j] = Q_raw[j, i] = -0.05
        end
        Q = sparse(Q_raw)
        μ_x = randn(n_latent)
        ga = GMRF(μ_x, Q)
        Σ_x = inv(Matrix(Q))

        σ_obs = 0.4
        y = randn(n_obs)
        base = ExponentialFamily(Normal)
        obs_model = LinearlyTransformedObservationModel(base, A)
        obs_lik = obs_model(y; σ = σ_obs)

        # The fix in action — should hit the analytic Normal-IdentityLink path
        # via the stripped η-likelihood.
        integrated_ll, expected_log_ll = _waic_pointwise_integrals(ga, obs_lik)

        # Closed-form reference: y ~ N(A μ, σ² I + A Σ A')
        μ_η_ref = A * μ_x
        v_η_ref = diag(A * Σ_x * A')
        expected_int = [
            logpdf(Normal(μ_η_ref[i], sqrt(σ_obs^2 + v_η_ref[i])), y[i])
                for i in 1:n_obs
        ]
        @test integrated_ll ≈ expected_int atol = 1.0e-2

        # E_η[log N(y; η, σ)] = -½log(2π) - log σ - [(y - μ_η)² + v_η] / (2σ²)
        expected_ell = [
            -0.5 * log(2π) - log(σ_obs) -
                ((y[i] - μ_η_ref[i])^2 + v_η_ref[i]) / (2 * σ_obs^2)
                for i in 1:n_obs
        ]
        @test expected_log_ll ≈ expected_ell atol = 1.0e-2

        # Sanity: the answer differs from what the pre-fix rank-1 surrogate
        # would have produced. Surrogate variance per row was the diagonal
        # of the *element-wise* perturbation: `(A · σ_x · I)`'s row norm
        # squared isn't even meaningful as a marginal — it just produces
        # different numbers. Confirm we're not coincidentally matching it.
        σ_x = sqrt.(diag(Σ_x))
        rank1_surrogate_v = [sum(A[i, k] * σ_x[k] for k in 1:n_latent)^2 for i in 1:n_obs]
        @test !isapprox(v_η_ref, rank1_surrogate_v; atol = 1.0e-6)
    end

    @testset "Sample-based fallback for unsupported obs likelihoods" begin
        # Stub likelihood (`LikWithoutLPM`, from test_helpers.jl) lacks
        # `linear_predictor_marginals` — only exposes `pointwise_loglik`.
        # The accumulator must fall back to sample-based aggregation. MC
        # estimate should converge to the analytic answer (computed via a
        # parallel direct-Normal path).
        @test !Latte._supports_lpm(LikWithoutLPM(zeros(3), 1.0))

        Random.seed!(2026)
        n = 5
        μ_x = randn(n)
        v = fill(0.25, n)
        Q = Diagonal(1.0 ./ v)
        ga = GMRF(μ_x, Q)
        σ_obs = 0.8
        y = randn(n)

        ef_lik = ExponentialFamily(Normal)(y; σ = σ_obs)
        analytic_int, analytic_ell = _waic_pointwise_integrals(ga, ef_lik)

        fake_lik = LikWithoutLPM(y, σ_obs)
        sampled_int, sampled_ell = _waic_pointwise_integrals(
            ga, fake_lik, WAICStrategy(; fallback = :sample, n_samples = 8192),
        )

        # MC error ~ σ/√n ≈ 0.01–0.05 per obs at 8192 samples for this setup.
        @test sampled_int ≈ analytic_int atol = 5.0e-2
        @test sampled_ell ≈ analytic_ell atol = 5.0e-2

        # :error fallback errors instead of falling back
        @test_throws ArgumentError _waic_pointwise_integrals(
            ga, fake_lik, WAICStrategy(; fallback = :error, n_samples = 64),
        )
    end

    @testset "Mixed composite: Normal-Identity supported + LPM-less" begin
        # Regression: composites with one supported Normal-Identity
        # component (emits `NormalIdentityClosedForm`) plus one
        # LPM-less component (forces MC, emits `PointwiseLogLikSamples`)
        # went through `_mixed_pointwise`. An overly narrow
        # `PointwiseLogLikSamples[]` initialiser silently `convert`-failed
        # on the closed-form records.
        n_phys, n_data = 4, 3
        A_phys = randn(n_phys, 5)
        β_post = randn(5)
        y_phys = A_phys * β_post .+ 0.1 .* randn(n_phys)
        y_data = randn(n_data)

        Q = sparse(Matrix(1.0 * I, 5, 5))
        ga = GMRF(zeros(5), Q)

        phys = LinearlyTransformedObservationModel(ExponentialFamily(Normal), A_phys)(y_phys; σ = 0.1)
        data = LikWithoutLPM(y_data, 0.5)

        composite = GaussianMarkovRandomFields.CompositeLikelihood((phys, data))
        # Smoke test — used to throw `convert(::PointwiseLogLikSamples, ::NormalIdentityClosedForm)`.
        integrated_ll, expected_log_ll = _waic_pointwise_integrals(
            ga, composite, WAICStrategy(; n_samples = 64),
        )
        @test length(integrated_ll) == n_phys + n_data
        @test all(isfinite, integrated_ll)
        @test all(isfinite, expected_log_ll)
    end

    @testset "pointwise_loglik consistency" begin
        # Verify that sum(pointwise_loglik) ≈ loglik for our observation models
        n = 10
        x = randn(n)

        # Normal (σ=1.0 is observation noise std dev)
        normal_lik = ExponentialFamily(Normal)(randn(n); σ = 1.0)
        @test sum(pointwise_loglik(x, normal_lik)) ≈ loglik(x, normal_lik)

        # Poisson (needs PoissonObservations wrapper)
        poisson_lik = ExponentialFamily(Poisson)(GaussianMarkovRandomFields.PoissonObservations(rand(0:10, n)))
        @test sum(pointwise_loglik(x, poisson_lik)) ≈ loglik(x, poisson_lik)
    end
end
