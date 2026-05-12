using Test
using Latte
using Latte: accumulate!, finalize!, _cpo_pointwise_integrals, _pointwise_cdf
using Random
using GaussianMarkovRandomFields
using Distributions
using SparseArrays
using LinearAlgebra
using Statistics

include("test_helpers.jl")

@testset "CPOAccumulator" begin

    @testset "CPO/PIT integrals: Normal analytic" begin
        # For Normal + IdentityLink:
        # log_h_i = 0.5*log(2π*σ²) - 0.5*log(1 - v_i/σ²) + (y_i-μ_i)²/(2σ²*(1-v_i/σ²))
        # PIT_i = Φ((y_i - μ_i) / √(σ² + v_i))
        n = 5
        μ = randn(n)
        v = rand(n) .* 0.5 .+ 0.1  # Keep v < σ² to avoid CPO failure
        Q = Diagonal(1.0 ./ v)
        ga = GMRF(μ, Q)

        σ_obs = 1.5
        y = randn(n)
        obs_lik = ExponentialFamily(Normal)(y; σ = σ_obs)

        log_h, pit, _ = _cpo_pointwise_integrals(ga, obs_lik)

        for i in 1:n
            ratio = v[i] / σ_obs^2
            expected_log_h = 0.5 * log(2π * σ_obs^2) - 0.5 * log(1 - ratio) +
                (y[i] - μ[i])^2 / (2 * σ_obs^2 * (1 - ratio))
            @test log_h[i] ≈ expected_log_h
        end

        for i in 1:n
            expected_pit = cdf(Normal(μ[i], sqrt(σ_obs^2 + v[i])), y[i])
            @test pit[i] ≈ expected_pit
        end

        @test all(0 .<= pit .<= 1)
        @test all(isfinite.(log_h))
    end

    @testset "CPO failure detection: Normal analytic" begin
        # When v_i >= σ², the harmonic mean integral diverges
        n = 3
        σ_obs = 1.0
        v = [0.5, 1.5, 2.0]  # v[2] and v[3] exceed σ²
        μ = zeros(n)
        Q = Diagonal(1.0 ./ v)
        ga = GMRF(μ, Q)
        y = zeros(n)
        obs_lik = ExponentialFamily(Normal)(y; σ = σ_obs)

        log_h, pit, _ = _cpo_pointwise_integrals(ga, obs_lik)

        # v[1] < σ² → should be finite
        @test isfinite(log_h[1])

        # v[2], v[3] >= σ² → should be Inf (CPO failure)
        @test log_h[2] == Inf
        @test log_h[3] == Inf

        # PIT should still be computable for all
        @test all(isfinite.(pit))
    end

    @testset "Pointwise CDF: Normal" begin
        n = 5
        x = randn(n)
        y = randn(n)
        σ_obs = 1.5
        obs_lik = ExponentialFamily(Normal)(y; σ = σ_obs)

        cdf_vals = _pointwise_cdf(x, obs_lik)

        for i in 1:n
            @test cdf_vals[i] ≈ cdf(Normal(x[i], σ_obs), y[i])
        end
        @test all(0 .<= cdf_vals .<= 1)
    end

    @testset "Pointwise CDF: Poisson (midpoint PIT)" begin
        n = 5
        x = randn(n)  # On log scale (LogLink)
        y = rand(0:10, n)
        obs_lik = ExponentialFamily(Poisson)(GaussianMarkovRandomFields.PoissonObservations(y))

        cdf_vals = _pointwise_cdf(x, obs_lik)

        for i in 1:n
            λ = exp(x[i])
            d = Poisson(λ)
            expected = cdf(d, y[i]) - 0.5 * pdf(d, y[i])
            @test cdf_vals[i] ≈ expected
        end
        @test all(0 .<= cdf_vals .<= 1)
    end

    @testset "Pretty printing" begin
        acc = CPOAccumulator()
        acc.CPO = [0.1, 0.2, 0.3]
        acc.log_CPO = log.([0.1, 0.2, 0.3])
        acc.LPML = sum(acc.log_CPO)
        acc.PIT = [0.4, 0.5, 0.6]
        acc.failure = [0.0, 0.0, 0.0]
        acc.n_failures = 0

        str = sprint(show, MIME("text/plain"), acc)
        @test occursin("CPO", str)
        @test occursin("LPML", str)
        @test occursin("PIT", str)

        # With failures
        acc.failure = [0.0, 1.5, 0.0]
        acc.n_failures = 1
        str2 = sprint(show, MIME("text/plain"), acc)
        @test occursin("Unreliable", str2)
        @test occursin("1.5", str2)
    end

    @testset "Constructor parameters" begin
        acc = CPOAccumulator(; n_nodes = 25, compute_pit = false)
        @test acc.n_nodes == 25
        @test acc.compute_pit == false

        acc2 = CPOAccumulator()
        @test acc2.n_nodes == 15
        @test acc2.compute_pit == true
    end

    @testset "Integration: Normal IID model" begin
        n = 20
        spec = @hyperparams begin
            (σ ~ InverseGamma(2, 1), transform = log, space = natural)
        end
        function latent_func_normal_cpo(; σ, kwargs...)
            Q = spdiagm(0 => fill(1 / σ^2, n))
            return (zeros(n), Q)
        end
        obs_model = ExponentialFamily(Normal)
        model = LatentGaussianModel(spec, FunctionLatentModel(latent_func_normal_cpo, n), obs_model)

        y = randn(n)

        result = inla(
            model, y;
            progress = false,
            accumulators = (DICStrategy(), MarginalLogLikelihoodStrategy(), WAICStrategy(), CPOStrategy()),
        )
        cpo_acc = result.accumulators[4]

        # All CPO values should be finite and positive
        @test all(isfinite.(cpo_acc.CPO))
        @test all(cpo_acc.CPO .> 0)

        # LPML should be finite and negative
        @test isfinite(cpo_acc.LPML)
        @test cpo_acc.LPML < 0

        # No failures expected for well-behaved Normal model
        @test all(cpo_acc.failure .== 0.0)
        @test cpo_acc.n_failures == 0

        # PIT values should be in [0, 1]
        @test all(0 .<= cpo_acc.PIT .<= 1)

        # Under a reasonable model, mean PIT should be roughly 0.5
        @test 0.1 < mean(cpo_acc.PIT) < 0.9

        # LPML should be less than or close to lppd
        # (leave-one-out penalizes more than in-sample)
        waic_acc = result.accumulators[3]
        @test cpo_acc.LPML <= waic_acc.lppd + 1.0
    end

    @testset "Integration: Poisson model" begin
        n = 15
        spec = @hyperparams begin
            (τ ~ Gamma(2, 1), transform = log, space = natural)
        end
        function latent_func_poisson_cpo(; τ, kwargs...)
            Q = spdiagm(0 => fill(τ, n))
            return (zeros(n), Q)
        end
        obs_model = ExponentialFamily(Poisson)
        model = LatentGaussianModel(spec, FunctionLatentModel(latent_func_poisson_cpo, n), obs_model)

        y = rand(Poisson(3.0), n)

        result = inla(
            model, y;
            progress = false,
            accumulators = (DICStrategy(), MarginalLogLikelihoodStrategy(), WAICStrategy(), CPOStrategy()),
        )
        cpo_acc = result.accumulators[4]

        @test all(cpo_acc.CPO .>= 0)
        @test all(isfinite.(cpo_acc.log_CPO))
        @test isfinite(cpo_acc.LPML)

        # Failure scores should be computed for all observations
        @test length(cpo_acc.failure) == n

        # PIT should be in [0, 1] (midpoint PIT for discrete)
        @test all(0 .<= cpo_acc.PIT .<= 1)
    end

    @testset "CPO without PIT" begin
        n = 10
        spec = @hyperparams begin
            (σ ~ InverseGamma(2, 1), transform = log, space = natural)
        end
        function latent_func_nopit(; σ, kwargs...)
            Q = spdiagm(0 => fill(1 / σ^2, n))
            return (zeros(n), Q)
        end
        model = LatentGaussianModel(
            spec, FunctionLatentModel(latent_func_nopit, n), ExponentialFamily(Normal)
        )

        result = inla(
            model, randn(n);
            progress = false,
            accumulators = (
                DICStrategy(), MarginalLogLikelihoodStrategy(), WAICStrategy(),
                CPOStrategy(; compute_pit = false),
            ),
        )
        cpo_acc = result.accumulators[4]

        @test all(isfinite.(cpo_acc.CPO))
        @test isempty(cpo_acc.PIT)
    end

    @testset "Multi-coefficient LTL: CPO/PIT use correct η marginal" begin
        # Regression test: pre-fix, the element-wise GH path used
        # `std(ga)` per latent coord (rank-1 surrogate) instead of
        # `sqrt(diag(A Σ A'))` per obs. For Normal-IdentityLink CPO has
        # closed-form `(y - μ_η)² / (2σ²(1-v_η/σ²))` (plus normalisation);
        # PIT has closed-form `Φ((y - μ_η)/√(σ² + v_η))`. Both should use
        # the *correct* η marginal post-fix.
        Random.seed!(43)
        n_latent = 6
        n_obs = 4
        A = sparse(
            [
                1.0 0.5 0.0 0.2 0.0 0.0
                0.0 0.7 -0.3 0.0 0.4 0.0
                0.1 0.0 0.5 0.6 0.0 -0.2
                0.0 0.0 0.4 -0.1 0.3 0.5
            ]
        )
        Q_raw = Matrix(1.5 * I, n_latent, n_latent)
        for i in 1:n_latent, j in (i + 1):n_latent
            Q_raw[i, j] = Q_raw[j, i] = -0.05
        end
        Q = sparse(Q_raw)
        μ_x = randn(n_latent)
        ga = GMRF(μ_x, Q)
        Σ_x = inv(Matrix(Q))

        σ_obs = 1.5  # larger than v_η so CPO doesn't diverge
        y = randn(n_obs)
        obs_lik = LinearlyTransformedObservationModel(ExponentialFamily(Normal), A)(y; σ = σ_obs)

        log_inv_lik, pit, _ = Latte._cpo_pointwise_integrals(
            ga, obs_lik, CPOStrategy(; n_nodes = 15, compute_pit = true),
        )

        μ_η_ref = A * μ_x
        v_η_ref = diag(A * Σ_x * A')

        # Verify v_η_ref is in the regime where CPO is finite
        @test all(v_η_ref .< σ_obs^2)

        expected_log_inv = [
            0.5 * log(2π * σ_obs^2) -
                0.5 * log(1 - v_η_ref[i] / σ_obs^2) +
                (y[i] - μ_η_ref[i])^2 / (2 * σ_obs^2 * (1 - v_η_ref[i] / σ_obs^2))
                for i in 1:n_obs
        ]
        @test log_inv_lik ≈ expected_log_inv atol = 1.0e-2

        expected_pit = [
            cdf(Normal(μ_η_ref[i], sqrt(σ_obs^2 + v_η_ref[i])), y[i])
                for i in 1:n_obs
        ]
        @test pit ≈ expected_pit atol = 1.0e-2
    end

    @testset "Sample-based CPO: PSIS pulls MC toward analytic reference" begin
        # Regression test for the CPO MC-bias pathology: straight MC for
        # `E[1/p(y_i|x)]` is unstable (heavy-tailed inverse weights —
        # rare-but-large `1/p_s` values undersampled with finite n_samples),
        # which systematically *underestimates* the expectation and
        # therefore *overestimates* CPO = 1/E[1/p]. PSIS smooths the
        # upper tail of the log weights via a fitted GPD, pulling the
        # estimate back toward the truth.
        #
        # Note: CPO is the per-obs LOO predictive *density* for continuous
        # likelihoods — it can legitimately exceed 1 (e.g. Normal with
        # small σ). The regression check is agreement with the analytic
        # reference, not a [0, 1] bound.
        # `LikWithoutLPM` (test_helpers.jl) is the shared LPM-less stub.
        @test !Latte._supports_lpm(LikWithoutLPM(zeros(3), 1.0))

        Random.seed!(2027)
        n = 6
        μ_x = randn(n)
        v = fill(0.25, n)
        Q = Diagonal(1.0 ./ v)
        ga = GMRF(μ_x, Q)
        σ_obs = 0.8
        y = randn(n)

        # Analytic reference via direct-Normal obs.
        ef_lik = ExponentialFamily(Normal)(y; σ = σ_obs)
        ref_log_inv, _, _ = _cpo_pointwise_integrals(ga, ef_lik)

        # Sample-based path via the LPM-less stub.
        fake_lik = LikWithoutLPM(y, σ_obs)
        log_inv, _, _ = _cpo_pointwise_integrals(
            ga, fake_lik,
            CPOStrategy(; fallback = :sample, n_samples = 4096, compute_pit = false),
        )

        # PSIS keeps the estimate close to the analytic reference. The
        # bound is loose because tail observations (large ref_log_inv)
        # carry residual MC noise even after smoothing.
        @test log_inv ≈ ref_log_inv atol = 1.0
        @test all(isfinite, log_inv)
    end
end
