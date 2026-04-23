using Test
using Latte
using Latte: accumulate!, finalize!, _cpo_pit_integrals, _pointwise_cdf
using Random
using GaussianMarkovRandomFields
using Distributions
using SparseArrays
using LinearAlgebra
using Statistics

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

        log_h, pit, inner_ess = _cpo_pit_integrals(ga, obs_lik)

        # Verify CPO integral against analytic formula
        for i in 1:n
            ratio = v[i] / σ_obs^2
            expected_log_h = 0.5 * log(2π * σ_obs^2) - 0.5 * log(1 - ratio) +
                (y[i] - μ[i])^2 / (2 * σ_obs^2 * (1 - ratio))
            @test log_h[i] ≈ expected_log_h
        end

        # Verify PIT against analytic formula
        for i in 1:n
            expected_pit = cdf(Normal(μ[i], sqrt(σ_obs^2 + v[i])), y[i])
            @test pit[i] ≈ expected_pit
        end

        # PIT should be in [0, 1]
        @test all(0 .<= pit .<= 1)

        # log_h should be positive (since h = E[1/p] >= 1/E[p] > 0 and typically > 1)
        @test all(isfinite.(log_h))

        # Analytic path has no inner quadrature — ESS = n_nodes (perfect)
        @test all(inner_ess .== 15.0)
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

        log_h, pit, inner_ess = _cpo_pit_integrals(ga, obs_lik)

        # v[1] < σ² → should be finite
        @test isfinite(log_h[1])

        # v[2], v[3] >= σ² → should be Inf (CPO failure)
        @test log_h[2] == Inf
        @test log_h[3] == Inf

        # PIT should still be computable for all
        @test all(isfinite.(pit))
    end

    @testset "Generic quadrature matches analytic for Normal" begin
        n = 5
        μ = randn(n)
        v = rand(n) .* 0.5 .+ 0.1
        Q = Diagonal(1.0 ./ v)
        ga = GMRF(μ, Q)

        σ_obs = 2.0
        y = randn(n)
        obs_lik = ExponentialFamily(Normal)(y; σ = σ_obs)

        # Analytic result (via NormalLikelihood dispatch)
        analytic_h, analytic_pit, analytic_ess = _cpo_pit_integrals(ga, obs_lik)

        # Force generic fallback
        generic_h, generic_pit, generic_ess = invoke(
            Latte._cpo_pit_integrals,
            Tuple{Any, Any},
            ga, obs_lik; n_nodes = 21
        )

        @test generic_h ≈ analytic_h atol = 1.0e-6
        @test generic_pit ≈ analytic_pit atol = 1.0e-6

        # Generic should have high ESS for well-behaved Normal (not dominated by one node)
        @test all(generic_ess .> 1.0)
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

    @testset "Inner failure detection: high-variance latent" begin
        # With very high latent variance, GH nodes spread into regions where
        # 1/p(y|x) spans many orders of magnitude → ESS drops toward 1
        n = 3
        Random.seed!(42)

        # High variance latent → large spread across GH nodes
        v = [100.0, 0.01, 50.0]  # obs 1 and 3 have huge variance, obs 2 is tight
        μ = [1.0, 1.0, 1.0]  # Reasonable Poisson mean (exp(1) ≈ 2.7)
        Q = Diagonal(1.0 ./ v)
        ga = GMRF(μ, Q)

        y = [3, 2, 5]
        obs_lik = ExponentialFamily(Poisson)(GaussianMarkovRandomFields.PoissonObservations(y))

        _, _, inner_ess = _cpo_pit_integrals(ga, obs_lik; n_nodes = 15)

        # High-variance observations should have lower ESS (fewer effective nodes)
        @test inner_ess[1] < inner_ess[2]
        @test inner_ess[3] < inner_ess[2]

        # The high-variance obs should have ESS close to 1 (dominated by one node)
        @test inner_ess[1] < 2.0
        @test inner_ess[3] < 2.0

        # The low-variance obs should have high ESS (many nodes contribute)
        @test inner_ess[2] > 2.0
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
end
