using Test
using Latte
using Latte: accumulate!, finalize!, _integrated_pointwise_loglik
using GaussianMarkovRandomFields
using Distributions
using SparseArrays
using LinearAlgebra
using Statistics

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

        integrated_ll, expected_log_ll = _integrated_pointwise_loglik(ga, obs_lik)

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

    @testset "Generic quadrature matches analytic for Normal" begin
        # Force the generic (quadrature) path via invoke and compare to analytic
        n = 5
        μ = randn(n)
        v = rand(n) .+ 0.1
        Q = Diagonal(1.0 ./ v)
        ga = GMRF(μ, Q)

        σ_obs = 2.0
        y = randn(n)
        obs_lik = ExponentialFamily(Normal)(y; σ = σ_obs)

        # Analytic result (via NormalLikelihood dispatch)
        analytic_int, analytic_ell = _integrated_pointwise_loglik(ga, obs_lik)

        # Force generic fallback by invoking with less-specific type signature
        generic_int, generic_ell = invoke(
            Latte._integrated_pointwise_loglik,
            Tuple{Any, Any},
            ga, obs_lik; n_nodes = 15
        )

        @test generic_int ≈ analytic_int atol = 1.0e-8
        @test generic_ell ≈ analytic_ell atol = 1.0e-8
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
