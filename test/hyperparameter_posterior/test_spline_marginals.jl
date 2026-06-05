using Test
using Latte
using GaussianMarkovRandomFields
using Distributions
using LinearAlgebra
using SparseArrays
using Random
using DataFrames
using HCubature

@testset "Spline-Based Hyperparameter Marginals" begin

    # ====== Shared test fixtures ======

    function setup_1d_model(; strategy = GridExplorationStrategy())
        Random.seed!(42)
        spec = @hyperparams begin
            (α ~ Gamma(2, 1), transform = log, space = natural)
        end

        function latent(; α, kwargs...)
            n = 5
            Q = spdiagm(0 => fill(α, n))
            return (zeros(n), Q)
        end
        obs_model = ExponentialFamily(Bernoulli)
        model = LatentGaussianModel(spec, FunctionLatentModel(latent, 5), obs_model)
        y = [true, false, true, false, true]

        θ_star, _, _ = find_hyperparameter_mode(model, y)
        exploration, _ = explore_hyperparameter_posterior(
            strategy, model, y, θ_star, GaussianMarginal(), 1:5
        )

        return model, y, exploration
    end

    function setup_2d_model()
        Random.seed!(123)
        k = 100

        spec = @hyperparams begin
            (σ_gmrf ~ Gamma(2, 3), transform = log, space = natural)
            (ρ ~ Uniform(0, 0.5), transform = logit, space = natural)
            σ = 1.0e-6
        end

        function ar_precision(ρ, k)
            return spdiagm(-1 => -ρ * ones(k - 1), 0 => ones(k), 1 => -ρ * ones(k - 1))
        end

        function latent(; σ_gmrf, ρ, kwargs...)
            Q = ar_precision(ρ, k) ./ σ_gmrf^2
            return (zeros(k), Q)
        end
        obs_model = ExponentialFamily(Normal)
        model = LatentGaussianModel(spec, FunctionLatentModel(latent, k), obs_model)

        σ_gmrf_true = 2.5
        ρ_true = 0.4
        x_gt = rand(GMRF(latent(; σ_gmrf = σ_gmrf_true, ρ = ρ_true)...))
        y = rand(conditional_distribution(obs_model, x_gt; σ = spec.fixed.σ))

        θ_star, _, _ = find_hyperparameter_mode(model, y)
        exploration, _ = explore_hyperparameter_posterior(
            GridExplorationStrategy(interpolation_subdivisions = 2),
            model, y, θ_star, GaussianMarginal(), 1:k
        )

        return model, y, exploration
    end

    # ====== Strategy types exist ======

    @testset "Strategy Types" begin
        @test GridSumMarginal <: HyperparameterMarginalizationMethod
        @test CCDInterpolantMarginal <: HyperparameterMarginalizationMethod
        @test AutoHyperparameterMarginal <: HyperparameterMarginalizationMethod

        # Constructors
        gs = GridSumMarginal()
        @test gs isa GridSumMarginal

        cci = CCDInterpolantMarginal()
        @test cci.n_grid == 200  # default

        cci2 = CCDInterpolantMarginal(n_grid = 100)
        @test cci2.n_grid == 100

        auto = AutoHyperparameterMarginal()
        @test auto.n_grid == 200  # default
    end

    # ====== GridSumMarginal tests ======

    @testset "GridSumMarginal D=1" begin
        model, y, exploration = setup_1d_model()

        result = marginalize_hyperparameters(
            GridSumMarginal(), exploration, model, y
        )

        # Should return NamedTuple with SplineMarginalDistribution
        @test result isa NamedTuple
        @test length(result) == 1
        @test first(result) isa SplineMarginalDistribution

        marginal = first(result)

        # Basic sanity: mode should be finite and in bounds
        @test isfinite(mode(marginal))
        @test isfinite(mean(marginal))
        @test var(marginal) > 0

        # CDF should be proper
        @test cdf(marginal, minimum(marginal)) ≈ 0.0 atol = 0.01
        @test cdf(marginal, maximum(marginal)) ≈ 1.0 atol = 0.01

        # Quantiles should be monotone
        q_values = [quantile(marginal, p) for p in 0.1:0.1:0.9]
        @test issorted(q_values)
    end

    @testset "GridSumMarginal summary_df" begin
        model, y, exploration = setup_1d_model()

        result = marginalize_hyperparameters(
            GridSumMarginal(), exploration, model, y
        )

        df = summary_df(result)
        @test df isa DataFrame
        @test nrow(df) == 1
        @test :mode in propertynames(df)
        @test :mean in propertynames(df)
        @test :std in propertynames(df)
        @test :q2_5 in propertynames(df)
        @test :q97_5 in propertynames(df)
        @test df.q2_5[1] < df.median[1] < df.q97_5[1]
    end

    # ====== CCDInterpolantMarginal tests ======

    @testset "CCDInterpolantMarginal D=2" begin
        model, y, exploration = setup_2d_model()

        result = marginalize_hyperparameters(
            CCDInterpolantMarginal(), exploration, model, y
        )

        # Should return NamedTuple with 2 SplineMarginalDistribution entries
        @test result isa NamedTuple
        @test length(result) == 2
        @test all(v isa SplineMarginalDistribution for v in values(result))

        # Check parameter names match
        spec = model.hyperparameter_spec
        param_names = collect(keys(spec.free))
        @test collect(keys(result)) == param_names

        for marginal in values(result)
            @test isfinite(mode(marginal))
            @test isfinite(mean(marginal))
            @test var(marginal) > 0
            @test cdf(marginal, minimum(marginal)) ≈ 0.0 atol = 0.01
            @test cdf(marginal, maximum(marginal)) ≈ 1.0 atol = 0.01
        end
    end

    @testset "CCDInterpolantMarginal summary_df" begin
        model, y, exploration = setup_2d_model()

        result = marginalize_hyperparameters(
            CCDInterpolantMarginal(), exploration, model, y
        )

        df = summary_df(result)
        @test df isa DataFrame
        @test nrow(df) == 2
        @test all(df.q2_5 .< df.median .< df.q97_5)
    end

    @testset "CCDInterpolantMarginal performance (D=2)" begin
        model, y, exploration = setup_2d_model()

        # summary_df with new approach should be fast (< 5s total including construction)
        t = @elapsed begin
            result = marginalize_hyperparameters(
                CCDInterpolantMarginal(), exploration, model, y
            )
            df = summary_df(result)
        end

        @test t < 5.0  # Should be << 1s, but allow margin for CI
        @test nrow(df) == 2
    end

    # ====== SplineMarginalDistribution mathematical properties ======

    @testset "SplineMarginalDistribution: PDF normalization" begin
        model, y, exploration = setup_1d_model()
        result = marginalize_hyperparameters(GridSumMarginal(), exploration, model, y)
        marginal = first(result)

        # PDF should integrate to 1
        total_mass, _ = hcubature(
            x -> pdf(marginal, x[1]),
            [minimum(marginal)], [maximum(marginal)],
            rtol = 1.0e-3, atol = 1.0e-6
        )
        @test total_mass ≈ 1.0 rtol = 1.0e-2
    end

    @testset "SplineMarginalDistribution: moments vs numerical integration" begin
        model, y, exploration = setup_1d_model()
        result = marginalize_hyperparameters(GridSumMarginal(), exploration, model, y)
        marginal = first(result)

        a, b = minimum(marginal), maximum(marginal)

        # Normalize by total mass to isolate moment accuracy from normalization error
        Z, _ = hcubature(x -> pdf(marginal, x[1]), [a], [b], rtol = 1.0e-3, atol = 1.0e-6)

        # E[X] via numerical integration
        true_mean, _ = hcubature(
            x -> x[1] * pdf(marginal, x[1]), [a], [b], rtol = 1.0e-3, atol = 1.0e-6
        )
        true_mean /= Z

        # E[X²] via numerical integration
        true_second, _ = hcubature(
            x -> x[1]^2 * pdf(marginal, x[1]), [a], [b], rtol = 1.0e-3, atol = 1.0e-6
        )
        true_second /= Z
        true_var = true_second - true_mean^2

        @test mean(marginal) ≈ true_mean rtol = 1.0e-2
        @test var(marginal) ≈ true_var rtol = 1.0e-2
    end

    # The marginal's normalization + moments must be self-consistent with its
    # continuous spline pdf, independent of how coarse the exploration grid was.
    # (Computing Z / moments via discrete quadrature on the exploration grid
    # diverges from the spline integral when the grid has few points.)
    @testset "SplineMarginalDistribution: self-consistent under a coarse grid" begin
        model, y, exploration = setup_1d_model(
            strategy = GridExplorationStrategy(integration_step_z = 1.0, max_log_drop = 2.5)
        )
        result = marginalize_hyperparameters(GridSumMarginal(), exploration, model, y)
        marginal = first(result)
        a, b = minimum(marginal), maximum(marginal)

        # ∫ pdf over the support must be 1 by construction
        total_mass, _ = hcubature(x -> pdf(marginal, x[1]), [a], [b], rtol = 1.0e-3, atol = 1.0e-6)
        @test total_mass ≈ 1.0 rtol = 1.0e-2

        # stored moments must match integration of the same spline pdf
        Z, _ = hcubature(x -> pdf(marginal, x[1]), [a], [b], rtol = 1.0e-3, atol = 1.0e-6)
        tm, _ = hcubature(x -> x[1] * pdf(marginal, x[1]), [a], [b], rtol = 1.0e-3, atol = 1.0e-6)
        ts, _ = hcubature(x -> x[1]^2 * pdf(marginal, x[1]), [a], [b], rtol = 1.0e-3, atol = 1.0e-6)
        tm /= Z
        ts /= Z
        @test mean(marginal) ≈ tm rtol = 1.0e-2
        @test var(marginal) ≈ (ts - tm^2) rtol = 2.0e-2
    end

    @testset "SplineMarginalDistribution: CDF-quantile round-trip" begin
        model, y, exploration = setup_2d_model()
        result = marginalize_hyperparameters(
            CCDInterpolantMarginal(), exploration, model, y
        )

        for marginal in values(result)
            for q in [0.1, 0.25, 0.5, 0.75, 0.9]
                x = quantile(marginal, q)
                @test cdf(marginal, x) ≈ q rtol = 1.0e-2

                # Round-trip: quantile(cdf(x)) ≈ x
                q_back = cdf(marginal, x)
                x_back = quantile(marginal, q_back)
                @test x_back ≈ x rtol = 1.0e-2
            end
        end
    end

    @testset "SplineMarginalDistribution: CDF integrates PDF" begin
        model, y, exploration = setup_1d_model()
        result = marginalize_hyperparameters(GridSumMarginal(), exploration, model, y)
        marginal = first(result)

        # Check at several points that CDF(x) ≈ ∫ pdf(t) dt from min to x
        for q in [0.25, 0.5, 0.75]
            x = quantile(marginal, q)
            integrated, _ = hcubature(
                t -> pdf(marginal, t[1]),
                [minimum(marginal)], [x],
                rtol = 1.0e-3, atol = 1.0e-6
            )
            @test cdf(marginal, x) ≈ integrated rtol = 0.05
        end
    end

    @testset "SplineMarginalDistribution: pdf/logpdf consistency" begin
        model, y, exploration = setup_2d_model()
        result = marginalize_hyperparameters(
            CCDInterpolantMarginal(), exploration, model, y
        )

        for marginal in values(result)
            for q in [0.1, 0.5, 0.9]
                x = quantile(marginal, q)
                @test pdf(marginal, x) ≈ exp(logpdf(marginal, x)) rtol = 1.0e-10
            end
        end
    end

    # ====== AutoHyperparameterMarginal tests ======

    @testset "AutoHyperparameterMarginal dispatches correctly" begin
        # D=1: should use GridSum
        model_1d, y_1d, exploration_1d = setup_1d_model()
        result_1d = marginalize_hyperparameters(
            AutoHyperparameterMarginal(), exploration_1d, model_1d, y_1d
        )
        @test length(result_1d) == 1
        @test first(result_1d) isa SplineMarginalDistribution

        # D=2: should use CCDInterpolant
        model_2d, y_2d, exploration_2d = setup_2d_model()
        result_2d = marginalize_hyperparameters(
            AutoHyperparameterMarginal(), exploration_2d, model_2d, y_2d
        )
        @test length(result_2d) == 2
        @test all(v isa SplineMarginalDistribution for v in values(result_2d))
    end

    @testset "Degenerate grid raises a clear error" begin
        # Too few grid points to build a cubic spline → actionable error, not a
        # cryptic Tridiagonal/interpolation failure.
        spec = @hyperparams begin
            (α ~ Gamma(2, 1), transform = log, space = natural)
        end
        @test_throws ArgumentError Latte._build_spline_marginal([0.0], [0.0], spec, 1)
        @test_throws ArgumentError Latte._build_spline_marginal([0.0, 0.1], [0.0, -0.5], spec, 1)
    end

end
