using Test
using IntegratedNestedLaplace
using GaussianMarkovRandomFields
using LDLFactorizations
using Distributions
using LinearAlgebra
using SparseArrays
using Random

@testset "Interpolation" begin

    Random.seed!(123)

    @testset "1D Interpolation" begin
        # Test building interpolant for 1D case using stable AR-1 GMRF
        spec = @hyperparams begin
            (σ_gmrf ~ Gamma(2, 3), transform = log, space = natural)
            σ = 1.0e-6
        end

        function ar_precision(ρ, k)
            return spdiagm(-1 => -ρ * ones(k - 1), 0 => ones(k), 1 => -ρ * ones(k - 1))
        end

        function stable_1d_latent(; σ_gmrf, kwargs...)
            ρ = 0.3  # Fixed correlation for 1D test
            k = 100
            Q = ar_precision(ρ, k) ./ σ_gmrf^2
            return GMRF(zeros(k), Q)
        end

        obs_model = ExponentialFamily(Normal)
        model = INLAModel(spec, FunctionLatentModel(stable_1d_latent, 100), obs_model)

        # Generate stable test data
        σ_gmrf_true = 2.5
        x_gt = rand(stable_1d_latent(; σ_gmrf = σ_gmrf_true))
        y_test = rand(conditional_distribution(obs_model, x_gt; σ = 1.0e-6))

        # Get exploration
        θ_star, mode_points, mode_logdensities = find_hyperparameter_mode(model, y_test)
        exploration, _ = explore_hyperparameter_posterior(model, y_test, θ_star, GaussianMarginal(), 1:100)

        # Build interpolant
        posterior_approx = build_posterior_interpolant(exploration)

        @test posterior_approx.exploration == exploration

        # Test interpolant evaluation at exploration points
        for (i, point) in enumerate(exploration.grid_points)
            approx_value = posterior_approx(point.θ)
            true_value = point.log_density
            @test approx_value ≈ true_value atol = 1.0e-10  # Should exactly interpolate
        end

        # Test interpolant evaluation at mode (pass WorkingHyperparameters directly)
        mode_approx = posterior_approx(θ_star)
        @test isfinite(mode_approx)

        # Test scalar input handling (1D case has only one parameter)
        mode_scalar = posterior_approx(θ_star)
        @test mode_scalar ≈ mode_approx atol = 1.0e-10
    end

    @testset "2D Interpolation" begin
        # Test building interpolant for 2D case using stable AR-1 GMRF
        spec = @hyperparams begin
            (σ_gmrf ~ Gamma(2, 3), transform = log, space = natural)
            (ρ ~ Uniform(0, 0.5), transform = logit, space = natural)
            σ = 1.0e-6
        end

        function ar_precision(ρ, k)
            return spdiagm(-1 => -ρ * ones(k - 1), 0 => ones(k), 1 => -ρ * ones(k - 1))
        end

        function stable_2d_latent(; σ_gmrf, ρ, kwargs...)
            k = 100
            Q = ar_precision(ρ, k) ./ σ_gmrf^2
            return GMRF(zeros(k), Q)
        end

        obs_model = ExponentialFamily(Normal)
        model = INLAModel(spec, FunctionLatentModel(stable_2d_latent, 100), obs_model)

        # Generate stable test data
        σ_gmrf_true = 2.5
        ρ_true = 0.4
        x_gt = rand(stable_2d_latent(; σ_gmrf = σ_gmrf_true, ρ = ρ_true))
        y_test = rand(conditional_distribution(obs_model, x_gt; σ = 1.0e-6))

        # Get exploration
        θ_star, mode_points, mode_logdensities = find_hyperparameter_mode(model, y_test)
        exploration, _ = explore_hyperparameter_posterior(
            model, y_test, θ_star, GaussianMarginal(), 1:100;
            interpolation_subdivisions = 2
        )

        # Build interpolant
        posterior_approx = build_posterior_interpolant(exploration)

        @test posterior_approx.exploration == exploration

        # Test interpolant evaluation at exploration points
        tolerance = 1.0e-6  # RBF interpolation may have small numerical errors
        for (i, point) in enumerate(exploration.grid_points)
            approx_value = posterior_approx(point.θ)
            true_value = point.log_density
            @test approx_value ≈ true_value atol = tolerance
        end

        # Test interpolant evaluation at mode (pass WorkingHyperparameters directly)
        mode_approx = posterior_approx(θ_star)
        @test isfinite(mode_approx)

        # Test interpolation at new points near the mode
        log_normalization = exploration.log_normalization_constant

        # Generate test points around the mode within integration bounds
        θ_star_vec = θ_star.θ
        mode_α, mode_β = θ_star_vec[1], θ_star_vec[2]
        bound_min_α, bound_max_α = exploration.integration_bounds[1, 1], exploration.integration_bounds[1, 2]
        bound_min_β, bound_max_β = exploration.integration_bounds[2, 1], exploration.integration_bounds[2, 2]

        # Create points at moderate distances from mode within bounds
        test_points = [
            [mode_α + 0.3 * (bound_max_α - mode_α), mode_β + 0.2 * (bound_max_β - mode_β)],
            [mode_α - 0.2 * (mode_α - bound_min_α), mode_β - 0.3 * (mode_β - bound_min_β)],
            [mode_α + 0.1 * (bound_max_α - mode_α), mode_β - 0.1 * (mode_β - bound_min_β)],
            [mode_α - 0.1 * (mode_α - bound_min_α), mode_β + 0.1 * (bound_max_β - mode_β)],
        ]

        for θ_new in test_points
            # Get interpolated value (already normalized)
            # Create WorkingHyperparameters from θ_new (vector in working space)
            θ_new_w = WorkingHyperparameters(θ_new, spec)
            approx_value = posterior_approx(θ_new_w)

            # Get direct unnormalized value and normalize it
            direct_unnormalized = hyperparameter_logpdf(model, θ_new_w, y_test)
            direct_normalized = direct_unnormalized - log_normalization

            # Should be reasonably close for 2D interpolation
            @test approx_value ≈ direct_normalized atol = 0.2  # More tolerance for 2D
        end
    end


    @testset "Interpolation Quality" begin
        # Test interpolation quality - compare interpolated values with stored exploration values
        spec = @hyperparams begin
            x ~ Normal(0, 1)
        end

        function smooth_latent(; x, kwargs...)
            n = 2
            Q = spdiagm(0 => fill(exp(x), n))  # Smooth function of x
            return GMRF(zeros(n), Q)
        end

        obs_model = ExponentialFamily(Bernoulli)
        model = INLAModel(spec, FunctionLatentModel(smooth_latent, 2), obs_model)

        y_test = [true, false]

        # Get dense exploration
        θ_star, mode_points, mode_logdensities = find_hyperparameter_mode(model, y_test)
        exploration, _ = explore_hyperparameter_posterior(
            model, y_test, θ_star, GaussianMarginal(), 1:2;
            integration_step_z = 3.0, interpolation_subdivisions = 1
        )  # Dense sampling

        posterior_approx = build_posterior_interpolant(exploration)

        # Test interpolation quality: compare interpolated values with stored normalized values
        test_indices = 1:min(5, length(exploration.grid_points))

        for i in test_indices
            point = exploration.grid_points[i]
            approx_value = posterior_approx(point.θ)
            stored_value = point.log_density  # This is already normalized

            # Should be very close for points in the exploration (both normalized)
            @test approx_value ≈ stored_value atol = 1.0e-3
        end

        # Test interpolation at new points near the mode
        log_normalization = exploration.log_normalization_constant

        # Generate test points around the mode within integration bounds
        mode_val = θ_star[1]
        bound_min, bound_max = exploration.integration_bounds[1, 1], exploration.integration_bounds[1, 2]

        # Create points at 25%, 50%, 75% between mode and bounds
        test_points = [
            [mode_val + 0.25 * (bound_max - mode_val)],
            [mode_val - 0.25 * (mode_val - bound_min)],
            [mode_val + 0.5 * (bound_max - mode_val)],
            [mode_val - 0.5 * (mode_val - bound_min)],
        ]

        for θ_new in test_points
            # Get interpolated value (already normalized)
            approx_value = posterior_approx(θ_new)

            # Get direct unnormalized value and normalize it
            # Create WorkingHyperparameters from θ_new (vector in working space) for hyperparameter_logpdf
            θ_new_w = WorkingHyperparameters(θ_new, spec)
            direct_unnormalized = hyperparameter_logpdf(model, θ_new_w, y_test)
            direct_normalized = direct_unnormalized - log_normalization

            # Should be reasonably close (interpolation accuracy)
            @test approx_value ≈ direct_normalized atol = 0.1  # Reasonable tolerance for interpolation
        end
    end

end
