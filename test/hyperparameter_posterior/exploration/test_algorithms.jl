using Test
using IntegratedNestedLaplace
using Distributions
using GaussianMarkovRandomFields
using SparseArrays
using LinearAlgebra

# Create a test model using the exact same setup as the working example
function create_test_model(k = 100)  # Smaller than example but still stable
    # AR-1 precision matrix function (exactly from the example)
    function ar_precision(ρ, k)
        return spdiagm(-1 => -ρ * ones(k - 1), 0 => ones(k), 1 => -ρ * ones(k - 1))
    end

    # Hyperparameter prior (exactly from the example)
    θ_prior = HyperparameterPrior((σ_gmrf = Gamma(2, 3), ρ = Uniform(0, 0.5)), fixed = (σ = 1.0e-6,))

    # Function to create latent GMRF (exactly from the example)
    function latent_gmrf(θ)
        σ = θ.σ_gmrf
        ρ = θ.ρ
        Q = ar_precision(ρ, k) ./ σ^2
        μ = zeros(k)
        return GMRF(μ, Q, CholeskySolverBlueprint())
    end

    # Observation model (exactly from the example)
    obs_model = ExponentialFamily(Normal)

    # Create INLA model
    return INLAModel(θ_prior, latent_gmrf, obs_model), k
end

# Generate test data using the exact same method as the working example
function generate_stable_test_data(model, k)
    # True hyperparameter values (from the example)
    σ_gmrf_true = 2.5   # marginal standard deviation
    ρ_true = 0.4        # autocorrelation coefficient

    # Generate synthetic data (exactly from the example)
    x_gt = rand(latent_gmrf(model, (σ_gmrf = σ_gmrf_true, ρ = ρ_true)))
    y_gt = rand(likelihood(model.observation_model, x_gt, (σ = 1.0e-6,)))

    return y_gt, [σ_gmrf_true, ρ_true]
end

@testset "explore_half_axis_by_steps Tests" begin
    model, k = create_test_model(50)  # Stable size for testing

    # Generate stable test data using the working example method
    y_test, θ_true = generate_stable_test_data(model, k)

    # Find the actual mode
    θ_mode, _, _ = find_hyperparameter_mode(model, y_test)

    # Create transformation
    transform = compute_reparameterization(model, y_test, θ_mode)
    mode_logpdf = hyperparameter_logpdf(model, θ_mode, y_test)

    @testset "Basic Exploration" begin
        # Test exploring in positive direction along first dimension
        keyed_points = explore_half_axis_by_steps(
            model, y_test, transform, mode_logpdf,
            1, 1, 0.5, 2.0, 2,
            GaussianMarginal(), 1:5  # Smaller subset for testing
        )

        @test length(keyed_points) > 0
        @test all(p -> p[1] isa NTuple{2, Int}, keyed_points)  # Keys are tuples of integers
        @test all(p -> p[2] isa GridPoint, keyed_points)  # Values are GridPoints

        # Check that keys are properly structured for dimension 1, positive direction
        for (key, point) in keyed_points
            @test key[1] > 0  # First dimension should be positive
            @test key[2] == 0  # Second dimension should be zero
        end
    end

    @testset "Negative Direction" begin
        # Test exploring in negative direction
        keyed_points = explore_half_axis_by_steps(
            model, y_test, transform, mode_logpdf,
            1, -1, 0.5, 2.0, 2,
            GaussianMarginal(), 1:5
        )

        @test length(keyed_points) > 0

        # Check that keys are properly structured for dimension 1, negative direction
        for (key, point) in keyed_points
            @test key[1] < 0  # First dimension should be negative
            @test key[2] == 0  # Second dimension should be zero
        end
    end

    @testset "Log Density Decreases" begin
        # Test that log density decreases as we move away from mode
        keyed_points = explore_half_axis_by_steps(
            model, y_test, transform, mode_logpdf,
            1, 1, 0.5, 2.0, 2,
            GaussianMarginal(), 1:5
        )

        if length(keyed_points) > 1
            # Sort by step distance from mode
            sorted_points = sort(keyed_points, by = p -> abs(p[1][1]))
            densities = [p[2].log_density for p in sorted_points]

            # Check that density generally decreases (allowing for some numerical noise)
            @test densities[1] >= densities[end] - 0.1  # Some tolerance for numerical issues
        end
    end

    @testset "Stopping Condition" begin
        # Test with small max_log_drop to ensure early stopping
        keyed_points = explore_half_axis_by_steps(
            model, y_test, transform, mode_logpdf,
            1, 1, 0.5, 0.5, 2,  # Small max_log_drop
            GaussianMarginal(), 1:5
        )

        # Should stop earlier with smaller max_log_drop
        if length(keyed_points) > 0
            final_density = keyed_points[end][2].log_density
            @test mode_logpdf - final_density <= 0.5 + 0.1  # Some tolerance
        end
    end

    @testset "Integration Point Marking" begin
        # Test that integration points are correctly marked
        keyed_points = explore_half_axis_by_steps(
            model, y_test, transform, mode_logpdf,
            1, 1, 0.5, 2.0, 2,  # interpolation_subdivisions = 2
            GaussianMarginal(), 1:5
        )

        # Check that marginal results are computed for integration points
        for (key, point) in keyed_points
            step_count = key[1]  # For dimension 1, positive direction
            is_integration_point = (step_count % 2 == 0)

            if is_integration_point
                @test point.marginal_result !== nothing
            end
        end
    end
end

@testset "explore_dimension_and_build_lookup Tests" begin
    model, k = create_test_model(50)

    # Generate stable test data using the working example method
    y_test, θ_true = generate_stable_test_data(model, k)

    # Find the actual mode
    θ_mode, _, _ = find_hyperparameter_mode(model, y_test)

    # Create transformation
    transform = compute_reparameterization(model, y_test, θ_mode)
    mode_logpdf = hyperparameter_logpdf(model, θ_mode, y_test)

    @testset "Basic Dimension Exploration" begin
        point_lookup, step_range = explore_dimension_and_build_lookup(
            model, y_test, transform, mode_logpdf,
            1, 0.5, 2.0, 2,
            GaussianMarginal(), 1:5
        )

        @test point_lookup isa Dict
        @test step_range isa UnitRange{Int}
        @test length(point_lookup) > 0

        # Check that the range includes both positive and negative steps
        @test minimum(step_range) <= 0
        @test maximum(step_range) >= 0
    end

    @testset "Lookup Table Structure" begin
        point_lookup, step_range = explore_dimension_and_build_lookup(
            model, y_test, transform, mode_logpdf,
            1, 0.5, 2.0, 2,
            GaussianMarginal(), 1:5
        )

        # Check that all keys in the lookup table have the right structure
        for (key, point) in point_lookup
            @test key isa NTuple{2, Int}  # 2D hyperparameter space
            @test point isa GridPoint
            @test key[2] == 0  # Second dimension should be zero for axis exploration
        end

        # Check that step_range covers the actual steps found
        step_indices = [key[1] for key in keys(point_lookup)]
        @test minimum(step_indices) >= minimum(step_range)
        @test maximum(step_indices) <= maximum(step_range)
    end

    @testset "Symmetry Check" begin
        # Test that exploration finds points in both directions
        point_lookup, step_range = explore_dimension_and_build_lookup(
            model, y_test, transform, mode_logpdf,
            1, 0.5, 2.0, 2,
            GaussianMarginal(), 1:5
        )

        step_indices = [key[1] for key in keys(point_lookup)]
        has_positive = any(s > 0 for s in step_indices)
        has_negative = any(s < 0 for s in step_indices)

        @test has_positive  # Should find positive steps
        @test has_negative  # Should find negative steps
    end
end

@testset "explore_hyperparameter_posterior Tests" begin
    model, k = create_test_model(1000)  # Large stable model for full exploration

    # Generate stable test data using the working example method
    y_test, θ_true = generate_stable_test_data(model, k)

    # Find the actual mode
    θ_mode, _, _ = find_hyperparameter_mode(model, y_test)

    @testset "Basic Full Exploration" begin
        exploration = explore_hyperparameter_posterior(
            model, y_test, θ_mode,
            GaussianMarginal(), 1:1000;
            integration_step_z = 1.0,
            max_log_drop = 2.0,
            interpolation_subdivisions = 2
        )

        @test exploration isa HyperparameterExploration
        @test length(exploration.grid_points) > 0
        @test length(exploration.integration_indices) > 0
        @test exploration.transform isa ReparameterizationTransform
        @test exploration.log_normalization_constant isa Float64
    end

    @testset "Mode Point Included" begin
        exploration = explore_hyperparameter_posterior(
            model, y_test, θ_mode,
            GaussianMarginal(), 1:1000;
            integration_step_z = 1.0,
            max_log_drop = 2.0,
            interpolation_subdivisions = 2
        )

        # Check that mode point is included
        mode_found = false

        for point in exploration.grid_points
            if isapprox(point.θ, θ_mode, atol = 1.0e-6)
                mode_found = true
                @test point.marginal_result !== nothing  # Mode should have marginals
            end
        end
        @test mode_found
    end

    @testset "Integration Points Have Marginals" begin
        exploration = explore_hyperparameter_posterior(
            model, y_test, θ_mode,
            GaussianMarginal(), 1:1000;
            integration_step_z = 1.0,
            max_log_drop = 2.0,
            interpolation_subdivisions = 2
        )

        # Check that all integration points have marginal results
        for idx in exploration.integration_indices
            @test exploration.grid_points[idx].marginal_result !== nothing
        end
    end

    @testset "Normalized Densities" begin
        exploration = explore_hyperparameter_posterior(
            model, y_test, θ_mode,
            GaussianMarginal(), 1:1000;
            integration_step_z = 1.0,
            max_log_drop = 2.0,
            interpolation_subdivisions = 2
        )

        # Check that integration points can be used to compute normalized weights
        integration_points = exploration.grid_points[exploration.integration_indices]
        log_densities = [p.log_density for p in integration_points]

        # Should be able to compute weights without numerical issues
        weights = exp.(log_densities)
        @test all(isfinite.(weights))
        @test sum(weights) > 0
    end

    @testset "Parameter Bounds" begin
        exploration = explore_hyperparameter_posterior(
            model, y_test, θ_mode,
            GaussianMarginal(), 1:1000;
            integration_step_z = 0.5,  # Smaller step size
            max_log_drop = 1.0,        # Smaller drop
            interpolation_subdivisions = 2
        )

        # Check that exploration respects parameter bounds
        for point in exploration.grid_points
            θ_named = to_named(point.θ, model.hyperparameter_prior)
            @test θ_named.σ > 0      # Sigma should be positive
            @test 0 <= θ_named.ρ <= 0.5  # Rho should be in [0, 0.5]
        end
    end
end
