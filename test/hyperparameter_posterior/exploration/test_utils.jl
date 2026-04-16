using Test
using IntegratedNestedLaplace
using Distributions
using GaussianMarkovRandomFields
using SparseArrays
using LinearAlgebra

# Create a simple test model for testing exploration functions
function create_test_model(k = 10)
    # Simple AR-1 precision matrix
    function ar_precision(ρ, k)
        return spdiagm(-1 => -ρ * ones(k - 1), 0 => ones(k), 1 => -ρ * ones(k - 1))
    end

    # Hyperparameter prior (matching the working example)
    θ_prior = HyperparameterPrior((σ_gmrf = Gamma(2, 3), ρ = Uniform(0, 0.5)), fixed = (σ = 1.0e-6,))

    # Function to create latent GMRF (matching the working example)
    function latent_gmrf(; σ_gmrf, ρ, kwargs...)
        Q = ar_precision(ρ, k) ./ σ_gmrf^2
        μ = zeros(k)
        return GMRF(μ, Q)
    end

    # Observation model
    obs_model = ExponentialFamily(Normal)

    # Create INLA model
    return INLAModel(θ_prior, FunctionLatentModel(latent_gmrf, k), obs_model), k
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

@testset "create_weighted_mixtures Tests" begin
    @testset "Valid WeightedMixture Creation" begin
        # Create mock marginal results
        component1 = Normal(0.0, 1.0)
        component2 = Normal(1.0, 1.5)
        component3 = Normal(-0.5, 0.8)

        mock_marginal_result1 = MarginalResult([1, 2], [component1, component2], GaussianMarginal(), 0.1)
        mock_marginal_result2 = MarginalResult([1, 2], [component2, component3], GaussianMarginal(), 0.1)
        mock_marginal_result3 = MarginalResult([1, 2], [component3, component1], GaussianMarginal(), 0.1)

        # Create GridPoints with marginal results
        θ1, θ2, θ3 = [1.0, 2.0], [1.5, 2.5], [2.0, 3.0]
        points = [
            GridPoint(θ1, -1.0, mock_marginal_result1),
            GridPoint(θ2, -1.5, mock_marginal_result2),
            GridPoint(θ3, -2.0, mock_marginal_result3),
        ]

        # Create exploration with all points as integration points
        integration_indices = [1, 2, 3]
        mock_transform = ReparameterizationTransform(
            θ1,
            Matrix{Float64}(I, 2, 2),
            Diagonal([1.0, 1.0]),
            Matrix{Float64}(I, 2, 2)
        )
        exploration = GridExploration(
            points, integration_indices, mock_transform, -10.0
        )

        # Test the function
        mixtures = create_weighted_mixtures(exploration)

        @test length(mixtures) == 2  # Two variables
        @test all(m isa WeightedMixture for m in mixtures)

        # Test that weights are properly normalized
        for mixture in mixtures
            @test sum(mixture.weights) ≈ 1.0 atol = 1.0e-10
        end
    end

    @testset "Weight Normalization" begin
        # Test that weights are computed correctly from log densities
        component = Normal(0.0, 1.0)
        mock_marginal_result = MarginalResult([1], [component], GaussianMarginal(), 0.1)

        # Create points with different log densities
        points = [
            GridPoint([1.0], -1.0, mock_marginal_result),
            GridPoint([2.0], -2.0, mock_marginal_result),
            GridPoint([3.0], -3.0, mock_marginal_result),
        ]

        integration_indices = [1, 2, 3]
        mock_transform = ReparameterizationTransform(
            [1.0],
            reshape([1.0], 1, 1),
            Diagonal([1.0]),
            reshape([1.0], 1, 1)
        )
        exploration = GridExploration(
            points, integration_indices, mock_transform, -10.0
        )

        mixtures = create_weighted_mixtures(exploration)

        # Check that weights correspond to normalized exponentials of log densities
        expected_weights = exp.([-1.0, -2.0, -3.0])
        expected_weights ./= sum(expected_weights)

        @test mixtures[1].weights ≈ expected_weights
    end

    @testset "Error Handling" begin
        # Test error when no marginal results available
        points = [
            GridPoint([1.0], -1.0, nothing),
            GridPoint([2.0], -2.0, nothing),
        ]

        integration_indices = [1, 2]
        mock_transform = ReparameterizationTransform(
            [1.0],
            reshape([1.0], 1, 1),
            Diagonal([1.0]),
            reshape([1.0], 1, 1)
        )
        exploration = GridExploration(
            points, integration_indices, mock_transform, -10.0
        )

        @test_throws AssertionError create_weighted_mixtures(exploration)
    end

    @testset "Empty Integration Points" begin
        # Test error when no integration points
        points = [GridPoint([1.0], -1.0, nothing)]
        integration_indices = Int[]
        mock_transform = ReparameterizationTransform(
            [1.0],
            reshape([1.0], 1, 1),
            Diagonal([1.0]),
            reshape([1.0], 1, 1)
        )
        exploration = GridExploration(
            points, integration_indices, mock_transform, -10.0
        )

        @test_throws AssertionError create_weighted_mixtures(exploration)
    end
end

@testset "evaluate_logpdf_and_marginals Tests" begin
    model, k = create_test_model(50)  # Stable model for testing

    # Generate stable test data
    y_test, θ_true = generate_stable_test_data(model, k)
    θ_test = θ_true

    @testset "Basic Function Call" begin
        log_density, marginal_result = evaluate_logpdf_and_marginals(
            model, y_test, θ_test;
            compute_marginals = false
        )

        @test log_density isa Float64
        @test marginal_result === nothing
    end

    @testset "Marginal Computation" begin
        # Test with marginals enabled
        log_density, marginal_result = evaluate_logpdf_and_marginals(
            model, y_test, θ_test;
            compute_marginals = true,
            marginalization_method = GaussianMarginal(),
            marginalization_indices = 1:5
        )

        @test log_density isa Float64
        @test marginal_result !== nothing
        @test hasfield(typeof(marginal_result), :marginals)
    end

    @testset "Consistency Check" begin
        # Test that same θ gives same log_density regardless of marginal computation
        log_density1, _ = evaluate_logpdf_and_marginals(
            model, y_test, θ_test; compute_marginals = false
        )

        log_density2, _ = evaluate_logpdf_and_marginals(
            model, y_test, θ_test;
            compute_marginals = true,
            marginalization_method = GaussianMarginal(),
            marginalization_indices = 1:5
        )

        @test log_density1 ≈ log_density2 atol = 1.0e-10
    end
end
