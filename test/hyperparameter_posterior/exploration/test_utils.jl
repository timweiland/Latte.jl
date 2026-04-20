using Test
using Latte
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

    # Hyperparameter spec (matching test_grid.jl)
    spec = @hyperparams begin
        (σ_gmrf ~ Gamma(2, 3), transform = log, space = natural)
        (ρ ~ Uniform(0, 0.5), transform = logit, space = natural)
        σ = 1.0e-6  # Fixed parameter
    end

    # Function to create latent GMRF (matching the working example).
    # Scale nzval in place rather than broadcasting `./` so the sparsity pattern
    # stays stable under Float64 overflow (required by GMRFWorkspace).
    function latent_gmrf(; σ_gmrf, ρ, kwargs...)
        Q = ar_precision(ρ, k)
        Q.nzval ./= σ_gmrf^2
        μ = zeros(k)
        return (μ, Q)
    end
    # Observation model
    obs_model = ExponentialFamily(Normal)

    # Create INLA model
    return LatentGaussianModel(spec, FunctionLatentModel(latent_gmrf, k), obs_model), k
end

# Generate test data using the exact same method as the working example
function generate_stable_test_data(model, k)
    # True hyperparameter values (from the example)
    σ_gmrf_true = 2.5   # marginal standard deviation
    ρ_true = 0.4        # autocorrelation coefficient

    # Generate synthetic data (exactly from the example)
    x_gt = rand(model.latent_prior(; σ_gmrf = σ_gmrf_true, ρ = ρ_true))
    y_gt = rand(conditional_distribution(model.observation_model, x_gt; σ = 1.0e-6))

    return y_gt, [σ_gmrf_true, ρ_true]
end

# Minimal spec for mock tests that exercise GridPoint/ReparameterizationTransform
# directly with hand-crafted vectors rather than running inference.
mock_spec(n) = HyperparameterSpec(
    free = NamedTuple(Symbol(:p, i) => Hyperparameter(Normal(0, 1)) for i in 1:n)
)

@testset "create_weighted_mixtures Tests" begin
    @testset "Valid WeightedMixture Creation" begin
        # Create mock marginal results (kld_values is the 5th positional arg)
        component1 = Normal(0.0, 1.0)
        component2 = Normal(1.0, 1.5)
        component3 = Normal(-0.5, 0.8)

        mock_marginal_result1 = MarginalResult([1, 2], [component1, component2], GaussianMarginal(), 0.1, zeros(2))
        mock_marginal_result2 = MarginalResult([1, 2], [component2, component3], GaussianMarginal(), 0.1, zeros(2))
        mock_marginal_result3 = MarginalResult([1, 2], [component3, component1], GaussianMarginal(), 0.1, zeros(2))

        # Create GridPoints with marginal results
        spec2 = mock_spec(2)
        θ1 = WorkingHyperparameters([1.0, 2.0], spec2)
        θ2 = WorkingHyperparameters([1.5, 2.5], spec2)
        θ3 = WorkingHyperparameters([2.0, 3.0], spec2)
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
        mixture_result = create_weighted_mixtures(exploration)

        @test length(mixture_result.marginals) == 2  # Two variables
        @test all(m isa WeightedMixture for m in mixture_result.marginals)

        # Test that weights are properly normalized
        for mixture in mixture_result.marginals
            @test sum(mixture.weights) ≈ 1.0 atol = 1.0e-10
        end
    end

    @testset "Weight Normalization" begin
        # Test that weights are computed correctly from log densities
        component = Normal(0.0, 1.0)
        mock_marginal_result = MarginalResult([1], [component], GaussianMarginal(), 0.1, zeros(1))

        # Create points with different log densities
        spec1 = mock_spec(1)
        points = [
            GridPoint(WorkingHyperparameters([1.0], spec1), -1.0, mock_marginal_result),
            GridPoint(WorkingHyperparameters([2.0], spec1), -2.0, mock_marginal_result),
            GridPoint(WorkingHyperparameters([3.0], spec1), -3.0, mock_marginal_result),
        ]

        integration_indices = [1, 2, 3]
        mock_transform = ReparameterizationTransform(
            WorkingHyperparameters([1.0], spec1),
            reshape([1.0], 1, 1),
            Diagonal([1.0]),
            reshape([1.0], 1, 1)
        )
        exploration = GridExploration(
            points, integration_indices, mock_transform, -10.0
        )

        mixture_result = create_weighted_mixtures(exploration)

        # Check that weights correspond to normalized exponentials of log densities
        expected_weights = exp.([-1.0, -2.0, -3.0])
        expected_weights ./= sum(expected_weights)

        @test mixture_result.marginals[1].weights ≈ expected_weights
    end

    @testset "Error Handling" begin
        # Test error when no marginal results available
        spec1 = mock_spec(1)
        points = [
            GridPoint(WorkingHyperparameters([1.0], spec1), -1.0, nothing),
            GridPoint(WorkingHyperparameters([2.0], spec1), -2.0, nothing),
        ]

        integration_indices = [1, 2]
        mock_transform = ReparameterizationTransform(
            WorkingHyperparameters([1.0], spec1),
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
        # The GridExploration constructor itself rejects empty integration indices
        # (via _compute_integration_bounds), so the error surfaces at construction
        # rather than inside create_weighted_mixtures.
        spec1 = mock_spec(1)
        points = [GridPoint(WorkingHyperparameters([1.0], spec1), -1.0, nothing)]
        mock_transform = ReparameterizationTransform(
            WorkingHyperparameters([1.0], spec1),
            reshape([1.0], 1, 1),
            Diagonal([1.0]),
            reshape([1.0], 1, 1)
        )

        @test_throws ErrorException GridExploration(points, Int[], mock_transform, -10.0)
    end
end

@testset "evaluate_at_grid_point Tests" begin
    model, k = create_test_model(50)  # Stable model for testing

    # Generate stable test data
    y_test, _ = generate_stable_test_data(model, k)

    # Evaluate at the mode, which is returned as a WorkingHyperparameters.
    θ_mode, _, _ = find_hyperparameter_mode(model, y_test)

    # One workspace reused across the testset
    θ_mode_nt = convert(NamedTuple, convert(NaturalHyperparameters, θ_mode))
    ws = make_workspace(model.latent_prior; θ_mode_nt...)

    @testset "Basic Function Call" begin
        result = evaluate_at_grid_point(
            model, y_test, θ_mode;
            ws = ws,
            compute_marginals = false,
        )

        @test result.log_density isa Float64
        @test result.marginal_result === nothing
    end

    @testset "Marginal Computation" begin
        # Test with marginals enabled
        result = evaluate_at_grid_point(
            model, y_test, θ_mode;
            ws = ws,
            compute_marginals = true,
            marginalization_method = GaussianMarginal(),
            marginalization_indices = 1:5,
        )

        @test result.log_density isa Float64
        @test result.marginal_result !== nothing
        @test hasfield(typeof(result.marginal_result), :marginals)
    end

    @testset "Consistency Check" begin
        # Test that same θ gives same log_density regardless of marginal computation
        result1 = evaluate_at_grid_point(
            model, y_test, θ_mode; ws = ws, compute_marginals = false,
        )

        result2 = evaluate_at_grid_point(
            model, y_test, θ_mode;
            ws = ws,
            compute_marginals = true,
            marginalization_method = GaussianMarginal(),
            marginalization_indices = 1:5,
        )

        @test result1.log_density ≈ result2.log_density atol = 1.0e-10
    end
end
