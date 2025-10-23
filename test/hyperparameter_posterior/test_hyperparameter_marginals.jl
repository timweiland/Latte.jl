using Test
using IntegratedNestedLaplace
using GaussianMarkovRandomFields
using LDLFactorizations
using Distributions
using LinearAlgebra
using SparseArrays

@testset "Marginal Posterior Computation" begin
    spec = @hyperparams begin
        (μ ~ Uniform(0.0, 2.0), transform = Bijectors.Logit(0.0, 2.0), space = natural)
        (σ ~ LogNormal(log(0.02), 0.1), transform = log, space = natural)
    end
    n = 10
    Q = sparse(1.0e6 * I, (n, n))
    function latent_gmrf(; μ, kwargs...)
        return GMRF(μ .* ones(n), Q)
    end
    obs_model = ExponentialFamily(Normal)
    model = INLAModel(spec, latent_gmrf, obs_model)
    θ_gt, x_gt, y_gt = rand(model)

    @testset "1D Marginals (Identity Case)" begin
        # For 1D case, marginal should equal the full posterior
        spec = @hyperparams begin
            (α ~ Gamma(2, 1), transform = log, space = natural)
        end

        function alpha_latent(; α, kwargs...)
            n = 3
            Q = spdiagm(0 => fill(α, n))
            return GMRF(zeros(n), Q)
        end

        obs_model = ExponentialFamily(Bernoulli)
        model = INLAModel(spec, alpha_latent, obs_model)

        y_test = [true, false, true]

        # Get posterior approximation
        θ_star, mode_points, mode_logdensities = find_hyperparameter_mode(model, y_test)
        exploration = explore_hyperparameter_posterior(model, y_test, θ_star, GaussianMarginal(), 1:3)
        posterior_approx = build_posterior_interpolant(exploration)

        # Test marginal computation
        test_values = [θ_star[1], θ_star[1] + 0.1, θ_star[1] - 0.1]

        for test_val in test_values
            marginal_logpdf = hyperparameter_marginal_logpdf(posterior_approx, 1, test_val)
            direct_logpdf = posterior_approx([test_val])

            # For 1D case, marginal should equal direct evaluation
            @test marginal_logpdf ≈ direct_logpdf atol = 1.0e-6
        end
    end

    @testset "2D Marginal Consistency" begin
        # Test 2D case with marginal consistency checks - use more informative priors
        spec = @hyperparams begin
            (σ_latent ~ InverseGamma(3, 2), transform = log, space = natural)
            (σ ~ InverseGamma(3, 2), transform = log, space = natural)
        end

        function two_variance_latent(; σ_latent, kwargs...)
            n = 6  # More data points
            Q = spdiagm(0 => fill(1 / σ_latent^2 + 1.0e-6, n))  # Add small regularization
            return GMRF(zeros(n), Q)
        end

        obs_model = ExponentialFamily(Normal)  # Uses σ hyperparameter
        model = INLAModel(spec, two_variance_latent, obs_model)

        # Use data that's not too extreme to avoid boundary modes
        y_test = [0.2, -0.1, 0.3, -0.2, 0.1, -0.15]

        # Get 2D posterior
        θ_star, mode_points, mode_logdensities = find_hyperparameter_mode(model, y_test)
        exploration = explore_hyperparameter_posterior(
            model, y_test, θ_star, GaussianMarginal(), 1:6;
            interpolation_subdivisions = 2
        )
        posterior_approx = build_posterior_interpolant(exploration)

        @test length(θ_star) == 2
        @test size(exploration.integration_bounds) == (2, 2)

        # Test marginal computation at mode
        test_val_1 = θ_star[1]
        test_val_2 = θ_star[2]

        marginal_1 = hyperparameter_marginal_logpdf(posterior_approx, 1, test_val_1)
        marginal_2 = hyperparameter_marginal_logpdf(posterior_approx, 2, test_val_2)

        @test isfinite(marginal_1)
        @test isfinite(marginal_2)

        # Test evaluation at joint mode
        joint_at_mode = posterior_approx(to_vector(θ_star, spec))
        @test isfinite(joint_at_mode)

        # Test that marginals are reasonable relative to joint
        # Marginal should be >= joint at mode (integration can only increase probability)
        @test marginal_1 >= joint_at_mode - 2.0  # Allow some numerical tolerance
        @test marginal_2 >= joint_at_mode - 2.0

        # Validate against numerical integration
        bounds = exploration.integration_bounds
        θ₂_grid = range(bounds[2, 1], bounds[2, 2], length = 100)
        joint_vals = [posterior_approx([test_val_1, θ₂]) for θ₂ in θ₂_grid]
        max_val = maximum(joint_vals)
        marginal_numerical = log(sum(exp.(joint_vals .- max_val)) * step(θ₂_grid)) + max_val
        @test marginal_1 ≈ marginal_numerical atol = 0.3
    end

    @testset "Marginal Integration Properties" begin
        # Test mathematical properties of marginal integration - use balanced data and informative priors
        spec = @hyperparams begin
            (μ ~ Normal(0, 0.5), transform = identity, space = working)
            (σ ~ InverseGamma(4, 3), transform = log, space = natural)
        end

        function location_scale_latent(; μ, σ, kwargs...)
            n = 5  # More data points
            Q = spdiagm(0 => fill(1 / σ^2 + 1.0e-6, n))  # Add small regularization
            return GMRF(fill(μ, n), Q)
        end

        obs_model = ExponentialFamily(Normal)
        model = INLAModel(spec, location_scale_latent, obs_model)

        # Use balanced data that should give interior mode
        y_test = [0.1, -0.05, 0.15, -0.08, 0.12]

        # Get 2D posterior
        θ_star, mode_points, mode_logdensities = find_hyperparameter_mode(model, y_test)
        exploration = explore_hyperparameter_posterior(
            model, y_test, θ_star, GaussianMarginal(), 1:3;
            interpolation_subdivisions = 2
        )
        posterior_approx = build_posterior_interpolant(exploration)

        # Test multiple evaluation points for each marginal
        μ_values = [θ_star[1] - 0.2, θ_star[1], θ_star[1] + 0.2]
        σ_values = [max(0.1, θ_star[2] - 0.2), θ_star[2], θ_star[2] + 0.2]

        # Test μ marginals
        μ_marginals = Float64[]
        for μ_val in μ_values
            marginal = hyperparameter_marginal_logpdf(posterior_approx, 1, μ_val)
            @test isfinite(marginal)
            push!(μ_marginals, marginal)
        end

        # Test σ marginals
        σ_marginals = Float64[]
        for σ_val in σ_values
            marginal = hyperparameter_marginal_logpdf(posterior_approx, 2, σ_val)
            @test isfinite(marginal)
            push!(σ_marginals, marginal)
        end

        # Test that marginals vary smoothly (no dramatic jumps)
        μ_diffs = diff(μ_marginals)
        σ_diffs = diff(σ_marginals)

        @test all(abs.(μ_diffs) .< 10.0)  # No extreme jumps
        @test all(abs.(σ_diffs) .< 10.0)
    end

    @testset "Bounds Checking and Error Handling" begin
        # Create stable 2D model using AR-1 GMRF pattern with proper regularization
        spec = @hyperparams begin
            (σ_gmrf ~ Gamma(2, 3), transform = log, space = natural)
            (ρ ~ Uniform(0, 0.5), transform = logit, space = natural)
            σ = 1.0e-6  # Fixed parameter
        end

        function ar_precision(ρ, k)
            return spdiagm(-1 => -ρ * ones(k - 1), 0 => ones(k), 1 => -ρ * ones(k - 1))
        end

        function stable_2d_latent(; σ_gmrf, ρ, kwargs...)
            k = 1000  # Large k for numerical stability
            Q = ar_precision(ρ, k) ./ σ_gmrf^2
            μ = zeros(k)
            return GMRF(μ, Q)
        end

        obs_model = ExponentialFamily(Normal)
        model = INLAModel(spec, stable_2d_latent, obs_model)

        # Generate stable test data using the same method as the working example
        σ_gmrf_true = 2.5
        ρ_true = 0.4
        x_gt = rand(stable_2d_latent(; σ_gmrf = σ_gmrf_true, ρ = ρ_true))
        y_test = rand(conditional_distribution(obs_model, x_gt; σ = spec.fixed.σ))

        # Get posterior
        θ_star, mode_points, mode_logdensities = find_hyperparameter_mode(model, y_test)
        exploration = explore_hyperparameter_posterior(model, y_test, θ_star, GaussianMarginal(), 1:1000)
        posterior_approx = build_posterior_interpolant(exploration)

        # Test bounds checking
        @test_throws BoundsError hyperparameter_marginal_logpdf(posterior_approx, 3, 1.0)  # dim > n_dims
        @test_throws BoundsError hyperparameter_marginal_logpdf(posterior_approx, 0, 1.0)  # dim < 1
        @test_throws BoundsError hyperparameter_marginal_logpdf(posterior_approx, -1, 1.0) # negative dim

        # Valid calls should work
        marginal_1 = hyperparameter_marginal_logpdf(posterior_approx, 1, θ_star[1])
        marginal_2 = hyperparameter_marginal_logpdf(posterior_approx, 2, θ_star[2])

        @test isfinite(marginal_1)
        @test isfinite(marginal_2)
    end

    @testset "Integration Tolerance Effects" begin
        # Test effect of integration tolerances using stable AR-1 GMRF pattern
        spec = @hyperparams begin
            (σ_gmrf ~ Gamma(2, 3), transform = log, space = natural)
            (ρ ~ Uniform(0, 0.5), transform = logit, space = natural)
            σ = 1.0e-6  # Fixed parameter
        end

        function ar_precision(ρ, k)
            return spdiagm(-1 => -ρ * ones(k - 1), 0 => ones(k), 1 => -ρ * ones(k - 1))
        end

        function tolerance_test_latent(; σ_gmrf, ρ, kwargs...)
            k = 1000  # Large k for numerical stability
            Q = ar_precision(ρ, k) ./ σ_gmrf^2
            μ = zeros(k)
            return GMRF(μ, Q)
        end

        obs_model = ExponentialFamily(Normal)  # Use Normal for stability
        model = INLAModel(spec, tolerance_test_latent, obs_model)

        # Generate stable test data using the same method as the working example
        σ_gmrf_true = 2.5
        ρ_true = 0.4
        x_gt = rand(tolerance_test_latent(; σ_gmrf = σ_gmrf_true, ρ = ρ_true))
        y_test = rand(conditional_distribution(obs_model, x_gt; σ = spec.fixed.σ))

        # Get posterior
        θ_star, mode_points, mode_logdensities = find_hyperparameter_mode(model, y_test)
        exploration = explore_hyperparameter_posterior(model, y_test, θ_star, GaussianMarginal(), 1:1000)
        posterior_approx = build_posterior_interpolant(exploration)

        # Test with different tolerances
        test_val = θ_star[1]

        # Loose tolerance
        marginal_loose = hyperparameter_marginal_logpdf(posterior_approx, 1, test_val; rtol = 1.0e-3, atol = 1.0e-6)

        # Tight tolerance
        marginal_tight = hyperparameter_marginal_logpdf(posterior_approx, 1, test_val; rtol = 1.0e-6, atol = 1.0e-10)

        @test isfinite(marginal_loose)
        @test isfinite(marginal_tight)

        # Results should be similar but tight tolerance should be more accurate
        @test abs(marginal_loose - marginal_tight) < 0.1
    end

end
