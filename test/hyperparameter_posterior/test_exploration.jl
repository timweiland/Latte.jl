using Test
using IntegratedNestedLaplace
using GaussianMarkovRandomFields
using LDLFactorizations
using Distributions
using LinearAlgebra
using SparseArrays
using FiniteDiff

@testset "Posterior Exploration" begin

    @testset "Reparameterization Computation" begin
        # Test reparameterization around mode
        spec = @hyperparams begin
            (ρ ~ Beta(3, 3), transform = logit, space = natural)
        end

        function correlation_latent(; ρ, kwargs...)
            n = 5
            # AR(1)-like structure with correlation ρ
            Q_diag = fill(1.0, n)
            Q_off = fill(-ρ, n - 1)
            Q = spdiagm(0 => Q_diag, 1 => Q_off, -1 => Q_off)
            Q[1, 1] = 1.0; Q[n, n] = 1.0  # Boundary conditions
            return GMRF(zeros(n), Symmetric(Q))
        end

        obs_model = ExponentialFamily(Bernoulli)
        model = INLAModel(spec, correlation_latent, obs_model)

        y_test = [true, false, true, true, false]

        # Find mode
        θ_star, _, _ = find_hyperparameter_mode(model, y_test)

        # Compute reparameterization (convert NamedTuple to vector, staying in natural space)
        θ_star_vec = to_vector(θ_star, model.hyperparameter_spec)
        transform = IntegratedNestedLaplace.compute_reparameterization(model, y_test, θ_star_vec)

        @test size(transform.H) == (1, 1)
        @test size(transform.V) == (1, 1)
        @test size(transform.Λ_inv_sqrt) == (1, 1)

        # H should be positive definite (negative Hessian of log-density)
        @test transform.H[1, 1] > 0  # Since H = -∇²log π(θ|y)

        # V should be orthogonal (eigenvalue decomposition)
        @test transform.V' * transform.V ≈ I(1) atol = 1.0e-10

        # Λ_inv_sqrt should be positive
        @test transform.Λ_inv_sqrt[1, 1] > 0
    end

    @testset "1D Exploration" begin
        # Test exploration around mode for 1D case
        spec = @hyperparams begin
            (σ ~ InverseGamma(2, 1), transform = log, space = natural)
        end

        function variance_latent(; σ, kwargs...)
            n = 6
            Q = spdiagm(0 => fill(1 / σ^2, n))
            return GMRF(zeros(n), Q)
        end

        obs_model = ExponentialFamily(Normal)
        model = INLAModel(spec, variance_latent, obs_model)

        y_test = [0.5, -0.2, 0.8, -0.1, 0.3, -0.4]

        # Find mode
        θ_star, mode_points, mode_logdensities = find_hyperparameter_mode(model, y_test)

        # Explore around mode
        exploration = explore_hyperparameter_posterior(
            model, y_test, θ_star, GaussianMarginal(), 1:6;
            integration_step_z = 2.0, interpolation_subdivisions = 2
        )

        # Find mode (point with highest log density)
        max_idx = argmax([point.log_density for point in exploration.grid_points])
        mode_point = exploration.grid_points[max_idx]
        # Convert θ_star to vector for comparison (exploration stores vectors in natural space)
        θ_star_vec = to_vector(θ_star, spec)
        @test mode_point.θ ≈ θ_star_vec atol = 1.0e-10

        @test length(exploration.grid_points) >= 3  # Should have multiple points
        @test all(isfinite, [point.log_density for point in exploration.grid_points])

        # Integration indices should be subset of all points
        @test all(idx -> 1 <= idx <= length(exploration.grid_points), exploration.integration_indices)
        @test length(exploration.integration_indices) > 0

        # Mode should be included in integration points
        mode_found = false
        for idx in exploration.integration_indices
            if exploration.grid_points[idx].θ ≈ to_vector(θ_star, spec)
                mode_found = true
                break
            end
        end
        @test mode_found

        # Integration bounds should encompass all integration points
        @test size(exploration.integration_bounds) == (1, 2)
        integration_values = [exploration.grid_points[idx].θ[1] for idx in exploration.integration_indices]
        @test exploration.integration_bounds[1, 1] <= minimum(integration_values)
        @test exploration.integration_bounds[1, 2] >= maximum(integration_values)
    end

    @testset "2D Exploration" begin
        # Test 2D case with two hyperparameters
        spec = @hyperparams begin
            (σ_latent ~ InverseGamma(2, 1), transform = log, space = natural)
            (σ ~ InverseGamma(2, 1), transform = log, space = natural)
        end

        function two_variance_latent(; σ_latent, kwargs...)
            n = 4
            Q = spdiagm(0 => fill(1 / σ_latent^2, n))
            return GMRF(zeros(n), Q)
        end

        obs_model = ExponentialFamily(Normal)  # Uses σ
        model = INLAModel(spec, two_variance_latent, obs_model)

        y_test = [0.5, -0.3, 0.8, -0.2]

        # Get 2D posterior
        θ_star, mode_points, mode_logdensities = find_hyperparameter_mode(model, y_test)
        exploration = explore_hyperparameter_posterior(
            model, y_test, θ_star, GaussianMarginal(), 1:4;
            interpolation_subdivisions = 2
        )

        @test length(θ_star) == 2
        @test size(exploration.integration_bounds) == (2, 2)

        # θ_star is now in working space (log scale), can be any real values
        @test all(isfinite, θ_star)

        # Test exploration structure
        @test length(exploration.grid_points) > 10  # Should have multiple points
        @test all(length(point.θ) == 2 for point in exploration.grid_points)  # All points are 2D
        @test all(isfinite, [point.log_density for point in exploration.grid_points])

        # Integration bounds should make sense
        for dim in 1:2
            integration_dim_values = [exploration.grid_points[idx].θ[dim] for idx in exploration.integration_indices]
            @test exploration.integration_bounds[dim, 1] <= minimum(integration_dim_values)
            @test exploration.integration_bounds[dim, 2] >= maximum(integration_dim_values)
        end
    end

end
