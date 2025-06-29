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
        hp_prior = HyperparameterPrior((ρ = Beta(3, 3),))
        
        function correlation_latent(θ_named)
            ρ = θ_named.ρ
            n = 5
            # AR(1)-like structure with correlation ρ
            Q_diag = fill(1.0, n)
            Q_off = fill(-ρ, n-1)
            Q = spdiagm(0 => Q_diag, 1 => Q_off, -1 => Q_off)
            Q[1,1] = 1.0; Q[n,n] = 1.0  # Boundary conditions
            return GMRF(zeros(n), Symmetric(Q), CholeskySolverBlueprint())
        end
        
        obs_model = ExponentialFamily(Bernoulli)
        model = INLAModel(hp_prior, correlation_latent, obs_model)
        
        y_test = [true, false, true, true, false]
        
        # Find mode
        θ_star, _, _ = find_hyperparameter_mode(model, y_test)
        
        # Compute reparameterization
        H, V, Λ_inv_sqrt, mode_logpdf = IntegratedNestedLaplace.compute_reparameterization(model, y_test, θ_star)
        
        @test size(H) == (1, 1)
        @test size(V) == (1, 1)
        @test size(Λ_inv_sqrt) == (1, 1)
        @test isfinite(mode_logpdf)
        
        # H should be negative definite (negative Hessian of log-density)
        @test H[1,1] > 0  # Since H = -∇²log π(θ|y)
        
        # V should be orthogonal (eigenvalue decomposition)
        @test V' * V ≈ I(1) atol=1e-10
        
        # Λ_inv_sqrt should be positive
        @test Λ_inv_sqrt[1,1] > 0
    end
    
    @testset "1D Exploration" begin
        # Test exploration around mode for 1D case
        hp_prior = HyperparameterPrior((σ = InverseGamma(2, 1),))
        
        function variance_latent(θ_named)
            σ = θ_named.σ
            n = 6
            Q = spdiagm(0 => fill(1/σ^2, n))
            return GMRF(zeros(n), Q, CholeskySolverBlueprint())
        end
        
        obs_model = ExponentialFamily(Normal)
        model = INLAModel(hp_prior, variance_latent, obs_model)
        
        y_test = [0.5, -0.2, 0.8, -0.1, 0.3, -0.4]
        
        # Find mode
        θ_star, mode_points, mode_logdensities = find_hyperparameter_mode(model, y_test)
        
        # Explore around mode
        exploration = explore_hyperparameter_posterior(model, y_test, θ_star, mode_points, mode_logdensities;
                                                      δ_π=2.0, interpolation_factor=2)
        
        @test exploration.mode == θ_star
        @test length(exploration.interpolation_points) > 5  # Should have multiple points
        @test length(exploration.log_densities) == length(exploration.interpolation_points)
        @test all(isfinite, exploration.log_densities)
        
        # Integration indices should be subset of all points
        @test all(idx -> 1 <= idx <= length(exploration.interpolation_points), exploration.integration_indices)
        @test length(exploration.integration_indices) > 0
        
        # Mode should be included in integration points
        mode_found = false
        for idx in exploration.integration_indices
            if exploration.interpolation_points[idx] ≈ θ_star
                mode_found = true
                break
            end
        end
        @test mode_found
        
        # Integration bounds should encompass all points
        @test size(exploration.integration_bounds) == (1, 2)
        all_values = [p[1] for p in exploration.interpolation_points]
        @test exploration.integration_bounds[1, 1] <= minimum(all_values)
        @test exploration.integration_bounds[1, 2] >= maximum(all_values)
    end
    
    @testset "2D Exploration" begin
        # Test 2D case with two hyperparameters
        hp_prior = HyperparameterPrior((σ_latent = InverseGamma(2, 1), σ = InverseGamma(2, 1)))
        
        function two_variance_latent(θ_named)
            σ_latent = θ_named.σ_latent
            n = 4
            Q = spdiagm(0 => fill(1/σ_latent^2, n))
            return GMRF(zeros(n), Q, CholeskySolverBlueprint())
        end
        
        obs_model = ExponentialFamily(Normal)  # Uses σ
        model = INLAModel(hp_prior, two_variance_latent, obs_model)
        
        y_test = [0.5, -0.3, 0.8, -0.2]
        
        # Get 2D posterior
        θ_star, mode_points, mode_logdensities = find_hyperparameter_mode(model, y_test)
        exploration = explore_hyperparameter_posterior(model, y_test, θ_star, mode_points, mode_logdensities;
                                                      interpolation_factor=2)
        
        @test length(θ_star) == 2
        @test size(exploration.integration_bounds) == (2, 2)
        
        # Test that mode is reasonable
        @test all(θ_star .> 0)  # Should be in support
        
        # Test exploration structure
        @test length(exploration.interpolation_points) > 10  # Should have multiple points
        @test all(length(p) == 2 for p in exploration.interpolation_points)  # All points are 2D
        @test length(exploration.log_densities) == length(exploration.interpolation_points)
        @test all(isfinite, exploration.log_densities)
        
        # Integration bounds should make sense
        for dim in 1:2
            all_dim_values = [p[dim] for p in exploration.interpolation_points]
            @test exploration.integration_bounds[dim, 1] <= minimum(all_dim_values)
            @test exploration.integration_bounds[dim, 2] >= maximum(all_dim_values)
        end
    end
    
end