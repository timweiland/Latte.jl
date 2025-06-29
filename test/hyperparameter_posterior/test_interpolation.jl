using Test
using IntegratedNestedLaplace
using GaussianMarkovRandomFields
using LDLFactorizations
using Distributions
using LinearAlgebra
using SparseArrays

@testset "Interpolation" begin
    
    @testset "1D Interpolation" begin
        # Test building interpolant for 1D case
        hp_prior = HyperparameterPrior((τ = Gamma(3, 1),))
        
        function precision_latent(θ_named)
            τ = θ_named.τ
            n = 4
            Q = spdiagm(0 => fill(τ, n))
            return GMRF(zeros(n), Q, CholeskySolverBlueprint())
        end
        
        obs_model = ExponentialFamily(Bernoulli)
        model = INLAModel(hp_prior, precision_latent, obs_model)
        
        y_test = [true, false, true, false]
        
        # Get exploration
        θ_star, mode_points, mode_logdensities = find_hyperparameter_mode(model, y_test)
        exploration = explore_hyperparameter_posterior(model, y_test, θ_star, mode_points, mode_logdensities)
        
        # Build interpolant
        posterior_approx = build_posterior_interpolant(exploration)
        
        @test posterior_approx.exploration == exploration
        
        # Test interpolant evaluation at exploration points
        for (i, θ_point) in enumerate(exploration.interpolation_points)
            approx_value = posterior_approx(θ_point)
            true_value = exploration.log_densities[i]
            @test approx_value ≈ true_value atol=1e-10  # Should exactly interpolate
        end
        
        # Test interpolant evaluation at mode
        mode_approx = posterior_approx(θ_star)
        @test isfinite(mode_approx)
        
        # Test scalar input handling
        mode_scalar = posterior_approx(θ_star[1])
        @test mode_scalar ≈ mode_approx atol=1e-10
    end
    
    @testset "2D Interpolation" begin
        # Test building interpolant for 2D case
        hp_prior = HyperparameterPrior((α = Gamma(2, 1), β = Gamma(2, 1)))
        
        function dual_precision_latent(θ_named)
            α, β = θ_named.α, θ_named.β
            n = 4
            # Different precisions for different components
            Q_diag = [α, α, β, β]
            Q = spdiagm(0 => Q_diag)
            return GMRF(zeros(n), Q, CholeskySolverBlueprint())
        end
        
        obs_model = ExponentialFamily(Bernoulli)
        model = INLAModel(hp_prior, dual_precision_latent, obs_model)
        
        y_test = [true, false, true, false]
        
        # Get exploration
        θ_star, mode_points, mode_logdensities = find_hyperparameter_mode(model, y_test)
        exploration = explore_hyperparameter_posterior(model, y_test, θ_star, mode_points, mode_logdensities;
                                                      interpolation_factor=2)
        
        # Build interpolant
        posterior_approx = build_posterior_interpolant(exploration)
        
        @test posterior_approx.exploration == exploration
        
        # Test interpolant evaluation at exploration points
        tolerance = 1e-6  # RBF interpolation may have small numerical errors
        for (i, θ_point) in enumerate(exploration.interpolation_points)
            approx_value = posterior_approx(θ_point)
            true_value = exploration.log_densities[i]
            @test approx_value ≈ true_value atol=tolerance
        end
        
        # Test interpolant evaluation at mode
        mode_approx = posterior_approx(θ_star)
        @test isfinite(mode_approx)

        # Test interpolation at new points near the mode
        log_normalization = exploration.transformation.log_normalization
        
        # Generate test points around the mode within integration bounds
        mode_α, mode_β = θ_star[1], θ_star[2]
        bound_min_α, bound_max_α = exploration.integration_bounds[1, 1], exploration.integration_bounds[1, 2]
        bound_min_β, bound_max_β = exploration.integration_bounds[2, 1], exploration.integration_bounds[2, 2]
        
        # Create points at moderate distances from mode within bounds
        test_points = [
            [mode_α + 0.3 * (bound_max_α - mode_α), mode_β + 0.2 * (bound_max_β - mode_β)],
            [mode_α - 0.2 * (mode_α - bound_min_α), mode_β - 0.3 * (mode_β - bound_min_β)],
            [mode_α + 0.1 * (bound_max_α - mode_α), mode_β - 0.1 * (mode_β - bound_min_β)],
            [mode_α - 0.1 * (mode_α - bound_min_α), mode_β + 0.1 * (bound_max_β - mode_β)]
        ]
        
        for θ_new in test_points
            # Get interpolated value (already normalized)
            approx_value = posterior_approx(θ_new)
            
            # Get direct unnormalized value and normalize it
            direct_unnormalized = hyperparameter_logpdf(model, θ_new, y_test)
            direct_normalized = direct_unnormalized - log_normalization
            
            # Should be reasonably close for 2D interpolation
            @test approx_value ≈ direct_normalized atol=0.2  # More tolerance for 2D
        end
    end
    
    
    @testset "Interpolation Quality" begin
        # Test interpolation quality - compare interpolated values with stored exploration values
        hp_prior = HyperparameterPrior((x = Normal(0, 1),))
        
        function smooth_latent(θ_named)
            x = θ_named.x
            n = 2
            Q = spdiagm(0 => fill(exp(x), n))  # Smooth function of x
            return GMRF(zeros(n), Q, CholeskySolverBlueprint())
        end
        
        obs_model = ExponentialFamily(Bernoulli)
        model = INLAModel(hp_prior, smooth_latent, obs_model)
        
        y_test = [true, false]
        
        # Get dense exploration
        θ_star, mode_points, mode_logdensities = find_hyperparameter_mode(model, y_test)
        exploration = explore_hyperparameter_posterior(model, y_test, θ_star, mode_points, mode_logdensities;
                                                      δ_π=3.0, interpolation_factor=1)  # Dense sampling
        
        posterior_approx = build_posterior_interpolant(exploration)
        
        # Test interpolation quality: compare interpolated values with stored normalized values
        test_indices = 1:min(5, length(exploration.interpolation_points))
        
        for i in test_indices
            θ_point = exploration.interpolation_points[i]
            approx_value = posterior_approx(θ_point)
            stored_value = exploration.log_densities[i]  # This is already normalized
            
            # Should be very close for points in the exploration (both normalized)
            @test approx_value ≈ stored_value atol=1e-3
        end
        
        # Test interpolation at new points near the mode
        log_normalization = exploration.transformation.log_normalization
        
        # Generate test points around the mode within integration bounds
        mode_val = θ_star[1]
        bound_min, bound_max = exploration.integration_bounds[1, 1], exploration.integration_bounds[1, 2]
        
        # Create points at 25%, 50%, 75% between mode and bounds
        test_points = [
            [mode_val + 0.25 * (bound_max - mode_val)],
            [mode_val - 0.25 * (mode_val - bound_min)],
            [mode_val + 0.5 * (bound_max - mode_val)],
            [mode_val - 0.5 * (mode_val - bound_min)]
        ]
        
        for θ_new in test_points
            # Get interpolated value (already normalized)
            approx_value = posterior_approx(θ_new)
            
            # Get direct unnormalized value and normalize it
            direct_unnormalized = hyperparameter_logpdf(model, θ_new, y_test)
            direct_normalized = direct_unnormalized - log_normalization
            
            # Should be reasonably close (interpolation accuracy)
            @test approx_value ≈ direct_normalized atol=0.1  # Reasonable tolerance for interpolation
        end
    end
    
end
