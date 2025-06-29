using Test
using IntegratedNestedLaplace
using GaussianMarkovRandomFields
using LDLFactorizations
using Distributions
using LinearAlgebra
using SparseArrays

@testset "Type Stability and Performance" begin
    
    @testset "Type Stability" begin
        hp_prior = HyperparameterPrior((τ_scale = Gamma(3, 2),))
        
        function beta_latent(θ_named)
            τ_scale = θ_named.τ_scale
            n = 2
            Q = spdiagm(0 => fill(τ_scale, n))
            return GMRF(zeros(n), Q, CholeskySolverBlueprint())
        end
        
        obs_model = ExponentialFamily(Bernoulli)
        model = INLAModel(hp_prior, beta_latent, obs_model)
        
        y_test = [true, true]  # Biased data to avoid boundary issues
        θ_test = [1.0]
        
        # Test type stability of key functions
        @inferred Float64 hyperparameter_logpdf(model, θ_test, y_test)
        
        θ_star, _, _ = find_hyperparameter_mode(model, y_test; collect_points=false)
        @inferred Vector{Float64} find_hyperparameter_mode(model, y_test; collect_points=false)[1]
        
        # Test mode computation from Product distribution
        prior_product = product_distribution([Gamma(3, 2)])
        @inferred Vector{Float64} mode(prior_product)
    end
    
    @testset "Memory Allocation" begin
        hp_prior = HyperparameterPrior((σ = InverseGamma(4, 3),))
        
        function allocation_test_latent(θ_named)
            σ = θ_named.σ
            n = 5
            Q = spdiagm(0 => fill(1/σ^2, n))
            return GMRF(zeros(n), Q, CholeskySolverBlueprint())
        end
        
        obs_model = ExponentialFamily(Normal)
        model = INLAModel(hp_prior, allocation_test_latent, obs_model)
        
        y_test = randn(5)
        θ_test = [1.5]
        
        # Warm up
        hyperparameter_logpdf(model, θ_test, y_test)
        
        # Test that repeated calls don't allocate excessively
        initial_memory = Base.gc_bytes()
        for i in 1:10
            hyperparameter_logpdf(model, θ_test, y_test)
        end
        final_memory = Base.gc_bytes()
        
        # Should not allocate too much memory for repeated evaluations
        memory_increase = final_memory - initial_memory
        @test memory_increase < 1_000_000  # Less than 1MB for 10 evaluations
    end
    
    @testset "Dimensional Scaling" begin
        # Test that algorithms work correctly across different dimensions
        
        # 1D case
        hp_prior_1d = HyperparameterPrior((τ = Gamma(2, 1),))
        
        function latent_1d(θ_named)
            τ = θ_named.τ
            n = 3
            Q = spdiagm(0 => fill(τ, n))
            return GMRF(zeros(n), Q, CholeskySolverBlueprint())
        end
        
        obs_model = ExponentialFamily(Bernoulli)
        model_1d = INLAModel(hp_prior_1d, latent_1d, obs_model)
        y_test_1d = [true, false, true]
        
        θ_star_1d, _, _ = find_hyperparameter_mode(model_1d, y_test_1d)
        @test length(θ_star_1d) == 1
        
        # 2D case
        hp_prior_2d = HyperparameterPrior((α = Gamma(2, 1), β = Gamma(2, 1)))
        
        function latent_2d(θ_named)
            α, β = θ_named.α, θ_named.β
            n = 4
            Q = spdiagm(0 => [α, α, β, β])
            return GMRF(zeros(n), Q, CholeskySolverBlueprint())
        end
        
        model_2d = INLAModel(hp_prior_2d, latent_2d, obs_model)
        y_test_2d = [true, false, true, false]
        
        θ_star_2d, _, _ = find_hyperparameter_mode(model_2d, y_test_2d)
        @test length(θ_star_2d) == 2
        
        # 3D case
        hp_prior_3d = HyperparameterPrior((γ₁ = Gamma(2, 1), γ₂ = Gamma(2, 1), γ₃ = Gamma(2, 1)))
        
        function latent_3d(θ_named)
            γ₁, γ₂, γ₃ = θ_named.γ₁, θ_named.γ₂, θ_named.γ₃
            n = 6
            Q = spdiagm(0 => [γ₁, γ₁, γ₂, γ₂, γ₃, γ₃])
            return GMRF(zeros(n), Q, CholeskySolverBlueprint())
        end
        
        model_3d = INLAModel(hp_prior_3d, latent_3d, obs_model)
        y_test_3d = [true, false, true, false, true, false]
        
        θ_star_3d, _, _ = find_hyperparameter_mode(model_3d, y_test_3d)
        @test length(θ_star_3d) == 3
        
        # All should be in valid range
        @test all(θ_star_1d .> 0)
        @test all(θ_star_2d .> 0)  
        @test all(θ_star_3d .> 0)
    end
    
    @testset "Numerical Stability" begin
        # Test with extreme parameter values
        hp_prior = HyperparameterPrior((σ = InverseGamma(0.1, 0.1),))  # Very peaked prior
        
        function extreme_latent(θ_named)
            σ = θ_named.σ
            n = 3
            Q = spdiagm(0 => fill(1/σ^2, n))
            return GMRF(zeros(n), Q, CholeskySolverBlueprint())
        end
        
        obs_model = ExponentialFamily(Normal)
        model = INLAModel(hp_prior, extreme_latent, obs_model)
        
        # Test with extreme data
        y_extreme_large = [10.0, 12.0, 15.0]  # Large values
        y_extreme_small = [0.001, 0.002, 0.003]  # Small values
        
        # Should handle extreme cases without crashing
        θ_star_large, _, _ = find_hyperparameter_mode(model, y_extreme_large)
        θ_star_small, _, _ = find_hyperparameter_mode(model, y_extreme_small)
        
        @test isfinite(θ_star_large[1])
        @test isfinite(θ_star_small[1])
        @test θ_star_large[1] > 0
        @test θ_star_small[1] > 0
        
        # The different data should lead to different posterior modes
        @test θ_star_large[1] != θ_star_small[1]
    end
    
    @testset "Consistency Across Runs" begin
        # Test that results are consistent across multiple runs
        hp_prior = HyperparameterPrior((ρ = Beta(2, 2),))
        
        function consistent_latent(θ_named)
            ρ = θ_named.ρ
            n = 4
            Q = spdiagm(0 => ones(n), 1 => fill(-ρ, n-1), -1 => fill(-ρ, n-1))
            Q[1,1] = 1+ρ^2; Q[n,n] = 1+ρ^2
            for i in 2:n-1
                Q[i,i] = 1+2*ρ^2
            end
            return GMRF(zeros(n), Symmetric(Q), CholeskySolverBlueprint())
        end
        
        obs_model = ExponentialFamily(Bernoulli)
        model = INLAModel(hp_prior, consistent_latent, obs_model)
        
        y_test = [true, false, true, false]
        
        # Run multiple times
        modes = Vector{Float64}[]
        for i in 1:3
            θ_star, _, _ = find_hyperparameter_mode(model, y_test)
            push!(modes, θ_star)
        end
        
        # All runs should give same result (within numerical tolerance)
        for i in 2:length(modes)
            @test modes[i] ≈ modes[1] atol=1e-6
        end
    end
    
end