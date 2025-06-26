using Test
using IntegratedNestedLaplace
using GaussianMarkovRandomFields
using LDLFactorizations
using Distributions
using LinearAlgebra
using SparseArrays

@testset "INLAModel" begin
    
    @testset "Construction and Validation" begin
        # Set up components
        hp_prior = HyperparameterPrior((σ = InverseGamma(2, 1),))
        
        function latent_gmrf(θ_named)
            σ = θ_named.σ
            n = 10
            Q = spdiagm(0 => fill(1/σ^2, n))
            μ = zeros(n)
            return GMRF(μ, Q, CholeskySolverBlueprint())
        end
        
        obs_model = ExponentialFamily(Normal)  # Requires σ hyperparameter
        
        # Test successful construction
        model = INLAModel(hp_prior, latent_gmrf, obs_model)
        @test model.hyperparameter_prior == hp_prior
        @test model.latent_prior == latent_gmrf
        @test model.observation_model == obs_model
        
        # Test type parameters
        @test model isa INLAModel{typeof(hp_prior), typeof(latent_gmrf), typeof(obs_model)}
    end
    
    @testset "Parameter Validation" begin
        # Missing required hyperparameter
        hp_prior_incomplete = HyperparameterPrior((μ = Normal(0, 1),))  # Missing σ
        
        function latent_gmrf(θ_named)
            n = 5
            Q = spdiagm(0 => ones(n))
            return GMRF(zeros(n), Q, CholeskySolverBlueprint())
        end
        
        obs_model = ExponentialFamily(Normal)  # Requires σ
        
        # Should error due to missing σ
        @test_throws ErrorException INLAModel(hp_prior_incomplete, latent_gmrf, obs_model)
    end
    
    @testset "latent_gmrf Function" begin
        hp_prior = HyperparameterPrior((σ = InverseGamma(2, 1),))
        
        function latent_gmrf_func(θ_named)
            σ = θ_named.σ
            n = 8
            Q = spdiagm(0 => fill(1/σ^2, n))
            return GMRF(zeros(n), Q, CholeskySolverBlueprint())
        end
        
        obs_model = ExponentialFamily(Normal)
        model = INLAModel(hp_prior, latent_gmrf_func, obs_model)
        
        # Test latent GMRF generation
        θ_named = (σ = 2.0,)
        gmrf = latent_gmrf(model, θ_named)
        
        @test gmrf isa GMRF
        @test length(mean(gmrf)) == 8
        @test all(mean(gmrf) .== 0)
        
        # Test different hyperparameter values
        θ_named2 = (σ = 0.5,)
        gmrf2 = latent_gmrf(model, θ_named2)
        
        # Different σ should give different precision matrices
        @test precision_matrix(gmrf) != precision_matrix(gmrf2)
    end
    
    @testset "log_joint_density" begin
        # Set up model
        hp_prior = HyperparameterPrior((σ = InverseGamma(2, 1),))
        
        function latent_gmrf_func(θ_named)
            σ = θ_named.σ
            n = 6
            Q = spdiagm(0 => fill(1/σ^2, n))
            return GMRF(zeros(n), Q, CholeskySolverBlueprint())
        end
        
        obs_model = ExponentialFamily(Normal)
        model = INLAModel(hp_prior, latent_gmrf_func, obs_model)
        
        # Test data
        θ = [1.5]  # σ² = 1.5
        x = randn(6)
        y = x + 0.1 * randn(6)  # Noisy observations
        
        # Test joint density evaluation
        log_joint = log_joint_density(model, x, θ, y)
        @test isa(log_joint, Real)
        @test isfinite(log_joint)
        
        # Test that different parameters give different densities
        θ2 = [0.8]
        log_joint2 = log_joint_density(model, x, θ2, y)
        @test log_joint != log_joint2
        
        # Test with different latent field
        x2 = 2 * x
        log_joint3 = log_joint_density(model, x2, θ, y)
        @test log_joint != log_joint3
    end
    
    @testset "Multiple Hyperparameters" begin
        # Model with multiple hyperparameters
        hp_prior = HyperparameterPrior((
            σ_latent = InverseGamma(2, 1),
            σ = InverseGamma(2, 1)
        ))
        
        function latent_gmrf_func(θ_named)
            σ_latent = θ_named.σ_latent
            n = 5
            Q = spdiagm(0 => fill(1/σ_latent^2, n))
            return GMRF(zeros(n), Q, CholeskySolverBlueprint())
        end
        
        obs_model = ExponentialFamily(Normal)  # Uses σ
        
        # Test construction with parameter name matching
        model = INLAModel(hp_prior, latent_gmrf_func, obs_model)
        
        # Test joint density with multiple parameters
        θ = [1.2, 0.8]  # [σ_latent, σ]
        x = randn(5)
        y = x + 0.1 * randn(5)
        
        log_joint = log_joint_density(model, x, θ, y)
        @test isfinite(log_joint)
    end
    
    @testset "Different Observation Models" begin
        # Test with Bernoulli observation model (no hyperparameters)
        hp_prior_simple = HyperparameterPrior((τ = Gamma(2, 1),))
        
        function ar1_latent(θ_named)
            τ = θ_named.τ
            n = 8
            # AR(1) precision matrix
            ϕ = 0.7
            diag_main = [τ; fill(τ * (1 + ϕ^2), n-2); τ]
            diag_off = fill(-τ * ϕ, n-1)
            Q = spdiagm(0 => diag_main, -1 => diag_off, 1 => diag_off)
            return GMRF(zeros(n), Q, CholeskySolverBlueprint())
        end
        
        obs_model_bernoulli = ExponentialFamily(Bernoulli)
        model_bernoulli = INLAModel(hp_prior_simple, ar1_latent, obs_model_bernoulli)
        
        # Test with binary data
        θ = [2.0]  # τ = 2.0
        x = randn(8)
        y = rand(8) .> 0.5  # Binary data
        
        log_joint = log_joint_density(model_bernoulli, x, θ, y)
        @test isfinite(log_joint)
    end
    
    @testset "Type Stability" begin
        hp_prior = HyperparameterPrior((σ = InverseGamma(2, 1),))
        
        function latent_gmrf_func(θ_named)
            σ = θ_named.σ
            n = 4
            Q = spdiagm(0 => fill(1/σ^2, n))
            return GMRF(zeros(n), Q, CholeskySolverBlueprint())
        end
        
        obs_model = ExponentialFamily(Normal)
        model = INLAModel(hp_prior, latent_gmrf_func, obs_model)
        
        θ = [1.0]
        x = randn(4)
        y = randn(4)
        θ_named = (σ = 1.0,)
        
        # Test type stability
        @inferred Float64 log_joint_density(model, x, θ, y)
        @inferred GMRF latent_gmrf(model, θ_named)
    end
    
    @testset "Pretty Printing" begin
        hp_prior = HyperparameterPrior((σ = InverseGamma(2, 1),))
        
        function latent_gmrf_func(θ_named)
            σ = θ_named.σ
            Q = spdiagm(0 => fill(1/σ^2, 3))
            return GMRF(zeros(3), Q, CholeskySolverBlueprint())
        end
        
        obs_model = ExponentialFamily(Normal)
        model = INLAModel(hp_prior, latent_gmrf_func, obs_model)
        
        # Test that show doesn't error
        str = string(model)
        @test occursin("INLAModel", str)
        @test occursin("Hyperparameter prior", str)
        @test occursin("Observation model", str)
    end
    
    @testset "Integration with Mixed Parameters" begin
        # Test with both free and fixed hyperparameters
        hp_prior = HyperparameterPrior(
            (σ = InverseGamma(2, 1),);  # Free parameter
            fixed = (df = 3.0,)        # Fixed parameter
        )
        
        function latent_gmrf_func(θ_named)
            σ = θ_named.σ
            # df = θ_named.df  # Could use fixed parameter if needed
            n = 6
            Q = spdiagm(0 => fill(1/σ^2, n))
            return GMRF(zeros(n), Q, CholeskySolverBlueprint())
        end
        
        # Custom observation model that uses both parameters
        struct TestObsModel <: ObservationModel end
        IntegratedNestedLaplace.hyperparameters(::TestObsModel) = (:σ, :df)
        IntegratedNestedLaplace.loglik(::TestObsModel, x, θ_named, y) = -0.5 * sum((y - x).^2) / θ_named.σ^2 - length(y) * log(θ_named.σ) / 2
        
        obs_model = TestObsModel()
        model = INLAModel(hp_prior, latent_gmrf_func, obs_model)
        
        θ = [1.5]  # Only free parameter
        x = randn(6)
        y = x + 0.1 * randn(6)
        
        log_joint = log_joint_density(model, x, θ, y)
        @test isfinite(log_joint)
    end
    
end
