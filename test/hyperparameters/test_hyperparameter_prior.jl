using Test
using IntegratedNestedLaplace
using Distributions

@testset "Hyperparameter System" begin
    
    @testset "HyperparameterPrior Construction" begin
        
        @testset "Method 1: Foundational Constructor with Type Parameter" begin
            # Test valid construction with any distribution
            joint_dist = product_distribution([Gamma(2, 3), Uniform(0, 0.5)])
            
            hp_prior = HyperparameterPrior{(:σ, :ρ)}(joint_dist)
            
            @test hp_prior.free_distribution === joint_dist
            @test hp_prior.name_to_index == Dict(:σ => 1, :ρ => 2)
            @test isempty(hp_prior.fixed_values)
            
            # Test dimension validation
            @test_throws ErrorException HyperparameterPrior{(:σ,)}(joint_dist)  # Wrong dimension
            @test_throws ErrorException HyperparameterPrior{(:σ, :ρ, :μ)}(joint_dist)  # Wrong dimension
            
            # Test duplicate name validation
            @test_throws ErrorException HyperparameterPrior{(:σ, :σ)}(joint_dist)  # Duplicate names
        end
        
        @testset "Method 2: NamedTuple of Distributions" begin
            named_dists = (σ = Gamma(2, 3), ρ = Uniform(0, 0.5))
            hp_prior = HyperparameterPrior(named_dists)
            
            @test isa(hp_prior.free_distribution, Product)
            @test length(hp_prior.free_distribution) == 2
            @test hp_prior.name_to_index[:σ] == 1
            @test hp_prior.name_to_index[:ρ] == 2
            @test isempty(hp_prior.fixed_values)
            # Parameter names are now in the type: HyperparameterPrior{(:σ, :ρ), (:ρ, :σ), ...}
        end
        
        @testset "Single Parameter" begin
            hp_prior = HyperparameterPrior((μ = Normal(0, 1),))
            
            @test hp_prior.name_to_index[:μ] == 1
            # Type should be HyperparameterPrior{(:μ,), ...}
        end
        
        @testset "Empty Parameters" begin
            # Empty hyperparameter priors should throw an error
            @test_throws ErrorException HyperparameterPrior(NamedTuple())
        end
    end
    
    @testset "Hyperparameter Access Methods" begin
        hp_prior = HyperparameterPrior((σ = Gamma(2, 3), ρ = Uniform(0, 0.5), μ = Normal(0, 1)))
        θ = [2.5, 0.4, 1.2]
        
        @testset "get_hyperparameter" begin
            @test get_hyperparameter(θ, hp_prior, :σ) == 2.5
            @test get_hyperparameter(θ, hp_prior, :ρ) == 0.4
            @test get_hyperparameter(θ, hp_prior, :μ) == 1.2
            
            @test_throws KeyError get_hyperparameter(θ, hp_prior, :nonexistent)
        end
        
        @testset "set_hyperparameter!" begin
            θ_copy = copy(θ)
            set_hyperparameter!(θ_copy, hp_prior, :σ, 3.0)
            @test θ_copy[hp_prior.name_to_index[:σ]] == 3.0
            @test θ_copy[hp_prior.name_to_index[:ρ]] == 0.4  # unchanged
            
            @test_throws KeyError set_hyperparameter!(θ_copy, hp_prior, :nonexistent, 1.0)
        end
        
        @testset "to_named" begin
            named = to_named(θ, hp_prior)
            @test named.σ == 2.5
            @test named.ρ == 0.4
            @test named.μ == 1.2
            @test typeof(named) <: NamedTuple{(:μ, :ρ, :σ)}  # Parameters are sorted alphabetically
        end
        
        @testset "to_vector" begin
            named = (σ = 3.0, ρ = 0.3, μ = 0.5)
            θ_new = to_vector(named, hp_prior)
            @test θ_new[hp_prior.name_to_index[:σ]] == 3.0
            @test θ_new[hp_prior.name_to_index[:ρ]] == 0.3
            @test θ_new[hp_prior.name_to_index[:μ]] == 0.5
            
            # Test partial NamedTuple (should error)
            @test_throws KeyError to_vector((σ = 3.0,), hp_prior)
        end
        
        @testset "extract_hyperparameters" begin
            subset = extract_hyperparameters(θ, hp_prior, (:σ, :μ))
            @test subset.σ == 2.5
            @test subset.μ == 1.2
            @test !haskey(subset, :ρ)
            @test typeof(subset) <: NamedTuple{(:σ, :μ)}
            
            # Test empty extraction
            empty_subset = extract_hyperparameters(θ, hp_prior, ())
            @test isempty(empty_subset)
        end
        
        @testset "Round-trip conversion" begin
            θ_original = [1.5, 0.25, -0.5]
            named = to_named(θ_original, hp_prior)
            θ_reconstructed = to_vector(named, hp_prior)
            @test θ_original ≈ θ_reconstructed
        end
    end
    
    @testset "Observation Model Interface" begin
        
        @testset "Default hyperparameters" begin
            # Test that undefined models return empty tuple
            struct DummyModel <: ObservationModel end
            @test hyperparameters(DummyModel()) == ()
        end
        
        @testset "ExponentialFamily hyperparameters" begin
            @test hyperparameters(ExponentialFamily(Normal)) == (:σ,)
            @test hyperparameters(ExponentialFamily(Bernoulli)) == ()
            # Note: We'll need to add more when we implement other distributions
        end
    end
    
    @testset "INLAModel Construction and Validation" begin
        
        @testset "Valid construction" begin
            # Create valid hyperparameter prior for Normal observation model
            hp_prior = HyperparameterPrior((σ = Gamma(2, 3),))
            obs_model = ExponentialFamily(Normal)
            
            # Mock latent GMRF function
            latent_gmrf = (θ) -> nothing  # Dummy function for testing
            
            model = INLAModel(hp_prior, latent_gmrf, obs_model)
            @test model.hyperparameter_prior === hp_prior
            @test model.observation_model === obs_model
            @test model.latent_prior === latent_gmrf
        end
        
        @testset "Missing required hyperparameters" begin
            # Normal requires σ, but we don't provide it
            hp_prior = HyperparameterPrior((μ = Normal(0, 1),))
            obs_model = ExponentialFamily(Normal)
            latent_gmrf = (θ) -> nothing
            
            @test_throws ErrorException INLAModel(hp_prior, latent_gmrf, obs_model)
        end
        
        @testset "Unused hyperparameters warning" begin
            # Provide more hyperparameters than needed
            hp_prior = HyperparameterPrior((σ = Gamma(2, 3), μ = Normal(0, 1)))
            obs_model = ExponentialFamily(Normal)  # Only needs σ
            latent_gmrf = (θ) -> nothing
            
            # This should work but warn about unused μ
            @test_logs (:warn, r"Unused hyperparameters.*μ") INLAModel(hp_prior, latent_gmrf, obs_model)
        end
        
        @testset "No hyperparameters needed" begin
            # Bernoulli needs no hyperparameters, but we can have hyperparameters for latent GMRF
            hp_prior = HyperparameterPrior((dummy = Normal(0, 1),))
            obs_model = ExponentialFamily(Bernoulli)
            latent_gmrf = (θ) -> nothing
            
            model = INLAModel(hp_prior, latent_gmrf, obs_model)
            @test length(model.hyperparameter_prior.name_to_index) == 1
        end
    end
    
    @testset "Integration with existing Distribution methods" begin
        hp_prior = HyperparameterPrior((σ = Gamma(2, 3), ρ = Uniform(0, 0.5)))
        
        @testset "mode function" begin
            # Test that mode works with the underlying distribution
            mode_vec = mode(hp_prior.free_distribution)
            @test length(mode_vec) == 2
            @test mode_vec[1] ≈ mode(Gamma(2, 3))  # σ mode
            @test mode_vec[2] ≈ mode(Uniform(0, 0.5))  # ρ mode
        end
        
        @testset "rand and logpdf" begin
            # Test sampling and density evaluation
            θ_sample = rand(hp_prior.free_distribution)
            @test length(θ_sample) == 2
            
            θ_test = [1.5, 0.3]
            log_dens = logpdf(hp_prior.free_distribution, θ_test)
            @test isfinite(log_dens)
        end
    end
    
    @testset "Type stability and performance" begin
        hp_prior = HyperparameterPrior((σ = Gamma(2, 3), ρ = Uniform(0, 0.5)))
        θ = [2.5, 0.4]
        
        @testset "Type inference" begin
            # Test that key functions are type-stable
            @inferred get_hyperparameter(θ, hp_prior, :σ)
            @inferred to_named(θ, hp_prior)
            @inferred to_vector((σ = 2.5, ρ = 0.4), hp_prior)
            @inferred extract_hyperparameters(θ, hp_prior, Val{(:σ,)}())
        end
        
        @testset "NamedTuple properties" begin
            named = to_named(θ, hp_prior)
            
            # NamedTuples should be immutable and have efficient field access
            @test isbitstype(typeof(named))
            @test named.σ == 2.5
            @test named.ρ == 0.4
        end
    end
    
    @testset "Fixed Hyperparameters" begin
        
        @testset "Mixed free and fixed parameters" begin
            hp_prior = HyperparameterPrior(
                (ρ = Beta(1, 1), τ = Gamma(1, 1)),
                fixed = (σ = 0.5, μ = 0.0)
            )
            
            # Test structure
            @test isa(hp_prior.free_distribution, Product)
            @test length(hp_prior.free_distribution) == 2
            @test hp_prior.fixed_values == (σ = 0.5, μ = 0.0)
            @test hp_prior.name_to_index == Dict(:ρ => 1, :τ => 2)
            
            # Test to_named conversion  
            θ_free = [0.3, 1.2]
            θ_named = to_named(θ_free, hp_prior)
            @test θ_named.μ == 0.0  # Fixed
            @test θ_named.ρ == 0.3  # Free
            @test θ_named.σ == 0.5  # Fixed  
            @test θ_named.τ == 1.2  # Free
            
            # Test to_vector conversion (only free parameters)
            θ_free_back = to_vector(θ_named, hp_prior)
            @test θ_free_back == [0.3, 1.2]  # Only free parameters
            
            # Test get_hyperparameter works for both free and fixed
            @test get_hyperparameter(θ_free, hp_prior, :ρ) == 0.3  # Free
            @test get_hyperparameter(θ_free, hp_prior, :σ) == 0.5  # Fixed
            @test get_hyperparameter(θ_free, hp_prior, :τ) == 1.2  # Free
            @test get_hyperparameter(θ_free, hp_prior, :μ) == 0.0  # Fixed
            
            # Test set_hyperparameter! only works for free parameters
            θ_free_copy = copy(θ_free)
            set_hyperparameter!(θ_free_copy, hp_prior, :ρ, 0.7)
            @test θ_free_copy[1] == 0.7
            
            @test_throws ErrorException set_hyperparameter!(θ_free_copy, hp_prior, :σ, 1.0)  # Can't set fixed
        end
        
        @testset "Foundational constructor with fixed parameters" begin
            # Test correlated free parameters + fixed parameters
            corr_dist = MvNormal([0.0, 0.0], [1.0 0.5; 0.5 1.0])
            hp_prior = HyperparameterPrior{(:ρ, :τ)}(
                corr_dist,
                fixed = (σ = 0.5, μ = 0.0)
            )
            
            @test hp_prior.free_distribution === corr_dist
            @test hp_prior.fixed_values == (σ = 0.5, μ = 0.0)
            @test hp_prior.name_to_index == Dict(:ρ => 1, :τ => 2)
            
            # Test to_named with correlated parameters
            θ_free = [0.2, 1.1]
            θ_named = to_named(θ_free, hp_prior)
            @test θ_named == (μ = 0.0, ρ = 0.2, σ = 0.5, τ = 1.1)
        end
        
        @testset "Error cases for fixed parameters" begin
            # Cannot have zero free parameters
            @test_throws ErrorException HyperparameterPrior(
                NamedTuple(),
                fixed = (σ = 0.5, ρ = 0.3)
            )
            
            # Cannot have parameter in both free and fixed
            @test_throws ErrorException HyperparameterPrior(
                (σ = Gamma(2, 0.5), ρ = Beta(1, 1)),
                fixed = (σ = 0.5,)
            )
            
            # Foundational constructor: cannot have parameter in both free and fixed
            @test_throws ErrorException HyperparameterPrior{(:σ, :ρ)}(
                product_distribution([Gamma(2, 3), Uniform(0, 0.5)]),
                fixed = (σ = 0.5,)
            )
        end
        
        @testset "INLAModel integration with fixed parameters" begin
            # Test that INLAModel validation works with fixed parameters
            hp_prior = HyperparameterPrior(
                (ρ = Beta(1, 1),),        # Free parameter
                fixed = (σ = 0.5,)        # Fixed σ for Normal model
            )
            
            obs_model = ExponentialFamily(Normal)
            latent_gmrf = (θ_named) -> nothing
            
            # Should work - σ is provided (even though fixed)
            model = INLAModel(hp_prior, latent_gmrf, obs_model)
            @test model.hyperparameter_prior === hp_prior
            
            # Test missing required parameter
            hp_prior_bad = HyperparameterPrior(
                (ρ = Beta(1, 1),),        # Free parameter
                fixed = (μ = 0.0,)        # Wrong parameter - Normal needs σ
            )
            
            @test_throws ErrorException INLAModel(hp_prior_bad, latent_gmrf, obs_model)
        end
        
        @testset "Distribution methods with fixed parameters" begin
            hp_prior = HyperparameterPrior(
                (ρ = Beta(2, 2), τ = Gamma(2, 1)),  # Beta(2,2) has well-defined mode
                fixed = (σ = 0.5,)
            )
            
            # Test mode (only free parameters)
            mode_vec = mode(hp_prior.free_distribution)
            @test length(mode_vec) == 2
            
            # Test rand (only free parameters)
            rand_vec = rand(hp_prior.free_distribution)
            @test length(rand_vec) == 2
            
            # Test logpdf (only free parameters)
            θ_test = [0.5, 1.0]
            log_dens = logpdf(hp_prior.free_distribution, θ_test)
            @test isfinite(log_dens)
        end
        
        @testset "Pretty printing with fixed parameters" begin
            hp_prior = HyperparameterPrior(
                (ρ = Beta(1, 1), τ = Gamma(1, 1)),
                fixed = (σ = 0.5, μ = 0.0)
            )
            
            # Test that show method works
            output = repr(hp_prior)
            @test contains(output, "HyperparameterPrior with 4 parameters")
            @test contains(output, "σ = 0.5 (fixed)")
            @test contains(output, "μ = 0.0 (fixed)")
            @test contains(output, "ρ ~ Beta") && contains(output, "(free)")
            @test contains(output, "τ ~ Gamma") && contains(output, "(free)")
            @test contains(output, "Free parameters: 2")
            @test contains(output, "Fixed parameters: 2")
        end
    end
    
    @testset "Edge cases and error handling" begin
        
        @testset "Empty hyperparameter prior" begin
            # Should throw error as empty priors don't make sense for INLA
            @test_throws ErrorException HyperparameterPrior(NamedTuple())
        end
        
        @testset "Type safety verification" begin
            # Test that the type parameter correctly encodes parameter names
            hp_prior = HyperparameterPrior((σ = Gamma(2, 3), ρ = Uniform(0, 0.5)))
            
            # The type should encode the parameter names
            @test typeof(hp_prior) <: HyperparameterPrior{(:σ, :ρ)}
        end
    end
end