using Test
using IntegratedNestedLaplace
using Distributions

@testset "Hyperparameter Integration Tests" begin

    @testset "Integration with hyperparameter_logpdf" begin
        # Test that existing hyperparameter_logpdf function works with new system

        # Create a simple mock setup
        hp_prior = HyperparameterPrior((σ = Gamma(2, 3), ρ = Uniform(0, 0.5)))

        # Mock latent GMRF function
        function mock_latent_gmrf(θ)
            # Extract hyperparameters by name
            σ = get_hyperparameter(θ, hp_prior, :σ)
            ρ = get_hyperparameter(θ, hp_prior, :ρ)

            # Return a mock GMRF that depends on these parameters
            return (σ = σ, ρ = ρ, dimension = 10)  # Mock GMRF
        end

        obs_model = ExponentialFamily(Normal)
        model = INLAModel(hp_prior, mock_latent_gmrf, obs_model)

        # Test that we can extract the underlying distribution
        @test model.hyperparameter_prior.free_distribution isa Product
        @test length(model.hyperparameter_prior.free_distribution) == 2

        # Test parameter name access
        @test :σ in keys(model.hyperparameter_prior.name_to_index)
        @test :ρ in keys(model.hyperparameter_prior.name_to_index)
    end

    @testset "Integration with exploration bounds" begin
        # Test that exploration bounds work with named parameters
        hp_prior = HyperparameterPrior((μ = Normal(0, 1), σ = Gamma(1, 1)))

        # Mock exploration points
        exploration_points = [
            [0.5, 1.2], [-0.3, 0.8], [1.1, 2.1], [-0.8, 0.6], [0.2, 1.8],
        ]

        # Test bound computation for each dimension by name
        μ_values = [get_hyperparameter(θ, hp_prior, :μ) for θ in exploration_points]
        σ_values = [get_hyperparameter(θ, hp_prior, :σ) for θ in exploration_points]

        @test minimum(μ_values) ≈ -0.8
        @test maximum(μ_values) ≈ 1.1
        @test minimum(σ_values) ≈ 0.6
        @test maximum(σ_values) ≈ 2.1
    end

    @testset "Parameter extraction for observation models" begin
        # Test extracting specific hyperparameters needed by observation models
        hp_prior = HyperparameterPrior((σ = Gamma(2, 3), μ = Normal(0, 1), ρ = Uniform(0, 1)))
        θ = [1.5, 0.2, 0.7]

        # Normal observation model only needs σ
        normal_params = extract_hyperparameters(θ, hp_prior, hyperparameters(ExponentialFamily(Normal)))
        @test haskey(normal_params, :σ)
        @test !haskey(normal_params, :μ)
        @test !haskey(normal_params, :ρ)
        @test normal_params.σ == 1.5

        # Bernoulli needs no hyperparameters
        bernoulli_params = extract_hyperparameters(θ, hp_prior, hyperparameters(ExponentialFamily(Bernoulli)))
        @test isempty(bernoulli_params)
    end

    @testset "Marginalization with named parameters" begin
        # Test that marginalization works with parameter names
        hp_prior = HyperparameterPrior((σ = Gamma(2, 3), ρ = Uniform(0, 0.5)))

        # Test parameter index lookup for marginalization
        σ_index = hp_prior.name_to_index[:σ]
        ρ_index = hp_prior.name_to_index[:ρ]

        @test σ_index == 1 || σ_index == 2
        @test ρ_index == 1 || ρ_index == 2
        @test σ_index != ρ_index

        # Test that we can identify dimensions by name
        @test hp_prior.name_to_index[:σ] == σ_index
        @test hp_prior.name_to_index[:ρ] == ρ_index
    end

    @testset "Printing and display" begin
        # Test that the new system has good string representations
        hp_prior = HyperparameterPrior((σ = Gamma(2, 3), ρ = Uniform(0, 0.5)))

        # Test that showing the prior includes parameter names
        io = IOBuffer()
        show(io, hp_prior)
        output = String(take!(io))

        @test occursin("σ", output)
        @test occursin("ρ", output)
        @test occursin("Gamma", output)
        @test occursin("Uniform", output)

        # Test INLAModel display
        latent_gmrf = (θ) -> nothing
        obs_model = ExponentialFamily(Normal)
        model = INLAModel(hp_prior, latent_gmrf, obs_model)

        io = IOBuffer()
        show(io, model)
        model_output = String(take!(io))

        @test occursin("σ", model_output)
        @test occursin("Normal", model_output)
    end
end
