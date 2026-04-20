using Test
using Latte
using GaussianMarkovRandomFields
using LinearAlgebra
using SparseArrays
using Distributions

import GaussianMarkovRandomFields: hyperparameters, precision_matrix, constraints, model_name

@testset "AugmentedLatentModel Tests" begin
    @testset "AugmentationInfo" begin
        info = AugmentationInfo(10, 5)

        @test info.n_linear_predictors == 10
        @test info.n_base_latent == 5
        @test info.linear_predictor_indices == 1:10
        @test info.base_latent_indices == 11:15
        @test length(info.linear_predictor_indices) + length(info.base_latent_indices) == 15
    end

    @testset "AugmentedLatentModel - Basic Construction" begin
        # Create a simple base model
        base_model = IIDModel(5)
        A = randn(10, 5)

        # Create augmented model
        aug_model = AugmentedLatentModel(base_model, A)

        @test length(aug_model) == 15  # 10 linear predictors + 5 base
        @test hyperparameters(aug_model) == (τ = Real,)  # Inherited from IIDModel
        @test linear_predictor_indices(aug_model) == 1:10
        @test base_latent_indices(aug_model) == 11:15
    end

    @testset "AugmentedLatentModel - GMRF Generation" begin
        # Create base model and design matrix
        n_base = 5
        n_obs = 10
        base_model = IIDModel(n_base)
        A = randn(n_obs, n_base)

        aug_model = AugmentedLatentModel(base_model, A; linear_predictor_precision = 1.0e6)

        # Generate GMRF
        gmrf = aug_model(τ = 2.0)

        # Check dimensions
        @test length(mean(gmrf)) == n_obs + n_base
        Q = precision_matrix(gmrf)
        @test size(Q) == (n_obs + n_base, n_obs + n_base)

        # Check mean structure: [A * μ_base; μ_base]
        μ = mean(gmrf)
        μ_η = μ[1:n_obs]
        μ_base = μ[(n_obs + 1):end]
        @test all(μ_base .== 0)  # IIDModel has zero mean
        @test all(μ_η .== 0)      # A * 0 = 0

        # Check precision matrix structure (should have off-diagonal coupling)
        # Top-left block: Q_η
        Q_η_block = Q[1:n_obs, 1:n_obs]
        @test Q_η_block ≈ Diagonal(fill(1.0e6, n_obs))

        # Off-diagonal should be non-zero (coupling)
        Q_off = Q[1:n_obs, (n_obs + 1):end]
        @test any(Q_off .!= 0)  # Should have coupling terms
    end

    @testset "AugmentedLatentModel - Non-zero Base Mean" begin
        # Test that mean is correctly propagated: μ_full = [A * μ_base; μ_base]
        n_base = 3
        n_obs = 5
        A = [
            1.0 0.0 0.0;
            0.0 1.0 0.0;
            0.0 0.0 1.0;
            1.0 1.0 0.0;
            0.0 1.0 1.0
        ]

        # Create a custom model with non-zero mean
        struct CustomMeanModel <: LatentModel
            n::Int
            μ_base::Vector{Float64}
        end

        Base.length(m::CustomMeanModel) = m.n
        GaussianMarkovRandomFields.hyperparameters(m::CustomMeanModel) = NamedTuple()
        GaussianMarkovRandomFields.precision_matrix(m::CustomMeanModel; kwargs...) = Diagonal(fill(1.0e-6, m.n))
        Distributions.mean(m::CustomMeanModel; kwargs...) = m.μ_base
        GaussianMarkovRandomFields.constraints(m::CustomMeanModel; kwargs...) = nothing
        GaussianMarkovRandomFields.model_name(::CustomMeanModel) = :custom

        μ_base_test = [1.0, 2.0, 3.0]
        base_model = CustomMeanModel(n_base, μ_base_test)

        aug_model = AugmentedLatentModel(base_model, A)
        gmrf = aug_model()

        μ_full = mean(gmrf)
        μ_η = μ_full[1:n_obs]
        μ_base_result = μ_full[(n_obs + 1):end]

        # Check base mean is preserved
        @test μ_base_result ≈ μ_base_test

        # Check linear predictor mean is A * μ_base
        expected_μ_η = A * μ_base_test
        @test μ_η ≈ expected_μ_η
    end

    @testset "AugmentedLatentModel - Constraints Propagation" begin
        # Test that constraints from base model are correctly offset
        # RW1Model has sum-to-zero constraint
        n_base = 5
        n_obs = 10
        base_model = RW1Model(n_base)
        A = randn(n_obs, n_base)

        aug_model = AugmentedLatentModel(base_model, A)

        # Check constraints
        constraint_info = constraints(aug_model, τ = 1.0)
        @test constraint_info !== nothing

        A_constr, e_constr = constraint_info
        n_constraints = size(A_constr, 1)

        # Constraint matrix should be [zeros(n_constraints, n_obs)  A_base]
        # Check that first n_obs columns are zero
        @test all(A_constr[:, 1:n_obs] .== 0)

        # Check that the constraint applies to the base components
        # (actual constraint from RW1Model is in the last n_base columns)
        @test any(A_constr[:, (n_obs + 1):end] .!= 0)
    end
end
