using Test
using IntegratedNestedLaplace
using GaussianMarkovRandomFields
using LDLFactorizations
using Distributions
using LinearAlgebra
using SparseArrays
using Random

@testset "INLAModel" begin

    @testset "Construction and Validation" begin
        # Set up components
        spec = @hyperparams begin
            (σ ~ InverseGamma(2, 1), transform = log, space = natural)
        end

        function latent_gmrf(; σ, kwargs...)
            n = 10
            Q = spdiagm(0 => fill(1 / σ^2, n))
            μ = zeros(n)
            return GMRF(μ, Q)
        end

        obs_model = ExponentialFamily(Normal)  # Requires σ hyperparameter

        # Test successful construction
        model = INLAModel(spec, latent_gmrf, obs_model)
        @test model.hyperparameter_spec == spec
        @test model.latent_prior == latent_gmrf
        @test model.observation_model == obs_model

        # Test type parameters
        @test model isa INLAModel{typeof(spec), typeof(latent_gmrf), typeof(obs_model)}
    end

    @testset "Parameter Validation" begin
        # Missing required hyperparameter
        spec_incomplete = @hyperparams begin
            μ ~ Normal(0, 1)  # Missing σ
        end

        function latent_gmrf(; kwargs...)
            n = 5
            Q = spdiagm(0 => ones(n))
            return GMRF(zeros(n), Q)
        end

        obs_model = ExponentialFamily(Normal)  # Requires σ

        # Should error due to missing σ
        @test_throws ErrorException INLAModel(spec_incomplete, latent_gmrf, obs_model)
    end

    @testset "latent_gmrf Function" begin
        spec = @hyperparams begin
            (σ ~ InverseGamma(2, 1), transform = log, space = natural)
        end

        function latent_gmrf_func(; σ, kwargs...)
            n = 8
            Q = spdiagm(0 => fill(1 / σ^2, n))
            return GMRF(zeros(n), Q)
        end

        obs_model = ExponentialFamily(Normal)
        model = INLAModel(spec, latent_gmrf_func, obs_model)

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
        spec = @hyperparams begin
            (σ ~ InverseGamma(2, 1), transform = log, space = natural)
        end

        function latent_gmrf_func(; σ, kwargs...)
            n = 6
            Q = spdiagm(0 => fill(1 / σ^2, n))
            return GMRF(zeros(n), Q)
        end

        obs_model = ExponentialFamily(Normal)
        model = INLAModel(spec, latent_gmrf_func, obs_model)

        # Test data
        θ = [log(1.5)]  # σ = 1.5 in natural space, log(1.5) in working space
        x = randn(6)
        y = x + 0.1 * randn(6)  # Noisy observations

        # Test joint density evaluation
        log_joint = log_joint_density(model, x, θ, y)
        @test isa(log_joint, Real)
        @test isfinite(log_joint)

        # Test that different parameters give different densities
        θ2 = [log(0.8)]
        log_joint2 = log_joint_density(model, x, θ2, y)
        @test log_joint != log_joint2

        # Test with different latent field
        x2 = 2 * x
        log_joint3 = log_joint_density(model, x2, θ, y)
        @test log_joint != log_joint3
    end

    @testset "Multiple Hyperparameters" begin
        # Model with multiple hyperparameters
        spec = @hyperparams begin
            (σ_latent ~ InverseGamma(2, 1), transform = log, space = natural)
            (σ ~ InverseGamma(2, 1), transform = log, space = natural)
        end

        function latent_gmrf_func(; σ_latent, kwargs...)
            n = 5
            Q = spdiagm(0 => fill(1 / σ_latent^2, n))
            return GMRF(zeros(n), Q)
        end

        obs_model = ExponentialFamily(Normal)  # Uses σ

        # Test construction with parameter name matching
        model = INLAModel(spec, latent_gmrf_func, obs_model)

        # Test joint density with multiple parameters
        θ = [log(1.2), log(0.8)]  # [σ_latent, σ] in working space
        x = randn(5)
        y = x + 0.1 * randn(5)

        log_joint = log_joint_density(model, x, θ, y)
        @test isfinite(log_joint)
    end

    @testset "Different Observation Models" begin
        # Test with Bernoulli observation model (no hyperparameters)
        # Use Gamma for positive parameters (alternative to InverseGamma)
        spec = @hyperparams begin
            (τ ~ Gamma(2, 1), transform = log, space = natural)
        end

        function ar1_latent(; τ, kwargs...)
            n = 8
            # AR(1) precision matrix
            ϕ = 0.7
            diag_main = [τ; fill(τ * (1 + ϕ^2), n - 2); τ]
            diag_off = fill(-τ * ϕ, n - 1)
            Q = spdiagm(0 => diag_main, -1 => diag_off, 1 => diag_off)
            return GMRF(zeros(n), Q)
        end

        obs_model_bernoulli = ExponentialFamily(Bernoulli)
        model_bernoulli = INLAModel(spec, ar1_latent, obs_model_bernoulli)

        # Test with binary data
        θ = [log(2.0)]  # τ = 2.0 in natural space
        x = randn(8)
        y = rand(8) .> 0.5  # Binary data

        log_joint = log_joint_density(model_bernoulli, x, θ, y)
        @test isfinite(log_joint)
    end

    @testset "Type Stability" begin
        spec = @hyperparams begin
            (σ ~ InverseGamma(2, 1), transform = log, space = natural)
        end

        function latent_gmrf_func(; σ, kwargs...)
            n = 4
            Q = spdiagm(0 => fill(1 / σ^2, n))
            return GMRF(zeros(n), Q)
        end

        obs_model = ExponentialFamily(Normal)
        model = INLAModel(spec, latent_gmrf_func, obs_model)

        θ = [log(1.0)]  # Working space
        x = randn(4)
        y = randn(4)
        θ_named = (σ = 1.0,)  # Natural space

        # Test type stability
        @inferred Float64 log_joint_density(model, x, θ, y)
        @inferred GMRF latent_gmrf(model, θ_named)
    end

    @testset "Pretty Printing" begin
        spec = @hyperparams begin
            (σ ~ InverseGamma(2, 1), transform = log, space = natural)
        end

        function latent_gmrf_func(; σ, kwargs...)
            Q = spdiagm(0 => fill(1 / σ^2, 3))
            return GMRF(zeros(3), Q)
        end

        obs_model = ExponentialFamily(Normal)
        model = INLAModel(spec, latent_gmrf_func, obs_model)

        # Test that show doesn't error
        str = string(model)
        @test occursin("INLAModel", str)
        @test occursin("Hyperparameter spec", str)
        @test occursin("Observation model", str)
    end

    @testset "Integration with Mixed Parameters" begin
        # Test with both free and fixed hyperparameters
        spec = @hyperparams begin
            (σ ~ InverseGamma(2, 1), transform = log, space = natural)  # Free parameter
            df = 3.0  # Fixed parameter
        end

        function latent_gmrf_func(; σ, df, kwargs...)
            # Can use both free (σ) and fixed (df) parameters
            n = 6
            Q = spdiagm(0 => fill(1 / σ^2, n))
            return GMRF(zeros(n), Q)
        end

        # Custom observation model that uses both parameters
        struct TestObsModel <: ObservationModel end
        IntegratedNestedLaplace.hyperparameters(::TestObsModel) = (:σ, :df)

        # Factory pattern implementation
        function (::TestObsModel)(y; σ, df, kwargs...)
            return MaterializedTestObsModel(y, σ)
        end

        struct MaterializedTestObsModel{Y}
            y::Y
            σ::Float64
        end

        IntegratedNestedLaplace.loglik(x, obs_lik::MaterializedTestObsModel) = -0.5 * sum((obs_lik.y - x) .^ 2) / obs_lik.σ^2 - length(obs_lik.y) * log(obs_lik.σ) / 2

        obs_model = TestObsModel()
        model = INLAModel(spec, latent_gmrf_func, obs_model)

        θ = [log(1.5)]  # Only free parameter in working space
        x = randn(6)
        y = x + 0.1 * randn(6)

        log_joint = log_joint_density(model, x, θ, y)
        @test isfinite(log_joint)
    end

    @testset "Random Sampling" begin
        # Simple model
        spec = @hyperparams begin
            (σ ~ InverseGamma(2, 1), transform = log, space = natural)
        end

        function latent_gmrf_func(; σ, kwargs...)
            n = 5
            Q = spdiagm(0 => fill(1 / σ^2, n))
            return GMRF(zeros(n), Q)
        end

        obs_model = ExponentialFamily(Normal)
        model = INLAModel(spec, latent_gmrf_func, obs_model)

        # Test basic sampling
        sample = rand(model)
        @test sample isa NamedTuple{(:θ, :x, :y)}
        # θ is now a NamedTuple in natural space
        @test sample.θ isa NamedTuple
        @test sample.θ.σ > 0
        @test length(sample.x) == length(sample.y) == 5
        @test all(isfinite, values(sample.θ)) && all(isfinite, sample.x) && all(isfinite, sample.y)

        # Test with explicit RNG
        rng = MersenneTwister(123)
        sample1 = rand(rng, model)
        rng = MersenneTwister(123)
        sample2 = rand(rng, model)
        @test sample1.θ == sample2.θ && sample1.x == sample2.x && sample1.y == sample2.y

        # Test different samples are different
        @test rand(model) != rand(model)

        # Test with fixed parameters
        spec_fixed = @hyperparams begin
            (σ_latent ~ InverseGamma(2, 1), transform = log, space = natural)
            σ = 0.5  # Fixed parameter
        end

        function latent_gmrf_fixed(; σ_latent, kwargs...)
            n = 3
            Q = spdiagm(0 => fill(1 / σ_latent^2, n))
            return GMRF(zeros(n), Q)
        end

        model_fixed = INLAModel(spec_fixed, latent_gmrf_fixed, obs_model)
        sample_fixed = rand(model_fixed)
        # θ should include both free and fixed parameters in natural space
        @test sample_fixed.θ isa NamedTuple
        @test haskey(sample_fixed.θ, :σ_latent)  # Free parameter
        @test haskey(sample_fixed.θ, :σ)         # Fixed parameter
        @test sample_fixed.θ.σ == 0.5            # Fixed value
        @test length(sample_fixed.x) == length(sample_fixed.y) == 3
    end

end
