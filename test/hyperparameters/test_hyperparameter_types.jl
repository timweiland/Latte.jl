using Test
using IntegratedNestedLaplace
using Distributions
using Bijectors
using LinearAlgebra

@testset "Hyperparameter Types" begin

    @testset "Hyperparameter Construction" begin
        # Test basic construction with identity transform
        hp1 = Hyperparameter(Normal(0, 10), transform = identity, prior_space = :working)
        @test hp1.prior isa Normal
        @test hp1.transform == identity
        @test IntegratedNestedLaplace.prior_space(hp1) == :working

        # Test construction with log transform in working space
        hp2 = Hyperparameter(Normal(0, 1), transform = elementwise(exp), prior_space = :working)
        @test hp2.prior isa Normal
        @test IntegratedNestedLaplace.prior_space(hp2) == :working

        # Test construction with log transform in natural space (PC prior)
        hp3 = Hyperparameter(Exponential(1.0), transform = elementwise(log), prior_space = :natural)
        # Prior should be transformed to working space
        @test hp3.prior isa Bijectors.TransformedDistribution
        @test IntegratedNestedLaplace.prior_space(hp3) == :natural

        # Test construction with logit transform
        hp4 = Hyperparameter(Beta(2, 2), transform = Bijectors.Logit(0.0, 1.0), prior_space = :natural)
        @test hp4.prior isa Bijectors.TransformedDistribution
        @test IntegratedNestedLaplace.prior_space(hp4) == :natural
    end

    @testset "Hyperparameter Error Handling" begin
        # Test invalid prior_space
        @test_throws ErrorException Hyperparameter(Normal(0, 1), transform = elementwise(exp), prior_space = :invalid)
    end

    @testset "HyperparameterSpec Construction" begin
        # Basic spec with one free parameter
        hp1 = Hyperparameter(Exponential(1.0), transform = elementwise(log), prior_space = :natural)
        spec1 = HyperparameterSpec(free = (σ = hp1,), fixed = NamedTuple())

        @test length(keys(spec1.free)) == 1
        @test length(keys(spec1.fixed)) == 0
        @test keys(spec1.free) == (:σ,)

        # Spec with free and fixed parameters
        hp2 = Hyperparameter(Beta(2, 2), transform = Bijectors.Logit(0.0, 1.0), prior_space = :natural)
        spec2 = HyperparameterSpec(free = (σ = hp1, ρ = hp2), fixed = (μ = 0.0, τ = 1.0))

        @test length(keys(spec2.free)) == 2
        @test length(keys(spec2.fixed)) == 2
        @test keys(spec2.free) == (:σ, :ρ)
        @test spec2.fixed.μ == 0.0
        @test spec2.fixed.τ == 1.0
    end

    @testset "HyperparameterSpec Error Handling" begin
        # Test empty free parameters
        @test_throws ErrorException HyperparameterSpec(free = NamedTuple(), fixed = NamedTuple())

        # Test overlap between free and fixed
        hp1 = Hyperparameter(Normal(0, 1), transform = elementwise(exp), prior_space = :working)
        @test_throws ErrorException HyperparameterSpec(free = (σ = hp1,), fixed = (σ = 1.0,))
    end

    @testset "to_natural and to_working transformations" begin
        # Test with log transform (natural → working is log)
        hp_σ = Hyperparameter(Exponential(1.0), transform = elementwise(log), prior_space = :natural)
        spec = HyperparameterSpec(free = (σ = hp_σ,), fixed = (μ = 0.0,))

        # Working space (log scale)
        θ_working = (σ = -0.5,)
        θ_natural = to_natural(θ_working, spec)

        @test θ_natural.σ ≈ exp(-0.5) atol = 1.0e-10
        @test θ_natural.μ == 0.0  # Fixed parameter included

        # Round-trip: natural → working → natural
        θ_working_back = to_working(θ_natural, spec)
        @test θ_working_back.σ ≈ θ_working.σ atol = 1.0e-10

        θ_natural_back = to_natural(θ_working_back, spec)
        @test θ_natural_back.σ ≈ θ_natural.σ atol = 1.0e-10
        @test θ_natural_back.μ == 0.0

        # Test with multiple parameters
        hp_ρ = Hyperparameter(Beta(2, 2), transform = Bijectors.Logit(0.0, 1.0), prior_space = :natural)
        spec2 = HyperparameterSpec(free = (σ = hp_σ, ρ = hp_ρ), fixed = (μ = 0.0,))

        θ_working2 = (σ = log(2.0), ρ = Bijectors.Logit(0.0, 1.0)(0.7))
        θ_natural2 = to_natural(θ_working2, spec2)

        @test θ_natural2.σ ≈ 2.0 atol = 1.0e-10
        @test θ_natural2.ρ ≈ 0.7 atol = 1.0e-10
        @test θ_natural2.μ == 0.0

        # Test round-trip with multiple parameters
        θ_working2_back = to_working(θ_natural2, spec2)
        @test θ_working2_back.σ ≈ θ_working2.σ atol = 1.0e-10
        @test θ_working2_back.ρ ≈ θ_working2.ρ atol = 1.0e-10

        # Test with identity transform
        hp_μ = Hyperparameter(Normal(0, 10), transform = identity, prior_space = :working)
        spec3 = HyperparameterSpec(free = (μ = hp_μ,), fixed = NamedTuple())

        θ_working3 = (μ = 5.0,)
        θ_natural3 = to_natural(θ_working3, spec3)
        @test θ_natural3.μ == 5.0  # Identity transform

        θ_working3_back = to_working(θ_natural3, spec3)
        @test θ_working3_back.μ == 5.0
    end

    @testset "logpdf_prior with Jacobian correction" begin
        # Test PC prior: Exponential on σ, optimized on log(σ)
        hp_σ = Hyperparameter(Exponential(1.0), transform = elementwise(log), prior_space = :natural)
        spec = HyperparameterSpec(free = (σ = hp_σ,), fixed = NamedTuple())

        # In working space (log σ)
        θ_working = (σ = log(2.0),)
        log_p_working = logpdf_prior(θ_working, spec)

        # Manually compute: log[p(σ) * |dσ/d(log σ)|] = log[p(σ) * σ]
        σ_natural = 2.0
        log_p_natural = logpdf(Exponential(1.0), σ_natural)
        log_jacobian = log(σ_natural)  # Jacobian of exp transformation
        log_p_expected = log_p_natural + log_jacobian

        @test log_p_working ≈ log_p_expected atol = 1.0e-10

        # Test with multiple parameters
        hp_ρ = Hyperparameter(Beta(2, 2), transform = Bijectors.Logit(0.0, 1.0), prior_space = :natural)
        spec2 = HyperparameterSpec(free = (σ = hp_σ, ρ = hp_ρ), fixed = NamedTuple())

        θ_working2 = (σ = log(2.0), ρ = Bijectors.Logit(0.0, 1.0)(0.7))
        log_p_working2 = logpdf_prior(θ_working2, spec2)

        # Should be sum of individual log priors (independence)
        σ_nat = 2.0
        ρ_nat = 0.7
        log_p_σ = logpdf(Exponential(1.0), σ_nat) + log(σ_nat)
        log_p_ρ = logpdf(Beta(2, 2), ρ_nat) + log(ρ_nat * (1 - ρ_nat))
        log_p_expected2 = log_p_σ + log_p_ρ

        @test log_p_working2 ≈ log_p_expected2 atol = 1.0e-10

        # Test with identity transform (no Jacobian)
        hp_μ = Hyperparameter(Normal(0, 10), transform = identity, prior_space = :working)
        spec3 = HyperparameterSpec(free = (μ = hp_μ,), fixed = NamedTuple())

        θ_working3 = (μ = 5.0,)
        log_p_working3 = logpdf_prior(θ_working3, spec3)
        log_p_expected3 = logpdf(Normal(0, 10), 5.0)

        @test log_p_working3 ≈ log_p_expected3 atol = 1.0e-10
    end

    @testset "to_named_tuple and to_vector conversions" begin
        hp_σ = Hyperparameter(Exponential(1.0), transform = elementwise(log), prior_space = :natural)
        hp_ρ = Hyperparameter(Beta(2, 2), transform = Bijectors.Logit(0.0, 1.0), prior_space = :natural)
        spec = HyperparameterSpec(free = (σ = hp_σ, ρ = hp_ρ), fixed = (μ = 0.0,))

        # Vector to NamedTuple
        θ_vec = [log(2.0), Bijectors.Logit(0.0, 1.0)(0.7)]
        θ_nt = to_named_tuple(θ_vec, spec)

        @test θ_nt.σ ≈ log(2.0) atol = 1.0e-10
        @test θ_nt.ρ ≈ Bijectors.Logit(0.0, 1.0)(0.7) atol = 1.0e-10

        # NamedTuple to Vector
        θ_vec_back = to_vector(θ_nt, spec)

        @test θ_vec_back[1] ≈ θ_vec[1] atol = 1.0e-10
        @test θ_vec_back[2] ≈ θ_vec[2] atol = 1.0e-10

        # Test error handling
        θ_vec_wrong_size = [log(2.0)]
        @test_throws ErrorException to_named_tuple(θ_vec_wrong_size, spec)
    end

    @testset "Type stability" begin
        hp_σ = Hyperparameter(Exponential(1.0), transform = elementwise(log), prior_space = :natural)
        hp_ρ = Hyperparameter(Beta(2, 2), transform = Bijectors.Logit(0.0, 1.0), prior_space = :natural)
        spec = HyperparameterSpec(free = (σ = hp_σ, ρ = hp_ρ), fixed = (μ = 0.0,))

        θ_working = (σ = log(2.0), ρ = Bijectors.Logit(0.0, 1.0)(0.7))

        # Test type stability of key functions
        @test @inferred(to_natural(θ_working, spec)) isa NamedTuple
        @test @inferred(logpdf_prior(θ_working, spec)) isa Float64

        θ_vec = [log(2.0), Bijectors.Logit(0.0, 1.0)(0.7)]
        # Test that to_named_tuple returns correct concrete type (fieldnames extraction ensures type stability)
        result_nt = to_named_tuple(θ_vec, spec)
        @test result_nt isa NamedTuple{(:σ, :ρ)}
        @test result_nt.σ ≈ θ_vec[1]
        @test result_nt.ρ ≈ θ_vec[2]

        @test @inferred(to_vector(θ_working, spec)) isa Vector{Float64}

        # Test prior_space type stability
        @test @inferred(IntegratedNestedLaplace.prior_space(hp_σ)) isa Symbol
    end

    @testset "Display methods" begin
        # Test Hyperparameter display
        hp = Hyperparameter(Exponential(1.0), transform = elementwise(log), prior_space = :natural)
        io = IOBuffer()
        show(io, hp)
        output = String(take!(io))
        @test occursin("natural space", output)

        # Test HyperparameterSpec display
        spec = HyperparameterSpec(free = (σ = hp,), fixed = (μ = 0.0,))
        io = IOBuffer()
        show(io, spec)
        output = String(take!(io))
        @test occursin("Free parameters", output)
        @test occursin("Fixed parameters", output)
        @test occursin("σ", output)
        @test occursin("μ", output)
    end

    @testset "Edge cases and numerical stability" begin
        # Test with very small/large values
        hp_σ = Hyperparameter(Exponential(1.0), transform = elementwise(log), prior_space = :natural)
        spec = HyperparameterSpec(free = (σ = hp_σ,), fixed = NamedTuple())

        # Very small σ
        θ_working_small = (σ = -10.0,)  # σ ≈ 4.5e-5
        θ_natural_small = to_natural(θ_working_small, spec)
        @test θ_natural_small.σ ≈ exp(-10.0) atol = 1.0e-15
        @test θ_natural_small.σ > 0

        # Very large σ
        θ_working_large = (σ = 10.0,)  # σ ≈ 22026
        θ_natural_large = to_natural(θ_working_large, spec)
        @test θ_natural_large.σ ≈ exp(10.0) atol = 1.0e-10
        @test isfinite(θ_natural_large.σ)

        # Test logit with extreme values
        hp_ρ = Hyperparameter(Beta(2, 2), transform = Bijectors.Logit(0.0, 1.0), prior_space = :natural)
        spec2 = HyperparameterSpec(free = (ρ = hp_ρ,), fixed = NamedTuple())

        # Near 0
        θ_working_low = (ρ = -5.0,)  # ρ ≈ 0.0067
        θ_natural_low = to_natural(θ_working_low, spec2)
        @test θ_natural_low.ρ ≈ inverse(Bijectors.Logit(0.0, 1.0))(-5.0) atol = 1.0e-10
        @test 0 < θ_natural_low.ρ < 1

        # Near 1
        θ_working_high = (ρ = 5.0,)  # ρ ≈ 0.9933
        θ_natural_high = to_natural(θ_working_high, spec2)
        @test θ_natural_high.ρ ≈ inverse(Bijectors.Logit(0.0, 1.0))(5.0) atol = 1.0e-10
        @test 0 < θ_natural_high.ρ < 1
    end

end
