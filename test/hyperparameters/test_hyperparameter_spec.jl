using Test
using Latte
using Distributions
using Bijectors

@testset "HyperparameterSpec" begin

    @testset "Construction" begin
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

    @testset "Error Handling" begin
        # Test empty free parameters
        @test_throws ErrorException HyperparameterSpec(free = NamedTuple(), fixed = NamedTuple())

        # Test overlap between free and fixed
        hp1 = Hyperparameter(Normal(0, 1), transform = elementwise(exp), prior_space = :working)
        @test_throws ErrorException HyperparameterSpec(free = (σ = hp1,), fixed = (σ = 1.0,))
    end

    @testset "Display" begin
        # Test HyperparameterSpec display
        hp = Hyperparameter(Exponential(1.0), transform = elementwise(log), prior_space = :natural)
        spec = HyperparameterSpec(free = (σ = hp,), fixed = (μ = 0.0,))
        io = IOBuffer()
        show(io, spec)
        output = String(take!(io))
        @test occursin("Free parameters", output)
        @test occursin("Fixed parameters", output)
        @test occursin("σ", output)
        @test occursin("μ", output)
    end

end
