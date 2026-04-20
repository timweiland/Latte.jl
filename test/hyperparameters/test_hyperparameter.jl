using Test
using Latte
using Distributions
using Bijectors

@testset "Hyperparameter" begin

    @testset "Construction" begin
        # Test basic construction with identity transform in working space
        hp1 = Hyperparameter(Normal(0, 10), transform = identity, prior_space = :working)
        # Prior specified in working space → stored as-is
        @test hp1.prior isa Normal
        @test hp1.transform == identity
        @test Latte.prior_space(hp1) == :working

        # Test construction with log transform in working space
        hp2 = Hyperparameter(Normal(0, 1), transform = elementwise(log), prior_space = :working)
        # Prior specified in working space → stored as-is (not wrapped)
        @test hp2.prior isa Normal
        @test Latte.prior_space(hp2) == :working

        # Test construction with log transform in natural space (PC prior)
        hp3 = Hyperparameter(Exponential(1.0), transform = elementwise(log), prior_space = :natural)
        # Prior specified in natural space → transformed to working space
        @test hp3.prior isa Bijectors.TransformedDistribution
        @test Latte.prior_space(hp3) == :natural

        # Test construction with logit transform in natural space
        hp4 = Hyperparameter(Beta(2, 2), transform = Bijectors.Logit(0.0, 1.0), prior_space = :natural)
        # Prior specified in natural space → transformed to working space
        @test hp4.prior isa Bijectors.TransformedDistribution
        @test Latte.prior_space(hp4) == :natural

        # Test construction with identity transform in natural space
        hp5 = Hyperparameter(Normal(0, 10), transform = identity, prior_space = :natural)
        # Identity transform → working = natural, stored as-is
        @test hp5.prior isa Normal
        @test hp5.transform == identity
        @test Latte.prior_space(hp5) == :natural
    end

    @testset "Error Handling" begin
        # Test invalid prior_space
        @test_throws ErrorException Hyperparameter(Normal(0, 1), transform = elementwise(exp), prior_space = :invalid)
    end

    @testset "Display" begin
        # Test Hyperparameter display
        hp = Hyperparameter(Exponential(1.0), transform = elementwise(log), prior_space = :natural)
        io = IOBuffer()
        show(io, hp)
        output = String(take!(io))
        # The display shows the struct with its fields and type parameters
        @test occursin("Hyperparameter", output)
        @test occursin(":natural", output)  # Type parameter shows :natural
    end

end
