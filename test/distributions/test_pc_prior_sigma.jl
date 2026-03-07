using Test
using IntegratedNestedLaplace
using Distributions
using Random

@testset "PCPrior.Sigma" begin
    @testset "Construction" begin
        d = PCPrior.Sigma(1.0; α = 0.05)
        @test d isa Exponential
        λ = -log(0.05) / 1.0
        @test d.θ ≈ 1 / λ

        @test_throws ArgumentError PCPrior.Sigma(-1.0; α = 0.05)
        @test_throws ArgumentError PCPrior.Sigma(1.0; α = 0.0)
        @test_throws ArgumentError PCPrior.Sigma(1.0; α = 1.0)
    end

    @testset "Calibration: P(σ > U) ≈ α" begin
        U = 2.0
        α = 0.1
        d = PCPrior.Sigma(U; α = α)
        Random.seed!(42)
        samples = rand(d, 50_000)
        @test abs(mean(samples .> U) - α) < 0.02
    end
end
