using Test
using Latte
using Distributions
using Random

@testset "PCPrior.Range" begin
    @testset "Construction" begin
        # d=2: P(ρ < ρ0) = exp(-λ ρ0^{-1}) = p  ⇒  λ = -log(p)·ρ0
        d = PCPrior.Range(0.3; p = 0.5, dim = 2)
        @test d.λ ≈ -log(0.5) * 0.3
        @test d.dim == 2

        # direct-λ constructor
        @test PCPrior.Range(0.5).λ == 0.5

        @test_throws ArgumentError PCPrior.Range(-1.0; p = 0.5)
        @test_throws ArgumentError PCPrior.Range(0.3; p = 0.0)
        @test_throws ArgumentError PCPrior.Range(0.3; p = 1.0)
        @test_throws ArgumentError PCPrior.Range(0.3; p = 0.5, dim = 0)
    end

    @testset "Closed forms (d=2)" begin
        d = PCPrior.Range(0.3; p = 0.5, dim = 2)
        # calibration point + median both equal ρ0 at p=0.5
        @test cdf(d, 0.3) ≈ 0.5
        @test median(d) ≈ 0.3
        # mode = (λ·(d/2)/(d/2+1))^{2/d} = λ/2 for d=2
        @test mode(d) ≈ d.λ / 2
        # cdf/quantile are inverses
        @test quantile(d, cdf(d, 0.7)) ≈ 0.7
        # logpdf matches the d=2 density λ ρ⁻² exp(-λ/ρ)
        ρ = 0.42
        @test logpdf(d, ρ) ≈ log(d.λ) - 2 * log(ρ) - d.λ / ρ
        @test logpdf(d, -1.0) == -Inf
        @test support(d) == RealInterval(0.0, Inf)
    end

    @testset "Calibration: P(range < ρ0) ≈ p" begin
        ρ0, p = 0.4, 0.3
        d = PCPrior.Range(ρ0; p = p, dim = 2)
        Random.seed!(42)
        samples = rand(d, 50_000)
        @test abs(mean(samples .< ρ0) - p) < 0.02
    end

    @testset "Dimension d=1" begin
        # P(ρ<ρ0)=exp(-λ ρ0^{-1/2})=p ⇒ λ = -log(p)·ρ0^{1/2}
        d = PCPrior.Range(0.3; p = 0.5, dim = 1)
        @test d.λ ≈ -log(0.5) * sqrt(0.3)
        @test cdf(d, 0.3) ≈ 0.5
    end
end
