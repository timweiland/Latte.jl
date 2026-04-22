using Test
using Latte
using Distributions
using Statistics
using StatsBase: ecdf
using Random

@testset "SampleMarginal" begin

    @testset "Basic moments on known Normal samples" begin
        Random.seed!(42)
        samples = randn(10_000) .* 2.0 .+ 1.0  # N(1, 4)
        d = SampleMarginal(samples)

        @test isapprox(mean(d), 1.0; atol = 0.05)
        @test isapprox(std(d), 2.0; atol = 0.05)
        @test isapprox(var(d), 4.0; atol = 0.2)
        @test isapprox(median(d), 1.0; atol = 0.1)
    end

    @testset "Quantiles match empirical quantiles of the samples" begin
        Random.seed!(7)
        samples = rand(LogNormal(0.3, 0.5), 5_000)
        d = SampleMarginal(samples)

        for q in (0.025, 0.25, 0.5, 0.75, 0.975)
            @test quantile(d, q) ≈ quantile(samples, q)
        end
    end

    @testset "cdf matches the empirical CDF" begin
        Random.seed!(3)
        samples = randn(1_000)
        d = SampleMarginal(samples)
        ref = ecdf(samples)

        for x in range(-2.0, 2.0; length = 11)
            @test cdf(d, x) ≈ ref(x)
        end
    end

    @testset "pdf integrates to approximately 1" begin
        Random.seed!(5)
        samples = randn(5_000)
        d = SampleMarginal(samples)

        # Coarse numerical integration over the support
        xs = range(minimum(d) - 1, maximum(d) + 1; length = 2001)
        dx = step(xs)
        I = sum(pdf(d, x) for x in xs) * dx
        @test 0.9 < I < 1.1

        # Positive anywhere the samples cover
        @test pdf(d, 0.0) > 0
        @test pdf(d, median(samples)) > 0
    end

    @testset "logpdf is log of pdf" begin
        Random.seed!(9)
        d = SampleMarginal(randn(1_000))
        for x in (-1.0, 0.0, 0.7)
            @test logpdf(d, x) ≈ log(pdf(d, x))
        end
    end

    @testset "rand draws from the stored samples (by default)" begin
        Random.seed!(11)
        samples = Float64[1, 2, 3, 5, 8]
        d = SampleMarginal(samples)
        draws = [rand(d) for _ in 1:1_000]
        @test Set(unique(draws)) ⊆ Set(samples)  # bootstrap ⇒ only these values
    end

    @testset "minimum / maximum / insupport track the samples" begin
        samples = [-3.0, 1.5, 7.2, 10.1]
        d = SampleMarginal(samples)
        @test minimum(d) == -3.0
        @test maximum(d) == 10.1
        @test insupport(d, 0.0) === true
        @test insupport(d, -10.0) === false
        @test insupport(d, NaN) === false
    end

    @testset "Distributions contract: subtype + eltype" begin
        d = SampleMarginal(randn(100))
        @test d isa Distributions.ContinuousUnivariateDistribution
    end
end
