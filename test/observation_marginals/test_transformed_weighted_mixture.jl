using Test
using Latte
using Distributions
using Bijectors
using Random

@testset "TransformedWeightedMixture Tests" begin
    Random.seed!(12345)

    @testset "Construction and Basic Properties" begin
        # Create a simple base distribution
        base = WeightedMixture([Normal(2.0, 0.5), Normal(2.5, 0.3)], [0.6, 0.4])
        bij = elementwise(log)  # Log link

        # Create transformed distribution
        transformed = TransformedWeightedMixture(base, bij)

        @test transformed.base_distribution === base
        @test transformed.bijector === bij
    end

    @testset "Exp Transformation (LogLink)" begin
        # Linear predictor in log space
        base = WeightedMixture([Normal(2.0, 0.2)], [1.0])
        bij = elementwise(log)

        transformed = TransformedWeightedMixture(base, bij)

        # Support should be (0, ∞) (exp transforms R to R+)
        @test minimum(transformed) == 0.0
        @test maximum(transformed) == Inf

        # Test PDF evaluation
        y = 10.0  # Observation space value
        η = log(y)  # Corresponding linear predictor value

        # PDF should be non-zero and finite
        p = pdf(transformed, y)
        @test p > 0.0
        @test isfinite(p)

        # Log PDF should match manual calculation
        logp = logpdf(transformed, y)
        @test logp == logpdf(base, η) + logabsdetjac(bij, y)

        # Mean should be approximately exp(2.0) for base ~ N(2.0, 0.2)
        μ = mean(transformed)
        @test μ ≈ exp(2.0) atol = 0.5  # Some numerical integration tolerance
    end

    @testset "Logistic Transformation (LogitLink)" begin
        # Linear predictor in logit space
        base = WeightedMixture([Normal(0.0, 1.0)], [1.0])
        bij = Bijectors.Logit(0.0, 1.0)

        transformed = TransformedWeightedMixture(base, bij)

        # Support should be (0, 1) for logistic transformation
        @test minimum(transformed) == 0.0
        @test maximum(transformed) == 1.0

        # Test PDF at p = 0.5 (corresponds to η = 0)
        p = pdf(transformed, 0.5)
        @test p > 0.0
        @test isfinite(p)

        # Mean should be close to 0.5 for base ~ N(0, 1)
        μ = mean(transformed)
        @test 0.4 <= μ <= 0.6  # Reasonable tolerance
    end

    @testset "Identity Transformation" begin
        # No transformation
        base = WeightedMixture([Normal(3.0, 1.0), Normal(5.0, 0.5)], [0.5, 0.5])
        bij = identity

        transformed = TransformedWeightedMixture(base, bij)

        # Support should match base
        @test minimum(transformed) == minimum(base)
        @test maximum(transformed) == maximum(base)

        # PDF should match base (identity has Jacobian = 1)
        @test pdf(transformed, 3.0) ≈ pdf(base, 3.0)
        @test pdf(transformed, 5.0) ≈ pdf(base, 5.0)

        # Mean and variance should match base
        @test mean(transformed) ≈ mean(base) rtol = 1.0e-2
        @test var(transformed) ≈ var(base) rtol = 1.0e-2
    end

    @testset "Sampling" begin
        base = WeightedMixture([Normal(1.0, 0.3)], [1.0])
        bij = elementwise(log)
        transformed = TransformedWeightedMixture(base, bij)

        # Sample multiple times
        samples = [rand(transformed) for _ in 1:1000]

        # All samples should be positive (exp transformation)
        @test all(s > 0 for s in samples)

        # Sample mean should be close to distribution mean
        sample_mean = sum(samples) / length(samples)
        dist_mean = mean(transformed)
        @test sample_mean ≈ dist_mean rtol = 0.15
    end

    @testset "CDF and Quantile" begin
        base = WeightedMixture([Normal(2.0, 0.4)], [1.0])
        bij = elementwise(log)
        transformed = TransformedWeightedMixture(base, bij)

        # Test CDF properties
        @test cdf(transformed, minimum(transformed)) ≈ 0.0 atol = 0.01
        @test cdf(transformed, 1.0e10) ≈ 1.0 atol = 0.01

        # CDF should be monotone increasing
        y1, y2, y3 = 5.0, 7.0, 10.0
        @test cdf(transformed, y1) < cdf(transformed, y2) < cdf(transformed, y3)

        # Quantile should invert CDF
        p = 0.5
        q = quantile(transformed, p)
        @test cdf(transformed, q) ≈ p rtol = 0.01

        # Test edge quantiles
        @test quantile(transformed, 0.0) ≈ minimum(transformed) rtol = 0.01
        @test quantile(transformed, 1.0) >= quantile(transformed, 0.99)
    end

    @testset "Moment Caching" begin
        base = WeightedMixture([Normal(1.5, 0.25)], [1.0])
        bij = elementwise(log)
        transformed = TransformedWeightedMixture(base, bij)

        # Moments should be cached after first call
        @test transformed._moments === nothing

        μ1 = mean(transformed)
        @test transformed._moments !== nothing

        # Subsequent calls should return cached value
        μ2 = mean(transformed)
        @test μ1 === μ2

        # Variance uses same cache
        σ²1 = var(transformed)
        σ²2 = var(transformed)
        @test σ²1 === σ²2
    end

    @testset "Support Caching" begin
        base = WeightedMixture([Normal(0.0, 1.0)], [1.0])
        bij = elementwise(log)
        transformed = TransformedWeightedMixture(base, bij)

        # Support should be cached after first call
        @test transformed._support === nothing

        min_val = minimum(transformed)
        @test transformed._support !== nothing

        # Subsequent calls should return cached value
        @test minimum(transformed) === min_val
        @test maximum(transformed) === transformed._support[2]
    end

    @testset "Insupport" begin
        base = WeightedMixture([Normal(0.0, 1.0)], [1.0])
        bij = Bijectors.Logit(0.0, 1.0)
        transformed = TransformedWeightedMixture(base, bij)

        # Values strictly inside (0, 1) should be in support
        @test insupport(transformed, 0.5)
        @test insupport(transformed, 0.1)
        @test insupport(transformed, 0.9)

        # Boundary values and values outside should not be in support
        @test !insupport(transformed, 0.0)  # Boundary
        @test !insupport(transformed, 1.0)  # Boundary
        @test !insupport(transformed, -0.1)  # Outside
        @test !insupport(transformed, 1.1)  # Outside
        @test !insupport(transformed, Inf)  # Infinite
    end
end
