using Test
using Distributions
using StatsBase
using Random
using IntegratedNestedLaplace

@testset "WeightedMixture Tests" begin
    
    @testset "Constructor Tests" begin
        # Basic constructor
        components = [Normal(0, 1), Normal(2, 1)]
        weights = [0.3, 0.7]
        mixture = WeightedMixture(components, weights)
        
        @test length(mixture.components) == 2
        @test length(mixture.weights) == 2
        @test sum(mixture.weights) ≈ 1.0 atol=1e-10
        
        # Test weight normalization
        unnormalized_weights = [3.0, 7.0]
        mixture_normalized = WeightedMixture(components, unnormalized_weights)
        @test mixture_normalized.weights ≈ [0.3, 0.7] atol=1e-10
        
        # Test single component
        single_mixture = WeightedMixture([Normal(0, 1)], [1.0])
        @test length(single_mixture.components) == 1
        @test single_mixture.weights[1] ≈ 1.0
        
        # Test error cases
        @test_throws AssertionError WeightedMixture([Normal(0, 1)], [0.3, 0.7])  # Length mismatch
        @test_throws AssertionError WeightedMixture([Normal(0, 1)], [-0.5])      # Negative weight
        @test_throws MethodError WeightedMixture(Distribution[], Float64[])   # Empty components
    end
    
    @testset "PDF Tests" begin
        # Simple two-component mixture
        components = [Normal(0, 1), Normal(3, 1)]
        weights = [0.4, 0.6]
        mixture = WeightedMixture(components, weights)
        
        # Test PDF at specific points
        x = 1.0
        expected_pdf = 0.4 * pdf(Normal(0, 1), x) + 0.6 * pdf(Normal(3, 1), x)
        @test pdf(mixture, x) ≈ expected_pdf atol=1e-10
        
        # Test PDF at component means
        @test pdf(mixture, 0.0) ≈ 0.4 * pdf(Normal(0, 1), 0.0) + 0.6 * pdf(Normal(3, 1), 0.0)
        @test pdf(mixture, 3.0) ≈ 0.4 * pdf(Normal(0, 1), 3.0) + 0.6 * pdf(Normal(3, 1), 3.0)
        
        # Test logpdf numerical stability
        @test logpdf(mixture, x) ≈ log(pdf(mixture, x)) atol=1e-10
        
        # Test with extreme values where direct computation might underflow
        extreme_x = -10.0
        logpdf_stable = logpdf(mixture, extreme_x)
        @test isfinite(logpdf_stable)
    end
    
    @testset "CDF Tests" begin
        components = [Normal(0, 1), Normal(2, 1)]
        weights = [0.3, 0.7]
        mixture = WeightedMixture(components, weights)
        
        # Test CDF at specific points
        x = 1.0
        expected_cdf = 0.3 * cdf(Normal(0, 1), x) + 0.7 * cdf(Normal(2, 1), x)
        @test cdf(mixture, x) ≈ expected_cdf atol=1e-10
        
        # Test CDF properties
        @test cdf(mixture, -Inf) ≈ 0.0 atol=1e-10
        @test cdf(mixture, Inf) ≈ 1.0 atol=1e-10
        
        # Test monotonicity
        x1, x2 = 0.0, 1.0
        @test cdf(mixture, x1) <= cdf(mixture, x2)
    end
    
    @testset "Moments Tests" begin
        # Test with known analytical solution
        components = [Normal(1, 2), Normal(4, 1)]
        weights = [0.4, 0.6]
        mixture = WeightedMixture(components, weights)
        
        # Mean: E[X] = Σ wᵢ μᵢ
        expected_mean = 0.4 * 1 + 0.6 * 4
        @test mean(mixture) ≈ expected_mean atol=1e-10
        
        # Variance: Var(X) = E[Var(X|θ)] + Var(E[X|θ])
        # E[Var(X|θ)] = Σ wᵢ σᵢ²
        # Var(E[X|θ]) = Σ wᵢ μᵢ² - (Σ wᵢ μᵢ)²
        expected_var = 0.4 * 4 + 0.6 * 1 + 0.4 * 1^2 + 0.6 * 4^2 - expected_mean^2
        @test var(mixture) ≈ expected_var atol=1e-10
        
        # Standard deviation
        @test std(mixture) ≈ sqrt(expected_var) atol=1e-10
        
        # Test single component reduces to component moments
        single_mixture = WeightedMixture([Normal(2, 3)], [1.0])
        @test mean(single_mixture) ≈ 2.0 atol=1e-10
        @test var(single_mixture) ≈ 9.0 atol=1e-10
    end
    
    @testset "Support Tests" begin
        # Test with bounded and unbounded distributions
        components = [Uniform(-1, 1), Normal(0, 1)]
        weights = [0.5, 0.5]
        mixture = WeightedMixture(components, weights)
        
        # Support bounds
        @test minimum(mixture) == -Inf  # Normal has unbounded support
        @test maximum(mixture) == Inf
        
        # Test with all bounded distributions
        bounded_components = [Uniform(-2, 0), Uniform(1, 3)]
        bounded_weights = [0.4, 0.6]
        bounded_mixture = WeightedMixture(bounded_components, bounded_weights)
        
        @test minimum(bounded_mixture) == -2.0
        @test maximum(bounded_mixture) == 3.0
        
        # Test insupport
        @test insupport(bounded_mixture, -1.0)  # In first component
        @test insupport(bounded_mixture, 2.0)   # In second component
        @test !insupport(bounded_mixture, 0.5)  # In gap between components
        @test !insupport(bounded_mixture, -3.0) # Below minimum
        @test !insupport(bounded_mixture, 4.0)  # Above maximum
    end
    
    @testset "Quantile Tests" begin
        components = [Normal(0, 1), Normal(3, 1)]
        weights = [0.5, 0.5]
        mixture = WeightedMixture(components, weights)
        
        # Test edge cases
        @test quantile(mixture, 0.0) == minimum(mixture)
        @test quantile(mixture, 1.0) == maximum(mixture)
        
        # Test median
        median_val = quantile(mixture, 0.5)
        @test isfinite(median_val)
        @test cdf(mixture, median_val) ≈ 0.5 atol=1e-6
        
        # Test quartiles
        q25 = quantile(mixture, 0.25)
        q75 = quantile(mixture, 0.75)
        @test cdf(mixture, q25) ≈ 0.25 atol=1e-6
        @test cdf(mixture, q75) ≈ 0.75 atol=1e-6
        @test q25 < median_val < q75
        
        # Test with bounded distributions
        bounded_components = [Uniform(0, 1), Uniform(2, 3)]
        bounded_mixture = WeightedMixture(bounded_components, [0.6, 0.4])
        
        # Quantiles should be within support
        q_vals = [quantile(bounded_mixture, p) for p in [0.1, 0.3, 0.5, 0.7, 0.9]]
        @test all(0.0 <= q <= 3.0 for q in q_vals)
        
        # Test quantile error handling
        @test_throws AssertionError quantile(mixture, -0.1)
        @test_throws AssertionError quantile(mixture, 1.1)
    end
    
    @testset "Sampling Tests" begin
        components = [Normal(0, 1), Normal(5, 1)]
        weights = [0.3, 0.7]
        mixture = WeightedMixture(components, weights)
        
        # Test single sample
        Random.seed!(123)
        sample = rand(mixture)
        @test isfinite(sample)
        
        # Test multiple samples
        Random.seed!(123)
        samples = [rand(mixture) for _ in 1:1000]
        @test all(isfinite, samples)
        
        # Test sample statistics approximate mixture statistics
        sample_mean = mean(samples)
        sample_var = var(samples)
        
        @test sample_mean ≈ mean(mixture) atol=0.2  # Allow some sampling variation
        @test sample_var ≈ var(mixture) atol=0.5    # Variance has higher sampling error
        
        # Test with specific RNG
        rng = MersenneTwister(456)
        rng_sample = rand(rng, mixture)
        @test isfinite(rng_sample)
        
        # Test sampling from components approximately matches weights
        Random.seed!(789)
        large_samples = [rand(mixture) for _ in 1:10000]
        
        # Count samples likely from each component (rough heuristic)
        component1_samples = count(x -> x < 2.5, large_samples)  # Rough midpoint
        component2_samples = length(large_samples) - component1_samples
        
        empirical_weight1 = component1_samples / length(large_samples)
        @test empirical_weight1 ≈ 0.3 atol=0.1  # Allow sampling variation
    end
    
    @testset "Edge Cases and Error Handling" begin
        # Test with very small weights
        components = [Normal(0, 1), Normal(1, 1)]
        tiny_weights = [1e-10, 1.0 - 1e-10]
        mixture = WeightedMixture(components, tiny_weights)
        
        @test isfinite(pdf(mixture, 0.5))
        @test isfinite(cdf(mixture, 0.5))
        @test isfinite(mean(mixture))
        @test isfinite(var(mixture))
        
        # Test with identical components
        identical_components = [Normal(2, 1), Normal(2, 1)]
        identical_mixture = WeightedMixture(identical_components, [0.4, 0.6])
        
        @test mean(identical_mixture) ≈ 2.0 atol=1e-10
        @test var(identical_mixture) ≈ 1.0 atol=1e-10
        
        # Test with zero weights (after normalization)
        # This shouldn't happen in normal usage, but test robustness
        components_zero = [Normal(0, 1), Normal(1, 1)]
        zero_weights = [0.0, 1.0]
        mixture_zero = WeightedMixture(components_zero, zero_weights)
        
        @test pdf(mixture_zero, 0.5) ≈ pdf(Normal(1, 1), 0.5) atol=1e-10
    end
    
    @testset "Numerical Stability Tests" begin
        # Test with very different scales
        components = [Normal(0, 1e-6), Normal(1e6, 1e3)]
        weights = [0.5, 0.5]
        mixture = WeightedMixture(components, weights)
        
        # Should not throw or return NaN/Inf
        @test isfinite(pdf(mixture, 0.0))
        @test isfinite(pdf(mixture, 1e6))
        @test isfinite(logpdf(mixture, 0.0))
        @test isfinite(logpdf(mixture, 1e6))
        @test isfinite(cdf(mixture, 5e5))
        @test isfinite(mean(mixture))
        @test isfinite(var(mixture))
        
        # Test logsumexp utility function (internal function)
        log_terms = [-1000.0, -999.0, -1001.0]
        # We can't directly test the internal logsumexp function, so we test via logpdf
        mixture_extreme = WeightedMixture([Normal(-1000, 1), Normal(-999, 1)], [0.5, 0.5])
        @test isfinite(logpdf(mixture_extreme, -999.5))
    end
    
    @testset "Integration with Distributions.jl" begin
        # Test that our distribution works with Distributions.jl functions
        mixture = WeightedMixture([Normal(0, 1), Normal(2, 1)], [0.4, 0.6])
        
        # Test that it's recognized as a distribution
        @test mixture isa ContinuousUnivariateDistribution
        @test mixture isa Distribution
        
        # Test standard interface methods work
        @test hasmethod(pdf, (typeof(mixture), Real))
        @test hasmethod(cdf, (typeof(mixture), Real))
        @test hasmethod(quantile, (typeof(mixture), Real))
        @test hasmethod(mean, (typeof(mixture),))
        @test hasmethod(var, (typeof(mixture),))
        @test hasmethod(rand, (typeof(mixture),))
        
        # Test with common Distributions.jl operations that should work
        @test isfinite(mean(mixture))
        @test isfinite(var(mixture))
        @test isfinite(pdf(mixture, 1.0))
        @test isfinite(cdf(mixture, 1.0))
    end
end