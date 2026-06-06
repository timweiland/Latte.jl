using Test
using Latte
using Distributions
using Random
using HCubature

@testset "PCPrior.Precision" begin
    @testset "Construction" begin
        d = PCPrior.Precision(1.0; α = 0.05)
        @test d isa PCPrior.Precision
        @test d.λ ≈ -log(0.05) / 1.0

        # Direct λ construction
        d2 = PCPrior.Precision(3.0)
        @test d2.λ ≈ 3.0

        # Invalid inputs
        @test_throws ArgumentError PCPrior.Precision(-1.0; α = 0.05)
        @test_throws ArgumentError PCPrior.Precision(1.0; α = 0.0)
        @test_throws ArgumentError PCPrior.Precision(1.0; α = 1.0)
    end

    @testset "Support and boundaries" begin
        d = PCPrior.Precision(1.0; α = 0.05)
        @test minimum(support(d)) == 0.0
        @test maximum(support(d)) == Inf
        @test logpdf(d, 0.0) == -Inf
        @test logpdf(d, -1.0) == -Inf
    end

    @testset "logpdf known values" begin
        # λ = -log(0.05) ≈ 2.9957; verify against hand-computed values
        d = PCPrior.Precision(1.0; α = 0.05)
        λ = d.λ
        @test logpdf(d, 0.1) ≈ log(λ) - log(2) - 1.5 * log(0.1) - λ / sqrt(0.1)
        @test logpdf(d, 1.0) ≈ log(λ) - log(2) - λ  # simplifies at τ=1
        @test logpdf(d, 10.0) ≈ log(λ) - log(2) - 1.5 * log(10) - λ / sqrt(10)
    end

    @testset "mode is a local maximum" begin
        d = PCPrior.Precision(1.0; α = 0.05)
        m = mode(d)
        lp_mode = logpdf(d, m)
        ε = m * 1.0e-4
        @test lp_mode > logpdf(d, m - ε)
        @test lp_mode > logpdf(d, m + ε)
    end

    @testset "Calibration: P(σ > U) ≈ α" begin
        U = 1.0
        α = 0.05
        d = PCPrior.Precision(U; α = α)
        Random.seed!(123)
        samples = [rand(d) for _ in 1:50_000]
        empirical_α = mean(samples .< 1 / U^2)
        @test abs(empirical_α - α) < 0.02
    end

    @testset "Normalization: ∫pdf(τ)dτ ≈ 1" begin
        d = PCPrior.Precision(1.0; α = 0.05)
        integral, _ = hcubature([-20.0], [20.0]) do t
            τ = exp(t[1])
            return exp(logpdf(d, τ) + t[1])
        end
        @test abs(integral - 1.0) < 0.01
    end

    @testset "rand/logpdf consistency" begin
        # Verify that the empirical CDF from rand matches the logpdf
        # by checking P(τ < threshold) against numerical integration
        d = PCPrior.Precision(1.0; α = 0.05)
        threshold = mode(d) * 2  # pick a point with decent mass on both sides

        # Empirical from samples
        Random.seed!(42)
        samples = [rand(d) for _ in 1:20_000]
        empirical_cdf = mean(samples .< threshold)

        # Numerical CDF from logpdf
        log_threshold = log(threshold)
        cdf_integral, _ = hcubature([-20.0], [log_threshold]) do t
            τ = exp(t[1])
            return exp(logpdf(d, τ) + t[1])
        end

        @test abs(empirical_cdf - cdf_integral) < 0.02
    end

    @testset "cdf basic properties" begin
        d = PCPrior.Precision(1.0; α = 0.05)
        λ = d.λ
        @test cdf(d, 0.0) == 0.0
        @test cdf(d, -1.0) == 0.0
        @test cdf(d, Inf) == 1.0
        @test cdf(d, 1.0) ≈ exp(-λ)
        @test cdf(d, 1.0) ≈ 0.05 atol = 1.0e-12   # P(τ < 1/U²) = α at U=1
        ts = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 20.0]
        @test issorted(cdf.(Ref(d), ts))
        @test all(0 .<= cdf.(Ref(d), ts) .<= 1)
    end

    @testset "cdf consistent with logpdf (numerical integration)" begin
        d = PCPrior.Precision(1.0; α = 0.05)
        for t in (0.3, 1.0, 3.0, 10.0)
            num, _ = hcubature([-40.0], [log(t)]) do s
                τ = exp(s[1])
                exp(logpdf(d, τ) + s[1])   # Jacobian dτ = τ ds
            end
            @test cdf(d, t) ≈ num atol = 1.0e-6
        end
    end

    @testset "quantile is the cdf inverse" begin
        d = PCPrior.Precision(2.0)
        for p in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
            q = quantile(d, p)
            @test q > 0
            @test cdf(d, q) ≈ p atol = 1.0e-10
        end
        @test quantile(d, 0.0) == 0.0
        @test quantile(d, 1.0) == Inf
        @test_throws DomainError quantile(d, -0.1)
        @test_throws DomainError quantile(d, 1.5)
    end

    @testset "median == quantile(0.5), in support, above mode" begin
        d = PCPrior.Precision(1.0; α = 0.05)
        m = median(d)
        @test m == quantile(d, 0.5)
        @test m ≈ (d.λ / log(0.5))^2
        @test insupport(d, m)
        @test cdf(d, m) ≈ 0.5 atol = 1.0e-12
        @test m > mode(d)                  # right-skewed
    end

    @testset "introspection does not crash (regression)" begin
        d = PCPrior.Precision(1.0; α = 0.05)
        @test isfinite(cdf(d, mode(d)))
        @test isfinite(quantile(d, 0.5))
        @test isfinite(median(d))
        @test insupport(d, mode(d))
        @test insupport(d, median(d))
    end
end
