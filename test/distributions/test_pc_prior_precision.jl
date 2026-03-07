using Test
using IntegratedNestedLaplace
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
end
