using Test
using IntegratedNestedLaplace
using Distributions
using Random
using HCubature

@testset "PCPrior.AR1Correlation" begin
    @testset "Construction" begin
        d = PCPrior.AR1Correlation(0.9; α = 0.05)
        @test d isa PCPrior.AR1Correlation
        d_U = sqrt(-log(1 - 0.9^2))
        @test d.λ ≈ -log(0.05) / d_U
        @test d.positive_only == false

        d_pos = PCPrior.AR1Correlation(0.9; α = 0.05, positive_only = true)
        @test d_pos.positive_only == true

        @test_throws ArgumentError PCPrior.AR1Correlation(0.0; α = 0.05)
        @test_throws ArgumentError PCPrior.AR1Correlation(1.0; α = 0.05)
        @test_throws ArgumentError PCPrior.AR1Correlation(0.5; α = 0.0)
    end

    @testset "Support" begin
        d = PCPrior.AR1Correlation(0.9; α = 0.05)
        @test minimum(support(d)) == -1.0
        @test maximum(support(d)) == 1.0

        d_pos = PCPrior.AR1Correlation(0.9; α = 0.05, positive_only = true)
        @test minimum(support(d_pos)) == 0.0
        @test maximum(support(d_pos)) == 1.0
    end

    @testset "logpdf structural properties" begin
        d = PCPrior.AR1Correlation(0.9; α = 0.05)

        # Boundaries
        @test logpdf(d, -1.0) == -Inf
        @test logpdf(d, 1.0) == -Inf
        @test logpdf(d, 0.0) == -Inf  # integrable singularity at ρ=0

        # Symmetry: two-sided prior is symmetric around 0
        @test logpdf(d, 0.5) ≈ logpdf(d, -0.5)
        @test logpdf(d, 0.01) ≈ logpdf(d, -0.01)

        # positive_only should reject negative values
        d_pos = PCPrior.AR1Correlation(0.9; α = 0.05, positive_only = true)
        @test logpdf(d_pos, -0.5) == -Inf
        @test isfinite(logpdf(d_pos, 0.5))

        # positive_only logpdf = two-sided logpdf + log(2) (twice the mass)
        @test logpdf(d_pos, 0.5) ≈ logpdf(d, 0.5) + log(2)
    end

    @testset "Calibration (positive_only): P(ρ > U) ≈ α" begin
        U = 0.9
        α = 0.05
        d = PCPrior.AR1Correlation(U; α = α, positive_only = true)
        Random.seed!(123)
        samples = [rand(d) for _ in 1:50_000]
        @test all(s -> 0 < s < 1, samples)
        @test abs(mean(samples .> U) - α) < 0.02
    end

    @testset "Calibration (two-sided): P(|ρ| > U) ≈ α" begin
        U = 0.9
        α = 0.05
        d = PCPrior.AR1Correlation(U; α = α, positive_only = false)
        Random.seed!(123)
        samples = [rand(d) for _ in 1:50_000]
        @test all(s -> -1 < s < 1, samples)
        @test any(s -> s < 0, samples)  # should have both signs
        @test abs(mean(abs.(samples) .> U) - α) < 0.02
    end

    @testset "Normalization (positive_only): ∫pdf(ρ)dρ ≈ 1" begin
        d = PCPrior.AR1Correlation(0.9; α = 0.05, positive_only = true)
        integral, _ = hcubature([-10.0], [10.0]) do t
            ρ = (1 + tanh(t[1])) / 2  # maps ℝ → (0,1)
            dρdt = (1 - tanh(t[1])^2) / 2
            lp = logpdf(d, ρ)
            return isfinite(lp) ? exp(lp) * dρdt : 0.0
        end
        @test abs(integral - 1.0) < 0.02
    end

    @testset "Normalization (two-sided): ∫pdf(ρ)dρ ≈ 1" begin
        d = PCPrior.AR1Correlation(0.9; α = 0.05)
        integral, _ = hcubature([-10.0], [10.0]) do t
            ρ = tanh(t[1])  # maps ℝ → (-1,1)
            dρdt = 1 - tanh(t[1])^2
            lp = logpdf(d, ρ)
            return isfinite(lp) ? exp(lp) * dρdt : 0.0
        end
        @test abs(integral - 1.0) < 0.02
    end

    @testset "rand/logpdf consistency" begin
        d = PCPrior.AR1Correlation(0.9; α = 0.05, positive_only = true)

        # Compare empirical P(ρ < 0.5) against numerical CDF from logpdf
        Random.seed!(42)
        samples = [rand(d) for _ in 1:20_000]
        empirical_cdf = mean(samples .< 0.5)

        cdf_integral, _ = hcubature([-10.0], [0.0]) do t
            # tanh(0) = 0, tanh(-10) ≈ -1, map to (0, 0.5) via ρ = (1+tanh(t))/2
            ρ = (1 + tanh(t[1])) / 2
            ρ >= 0.5 && return 0.0
            dρdt = (1 - tanh(t[1])^2) / 2
            lp = logpdf(d, ρ)
            return isfinite(lp) ? exp(lp) * dρdt : 0.0
        end

        @test abs(empirical_cdf - cdf_integral) < 0.02
    end
end
