using Test
using Latte
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

    @testset "mode is the base model ρ=0" begin
        # The PC prior shrinks toward the base AR(1) model (ρ=0), so its mode
        # sits at 0. Without an explicit `mode`, the mode-finder's initial
        # guess falls through to the generic `mode`, which tries to iterate
        # the distribution and throws.
        @test mode(PCPrior.AR1Correlation(0.9; α = 0.05, positive_only = true)) == 0.0
        @test mode(PCPrior.AR1Correlation(0.9; α = 0.05)) == 0.0
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

    @testset "cdf / quantile / median (positive_only)" begin
        d = PCPrior.AR1Correlation(0.9; α = 0.05, positive_only = true)
        @test cdf(d, 0.9) ≈ 0.95 atol = 1.0e-12   # P(ρ < U) = 1-α
        @test cdf(d, 0.0) == 0.0
        @test cdf(d, -0.5) == 0.0
        @test cdf(d, 1.0) == 1.0
        @test cdf(d, 2.0) == 1.0
        cdfs = [cdf(d, x) for x in 0.01:0.02:0.99]
        @test all(diff(cdfs) .> 0)
        for p in (0.05, 0.25, 0.5, 0.75, 0.95)
            @test cdf(d, quantile(d, p)) ≈ p atol = 1.0e-10
        end
        @test quantile(d, 0.95) ≈ 0.9 atol = 1.0e-10
        @test median(d) == quantile(d, 0.5)
        @test 0 < median(d) < 1
        @test insupport(d, median(d))
        for ρ_hi in (0.3, 0.6, 0.85)
            num, _ = hcubature([0.0], [ρ_hi]) do x
                lp = logpdf(d, x[1])
                isfinite(lp) ? exp(lp) : 0.0
            end
            @test num ≈ cdf(d, ρ_hi) atol = 2.0e-3
        end
        @test_throws DomainError quantile(d, -0.1)
        @test_throws DomainError quantile(d, 1.1)
    end

    @testset "cdf / quantile / median (two-sided)" begin
        d = PCPrior.AR1Correlation(0.9; α = 0.05, positive_only = false)
        @test cdf(d, 0.9) ≈ 0.975 atol = 1.0e-12   # symmetric tails split evenly
        @test cdf(d, -0.9) ≈ 0.025 atol = 1.0e-12
        @test cdf(d, 0.0) == 0.5
        for ρ in (0.1, 0.4, 0.8)
            @test cdf(d, -ρ) ≈ 1 - cdf(d, ρ) atol = 1.0e-12
        end
        @test cdf(d, -1.0) == 0.0
        @test cdf(d, 1.0) == 1.0
        cdfs = [cdf(d, x) for x in -0.99:0.02:0.99]
        @test all(diff(cdfs) .> 0)
        for p in (0.05, 0.25, 0.5, 0.75, 0.95)
            @test cdf(d, quantile(d, p)) ≈ p atol = 1.0e-10
        end
        @test median(d) == 0.0          # symmetry; coincides with mode
        @test median(d) == quantile(d, 0.5)
        @test insupport(d, median(d))
        @test insupport(d, mode(d))
        # Integrate only the smooth positive piece (avoid the integrable cusp at
        # ρ=0 that hcubature under-resolves across): ∫[0,0.6] = cdf(0.6) - cdf(0).
        num, _ = hcubature([1.0e-6], [0.6]) do x
            lp = logpdf(d, x[1])
            isfinite(lp) ? exp(lp) : 0.0
        end
        @test num ≈ cdf(d, 0.6) - 0.5 atol = 2.0e-3
    end
end
