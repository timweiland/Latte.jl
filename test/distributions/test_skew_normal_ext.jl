using Test
using IntegratedNestedLaplace
using Distributions
using HCubature: hcubature

@testset "SkewNormal cdf and quantile" begin

    @testset "cdf: α=0 reduces to Normal" begin
        d_sn = SkewNormal(0.0, 1.0, 0.0)
        d_n = Normal(0.0, 1.0)

        for x in [-3.0, -1.0, 0.0, 1.0, 3.0]
            @test cdf(d_sn, x) ≈ cdf(d_n, x) atol = 1.0e-14
        end
    end

    @testset "cdf: non-zero location and scale with α=0" begin
        d_sn = SkewNormal(2.0, 3.0, 0.0)
        d_n = Normal(2.0, 3.0)

        for x in [-4.0, 0.0, 2.0, 5.0, 8.0]
            @test cdf(d_sn, x) ≈ cdf(d_n, x) atol = 1.0e-14
        end
    end

    @testset "cdf: monotonicity" begin
        for α in [-5.0, -1.0, 0.0, 1.0, 5.0]
            d = SkewNormal(0.0, 1.0, α)
            xs = range(-4.0, 4.0; length = 50)
            cdf_vals = [cdf(d, x) for x in xs]
            @test all(diff(cdf_vals) .>= -1.0e-14)
        end
    end

    @testset "cdf: boundary values" begin
        for α in [-3.0, 0.0, 3.0]
            d = SkewNormal(0.0, 1.0, α)
            @test cdf(d, -20.0) < 1.0e-10
            @test cdf(d, 20.0) > 1.0 - 1.0e-10
        end
    end

    @testset "cdf: validate against quadgk" begin
        test_dists = [
            SkewNormal(0.0, 1.0, 2.0),
            SkewNormal(0.0, 1.0, -3.0),
            SkewNormal(1.0, 2.0, 5.0),
            SkewNormal(-1.0, 0.5, -1.0),
        ]

        for d in test_dists
            for x in [mean(d) - 2 * std(d), mean(d), mean(d) + 2 * std(d)]
                # Integrate pdf from far left tail to x
                lo = mean(d) - 10 * std(d)
                expected, _ = hcubature(t -> pdf(d, t[1]), [lo], [x])
                @test cdf(d, x) ≈ expected atol = 1.0e-8
            end
        end
    end

    @testset "quantile: roundtrip cdf(d, quantile(d, p)) ≈ p" begin
        for α in [-5.0, -1.0, 0.0, 1.0, 5.0]
            d = SkewNormal(0.0, 1.0, α)
            for p in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
                x = quantile(d, p)
                @test cdf(d, x) ≈ p atol = 1.0e-10
            end
        end
    end

    @testset "quantile: α=0 matches Normal" begin
        d_sn = SkewNormal(0.0, 1.0, 0.0)
        d_n = Normal(0.0, 1.0)

        for p in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
            @test quantile(d_sn, p) ≈ quantile(d_n, p) atol = 1.0e-10
        end
    end

    @testset "quantile: non-zero location and scale" begin
        d = SkewNormal(3.0, 2.0, 4.0)
        for p in [0.1, 0.5, 0.9]
            x = quantile(d, p)
            @test cdf(d, x) ≈ p atol = 1.0e-10
        end
    end

    @testset "median consistency" begin
        for α in [-3.0, 0.0, 3.0]
            d = SkewNormal(0.0, 1.0, α)
            @test quantile(d, 0.5) ≈ median(d) atol = 1.0e-10
        end
    end
end
