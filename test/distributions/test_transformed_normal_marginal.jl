using Test
using Latte
using Distributions
using Bijectors
using Random
using Statistics

@testset "TransformedNormalMarginal" begin

    @testset "Matches LogNormal under elementwise(log)" begin
        # `inv_transform = exp` means the natural-space density is
        # exactly LogNormal; the wrapper should match it to machine
        # precision.
        for (μ, σ) in [(0.0, 1.0), (1.0, 0.5), (-2.0, 0.3), (3.0, 2.0)]
            tr = elementwise(log)
            d = TransformedNormalMarginal(μ, σ, tr, inverse(tr))
            ln = LogNormal(μ, σ)

            @test mean(d) ≈ mean(ln) atol = 1.0e-8
            @test std(d) ≈ std(ln) atol = 1.0e-8
            @test median(d) ≈ median(ln) atol = 1.0e-12
            for p in (0.025, 0.25, 0.5, 0.75, 0.975)
                @test quantile(d, p) ≈ quantile(ln, p) atol = 1.0e-12
            end
            for x in (0.5, 1.0, 2.5)
                @test pdf(d, x) ≈ pdf(ln, x) atol = 1.0e-10
                @test cdf(d, x) ≈ cdf(ln, x) atol = 1.0e-12
            end
        end
    end

    @testset "Matches Normal under identity" begin
        d = TransformedNormalMarginal(2.5, 0.7, identity, identity)
        n = Normal(2.5, 0.7)
        @test mean(d) ≈ mean(n) atol = 1.0e-10
        @test std(d) ≈ std(n) atol = 1.0e-10
        @test quantile(d, 0.025) ≈ quantile(n, 0.025) atol = 1.0e-12
        @test quantile(d, 0.975) ≈ quantile(n, 0.975) atol = 1.0e-12
    end

    @testset "Quantile push-through is exact for monotone transforms" begin
        tr = Bijectors.Logit(0.0, 1.0)
        inv_tr = inverse(tr)
        d = TransformedNormalMarginal(0.5, 1.2, tr, inv_tr)
        n = Normal(0.5, 1.2)
        for p in (0.025, 0.1, 0.5, 0.9, 0.975)
            @test quantile(d, p) ≈ inv_tr(quantile(n, p)) atol = 1.0e-12
            # Result lies in (0,1) for the logit transform.
            @test 0 < quantile(d, p) < 1
        end
    end

    @testset "Out-of-support evaluation matches standard Distribution semantics" begin
        d_log = TransformedNormalMarginal(0.0, 1.0, elementwise(log), inverse(elementwise(log)))
        for x in (-1.0, -100.0, 0.0)
            @test cdf(d_log, x) == 0.0
            @test logcdf(d_log, x) == -Inf
            @test pdf(d_log, x) == 0.0
            @test logpdf(d_log, x) == -Inf
        end

        b = Bijectors.Logit(0.0, 1.0)
        d_logit = TransformedNormalMarginal(0.0, 1.0, b, inverse(b))
        for x in (-0.1, -1.0)
            @test cdf(d_logit, x) == 0.0
            @test pdf(d_logit, x) == 0.0
        end
        for x in (1.1, 5.0)
            @test cdf(d_logit, x) == 1.0
            @test pdf(d_logit, x) == 0.0
        end
    end

    @testset "rand draws are in the natural-space support" begin
        rng = MersenneTwister(0)
        d = TransformedNormalMarginal(0.0, 1.0, elementwise(log), inverse(elementwise(log)))
        for _ in 1:50
            @test rand(rng, d) > 0
        end
    end

end
