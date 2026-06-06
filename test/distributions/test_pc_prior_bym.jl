using Test
using Latte
using Distributions
using Random
using HCubature
using SparseArrays
using LinearAlgebra

@testset "PCPrior.BYMProportion" begin
    # 6-node graph with varied connectivity (star + tail)
    Q_scaled = Float64[
        3 -1 -1 -1 0 0
        -1 1 0 0 0 0
        -1 0 1 0 0 0
        -1 0 0 2 -1 0
        0 0 0 -1 2 -1
        0 0 0 0 -1 1
    ]

    @testset "Construction" begin
        d = PCPrior.BYMProportion(Q_scaled, 0.5; α = 0.05)
        @test d isa PCPrior.BYMProportion
        @test d.λ > 0
        @test length(d.gamma_inv_m1) == 5  # 6 nodes, 1 zero eigenvalue

        @test_throws ArgumentError PCPrior.BYMProportion(Q_scaled, 0.0; α = 0.05)
        @test_throws ArgumentError PCPrior.BYMProportion(Q_scaled, 1.0; α = 0.05)
        @test_throws ArgumentError PCPrior.BYMProportion(Q_scaled, 0.5; α = 0.0)
    end

    @testset "Support and boundaries" begin
        d = PCPrior.BYMProportion(Q_scaled, 0.5; α = 0.05)
        @test minimum(support(d)) == 0.0
        @test maximum(support(d)) == 1.0
        @test logpdf(d, 0.0) == -Inf
        @test logpdf(d, 1.0) == -Inf
        @test logpdf(d, -0.1) == -Inf
        @test logpdf(d, 1.1) == -Inf
    end

    @testset "Calibration: P(φ > U) ≈ α" begin
        U = 0.5
        α = 0.05
        d = PCPrior.BYMProportion(Q_scaled, U; α = α)
        Random.seed!(123)
        samples = [rand(d) for _ in 1:50_000]
        @test all(s -> 0 < s < 1, samples)
        @test abs(mean(samples .> U) - α) < 0.02
    end

    @testset "Normalization: ∫pdf(φ)dφ ≈ 1" begin
        d = PCPrior.BYMProportion(Q_scaled, 0.5; α = 0.05)
        integral, _ = hcubature([-20.0], [20.0]) do t
            et = exp(t[1])
            φ = et / (1 + et)
            dφdt = et / (1 + et)^2
            lp = logpdf(d, φ)
            return isfinite(lp) ? exp(lp) * dφdt : 0.0
        end
        @test abs(integral - 1.0) < 0.02
    end

    @testset "rand/logpdf consistency" begin
        d = PCPrior.BYMProportion(Q_scaled, 0.5; α = 0.05)

        Random.seed!(42)
        samples = [rand(d) for _ in 1:20_000]
        empirical_cdf = mean(samples .< 0.3)

        # CDF via logit substitution: φ = sigmoid(t), integrate t ∈ (-∞, logit(0.3))
        logit_threshold = log(0.3 / 0.7)
        cdf_integral, _ = hcubature([-20.0], [logit_threshold]) do t
            et = exp(t[1])
            φ = et / (1 + et)
            dφdt = et / (1 + et)^2
            lp = logpdf(d, φ)
            return isfinite(lp) ? exp(lp) * dφdt : 0.0
        end

        @test abs(empirical_cdf - cdf_integral) < 0.02
    end

    @testset "Different graphs produce different priors" begin
        Q_path = Float64[1 -1 0 0; -1 2 -1 0; 0 -1 2 -1; 0 0 -1 1]
        Q_complete = Float64[3 -1 -1 -1; -1 3 -1 -1; -1 -1 3 -1; -1 -1 -1 3]

        d_path = PCPrior.BYMProportion(Q_path, 0.5; α = 0.05)
        d_complete = PCPrior.BYMProportion(Q_complete, 0.5; α = 0.05)

        @test d_path.λ != d_complete.λ
        @test logpdf(d_path, 0.3) != logpdf(d_complete, 0.3)
    end

    @testset "Larger graph (10-node cycle)" begin
        n = 10
        W = spzeros(Int, n, n)
        for i in 1:n
            j = mod1(i + 1, n)
            W[i, j] = 1
            W[j, i] = 1
        end
        D = spdiagm(0 => vec(sum(W; dims = 2)))
        Q = Matrix(Float64.(D - W))

        d = PCPrior.BYMProportion(Q, 0.5; α = 0.05)
        Random.seed!(42)
        samples = [rand(d) for _ in 1:50_000]
        @test all(s -> 0 < s < 1, samples)
        @test abs(mean(samples .> 0.5) - 0.05) < 0.02
    end

    @testset "cdf / quantile / median / mode" begin
        d = PCPrior.BYMProportion(Q_scaled, 0.5; α = 0.05)

        @test cdf(d, 0.0) == 0.0
        @test cdf(d, 1.0) == 1.0
        cvals = cdf.(Ref(d), range(0.0, 1.0; length = 50))
        @test all(diff(cvals) .>= -1.0e-12)        # nondecreasing
        @test all(0 .<= cvals .<= 1)

        for p in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
            @test cdf(d, quantile(d, p)) ≈ p atol = 1.0e-8
        end

        @test median(d) == quantile(d, 0.5)
        @test insupport(d, median(d))
        @test mode(d) == 0.0

        # cdf consistent with logpdf via numerical integration of the density
        for φ in (0.05, 0.2, 0.5, 0.8)
            num, _ = hquadrature(t -> exp(logpdf(d, t)), 1.0e-12, φ; rtol = 1.0e-9)
            @test cdf(d, φ) ≈ num atol = 1.0e-7
        end

        @test quantile(d, 0.0) == 0.0
        @test quantile(d, 1.0) == 1.0
        @test_throws DomainError quantile(d, 1.5)

        # density monotone decreasing ⇒ mode at boundary 0
        @test logpdf(d, 1.0e-3) > logpdf(d, 0.5) > logpdf(d, 0.9)
    end
end
