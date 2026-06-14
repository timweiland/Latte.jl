using Test
using Latte
using Distributions
using Statistics
import Bijectors

@testset "pushforward (back-transform a marginal)" begin
    base = Normal(0.0, 1.0)

    @testset "integrated mean avoids the Jensen trap" begin
        # X ~ N(0,1) ⇒ E[exp X] = exp(μ + σ²/2) = exp(0.5), NOT exp(E[X]) = 1.
        d = pushforward(base, exp)
        @test mean(d) ≈ exp(0.5) rtol = 1.0e-3
        @test !isapprox(mean(d), exp(mean(base)); atol = 0.1)   # not the naive value
        @test quantile(d, 0.5) ≈ 1.0 rtol = 1.0e-2              # median exp(0) = 1
        @test std(d) > 0
    end

    @testset "identity and log round-trips" begin
        @test mean(pushforward(base, identity)) ≈ mean(base) atol = 1.0e-6
        # log of a positive base: use a LogNormal so log(X) ~ Normal(0,1), E = 0.
        ln = LogNormal(0.0, 1.0)
        @test mean(pushforward(ln, log)) ≈ 0.0 atol = 1.0e-2
    end

    @testset "generic Bijectors bijector" begin
        # Y = logit(U), U ~ Uniform(0,1) ⇒ Y ~ standard logistic, mean 0.
        d = pushforward(Uniform(0.0, 1.0), Bijectors.Logit(0.0, 1.0))
        @test mean(d) ≈ 0.0 atol = 1.0e-2
    end

    @testset "wraps any 1-D base (generalized TransformedWeightedMixture)" begin
        # The base need not be a WeightedMixture anymore.
        d = pushforward(Gamma(2.0, 1.5), identity)
        @test d isa ContinuousUnivariateDistribution
        @test mean(d) ≈ mean(Gamma(2.0, 1.5)) atol = 1.0e-6
    end
end
