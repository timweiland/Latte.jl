using Test
using Latte
import Bijectors

@testset "Bijectors" begin
    @testset "PrecisionToStdBijector" begin
        b = PrecisionToStdBijector()

        # Forward transform: σ = 1/√τ
        @test Bijectors.transform(b, 4.0) ≈ 0.5
        @test Bijectors.transform(b, 1.0) ≈ 1.0
        @test Bijectors.transform(b, 0.25) ≈ 2.0

        # Invalid input
        @test_throws ArgumentError Bijectors.transform(b, -1.0)

        # Array transform
        @test Bijectors.transform(b, [4.0, 1.0]) ≈ [0.5, 1.0]

        # logabsdetjac
        @test Bijectors.logabsdetjac(b, 1.0) ≈ log(0.5)
        @test Bijectors.logabsdetjac(b, -1.0) == -Inf

        # with_logabsdet_jacobian
        y, ladj = Bijectors.with_logabsdet_jacobian(b, 4.0)
        @test y ≈ 0.5
        @test ladj ≈ log(0.5) - 1.5 * log(4.0)
    end

    @testset "StdToPrecisionBijector" begin
        b = StdToPrecisionBijector()

        # Forward transform: τ = 1/σ²
        @test Bijectors.transform(b, 0.5) ≈ 4.0
        @test Bijectors.transform(b, 1.0) ≈ 1.0
        @test Bijectors.transform(b, 2.0) ≈ 0.25

        # Invalid input
        @test_throws ArgumentError Bijectors.transform(b, -1.0)

        # logabsdetjac
        @test Bijectors.logabsdetjac(b, 1.0) ≈ log(2)
        @test Bijectors.logabsdetjac(b, -1.0) == -Inf
    end

    @testset "Round-trip" begin
        b_fwd = PrecisionToStdBijector()
        b_inv = StdToPrecisionBijector()

        @test Bijectors.inverse(b_fwd) isa StdToPrecisionBijector
        @test Bijectors.inverse(b_inv) isa PrecisionToStdBijector
        @test inv(b_fwd) isa StdToPrecisionBijector
        @test inv(b_inv) isa PrecisionToStdBijector

        # Round-trip precision → std → precision
        for τ in [0.1, 1.0, 10.0, 100.0]
            σ = Bijectors.transform(b_fwd, τ)
            τ_back = Bijectors.transform(b_inv, σ)
            @test τ_back ≈ τ
        end

        # logabsdetjac should be negatives of each other
        for τ in [0.5, 1.0, 5.0]
            σ = Bijectors.transform(b_fwd, τ)
            ladj_fwd = Bijectors.logabsdetjac(b_fwd, τ)
            ladj_inv = Bijectors.logabsdetjac(b_inv, σ)
            @test ladj_fwd ≈ -ladj_inv
        end
    end
end
