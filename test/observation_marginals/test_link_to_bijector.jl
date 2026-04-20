using Test
using Latte
using GaussianMarkovRandomFields
using Bijectors

@testset "LinkFunction to Bijector Mapping" begin
    @testset "LogLink → elementwise(log)" begin
        link = LogLink()
        bij = get_bijector(link)

        # Test that it's the log function
        @test bij(exp(2.0)) ≈ 2.0
        @test bij(exp(5.0)) ≈ 5.0

        # Test inverse (should be exp)
        inv_bij = inverse(bij)
        @test inv_bij(2.0) ≈ exp(2.0)
        @test inv_bij(0.0) ≈ 1.0
        @test inv_bij(-1.0) ≈ exp(-1.0)
    end

    @testset "LogitLink → Bijectors.Logit(0.0, 1.0)" begin
        link = LogitLink()
        bij = get_bijector(link)

        # Should be Logit bijector
        @test bij isa Bijectors.Logit

        # Test logit function (maps (0,1) to R)
        @test bij(0.5) ≈ 0.0  # logit(0.5) = 0
        @test bij(0.731) ≈ 1.0 atol = 0.01  # logit(0.731) ≈ 1

        # Test inverse (logistic function, maps R to (0,1))
        inv_bij = inverse(bij)
        @test inv_bij(0.0) ≈ 0.5  # logistic(0) = 0.5
        @test inv_bij(1.0) ≈ 0.731 atol = 0.01  # logistic(1) ≈ 0.731
        @test inv_bij(-1.0) ≈ 0.269 atol = 0.01  # logistic(-1) ≈ 0.269
    end

    @testset "IdentityLink → identity" begin
        link = IdentityLink()
        bij = get_bijector(link)

        # Should be identity function
        @test bij === identity

        # Test identity behavior
        @test bij(5.0) === 5.0
        @test bij(-3.0) === -3.0
        @test bij(0.0) === 0.0

        # Inverse of identity is identity
        @test inverse(bij) === identity
    end

    @testset "Unsupported link function" begin
        # Create a custom unsupported link
        struct CustomLink <: LinkFunction end

        custom_link = CustomLink()
        @test_throws ErrorException get_bijector(custom_link)
    end
end
