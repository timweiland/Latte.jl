using Test
using Latte
using Latte: symmetric_kld
using Distributions

@testset "symmetric_kld" begin

    @testset "Identical distributions → SKLD = 0" begin
        d = Normal(1.0, 2.0)
        @test symmetric_kld(d, d) ≈ 0.0 atol = 1.0e-10
    end

    @testset "Two Normals — matches closed-form" begin
        # Closed-form KL(N(μ₁,σ₁) || N(μ₂,σ₂)) = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2
        μ₁, σ₁ = 0.0, 1.0
        μ₂, σ₂ = 0.5, 1.5

        kl_12 = log(σ₂ / σ₁) + (σ₁^2 + (μ₁ - μ₂)^2) / (2 * σ₂^2) - 0.5
        kl_21 = log(σ₁ / σ₂) + (σ₂^2 + (μ₂ - μ₁)^2) / (2 * σ₁^2) - 0.5
        skld_exact = kl_12 + kl_21

        p = Normal(μ₁, σ₁)
        q = Normal(μ₂, σ₂)

        @test symmetric_kld(p, q) ≈ skld_exact rtol = 1.0e-4
    end

    @testset "Symmetry: skld(p,q) == skld(q,p)" begin
        p = Normal(0.0, 1.0)
        q = Normal(1.0, 2.0)
        @test symmetric_kld(p, q) ≈ symmetric_kld(q, p) rtol = 1.0e-10
    end

    @testset "Normal vs SkewNormal with nonzero α → positive SKLD" begin
        p = Normal(0.0, 1.0)
        q = SkewNormal(0.0, 1.0, 3.0)
        skld = symmetric_kld(p, q)
        @test skld > 0.0
        @test isfinite(skld)
    end

    @testset "SkewNormal with α ≈ 0 → near-zero SKLD vs Normal" begin
        p = Normal(0.0, 1.0)
        q = SkewNormal(0.0, 1.0, 0.0)
        @test symmetric_kld(p, q) ≈ 0.0 atol = 1.0e-8
    end

    @testset "Larger difference → larger SKLD" begin
        p = Normal(0.0, 1.0)
        q_close = Normal(0.1, 1.0)
        q_far = Normal(1.0, 1.0)
        @test symmetric_kld(p, q_close) < symmetric_kld(p, q_far)
    end
end
