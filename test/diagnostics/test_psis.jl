using Test
using Latte
using Distributions
using Random

# Exercise the PSIS primitives on synthetic distributions with known tail
# properties. Results match theory to within MC error at n=1000.
@testset "PSIS primitives on synthetic log-weights" begin
    @testset "ess_is / rel_ess_is" begin
        # Uniform weights → max ESS
        log_w = zeros(1000)
        @test ess_is(log_w) ≈ 1000 atol = 1.0e-6
        @test rel_ess_is(log_w) ≈ 1.0 atol = 1.0e-6

        # One weight dominates → ESS → 1
        log_w = fill(-100.0, 1000)
        log_w[1] = 0.0
        @test ess_is(log_w) ≈ 1.0 atol = 1.0e-4
        @test rel_ess_is(log_w) < 0.01
    end

    @testset "pareto_k identifies heavy tails" begin
        Random.seed!(2029)

        # Light-tailed: log(U(0,1)) = -Exp(1) → k̂ should be small
        log_w_light = log.(rand(1000))
        k_light = pareto_k(log_w_light)
        @test k_light < 0.5

        # Pareto(α=1) has very heavy tails → k̂ should spike (≳ 0.7 → unreliable)
        log_w_heavy = log.(rand(Pareto(1.0, 1.0), 1000))
        k_heavy = pareto_k(log_w_heavy)
        @test k_heavy > 0.5   # should be flagged as at least acceptable-borderline

        # Intermediate: Pareto(α=2) → true k ≈ 0.5
        log_w_mid = log.(rand(Pareto(2.0, 1.0), 1000))
        k_mid = pareto_k(log_w_mid)
        @test 0.2 < k_mid < 0.8
    end

    @testset "trust_verdict thresholds" begin
        @test trust_verdict(0.9) === :excellent
        @test trust_verdict(0.5001) === :excellent
        @test trust_verdict(0.3) === :acceptable
        @test trust_verdict(0.2001) === :acceptable
        @test trust_verdict(0.1) === :unreliable
        @test trust_verdict(0.0) === :unreliable
    end

    @testset "pareto_k returns NaN for too-short tails" begin
        # Tiny log_w → tail too short to fit
        @test isnan(pareto_k([1.0, 2.0, 3.0]))
    end
end
