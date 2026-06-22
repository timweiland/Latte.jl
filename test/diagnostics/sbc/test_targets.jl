using Test
using Latte
using Latte: resolve_targets, Hyperparameters, NamedScalars, TargetDescriptor
using DynamicPPL: @model
using Distributions
using GaussianMarkovRandomFields: IIDModel
using LinearAlgebra

# Shared canonical models compile the inference pipeline once for the whole block.
isdefined(@__MODULE__, :sbc_pois) || include("shared_models.jl")

@testset "resolve_targets" begin

    @testset "single scalar hyperparameter" begin
        n = 5
        lgm = latte_from_dppl(sbc_pois(rand(Int, n), n); random = (:x,))
        descs = resolve_targets(Hyperparameters(), lgm)

        @test length(descs) == 1
        d = descs[1]
        @test d isa TargetDescriptor
        @test d.label == :τ
        @test d.sym == :τ
        @test d.index === nothing
    end

    @testset "extract_truth/extract_posterior round-trips correctly" begin
        n = 5
        lgm = latte_from_dppl(sbc_pois(rand(Int, n), n); random = (:x,))
        descs = resolve_targets(Hyperparameters(), lgm)
        d = descs[1]

        truth_nt = (τ = 2.5, x = randn(n))
        @test d.extract_truth(truth_nt) ≈ 2.5

        θ_mat = reshape(collect(1.0:10.0), 10, 1)  # 10 draws, 1 hp
        @test d.extract_posterior(θ_mat) == collect(1.0:10.0)
    end

    @testset "two scalar hyperparameters, preserved order" begin
        n = 4
        lgm = latte_from_dppl(sbc_pois_beta(rand(Int, n), n); random = (:x,))
        descs = resolve_targets(Hyperparameters(), lgm)

        @test length(descs) == 2
        @test [d.label for d in descs] == [:τ, :β]

        # Posterior extraction maps to correct columns
        θ_mat = [1.0 10.0; 2.0 20.0; 3.0 30.0]  # 3 draws
        @test descs[1].extract_posterior(θ_mat) == [1.0, 2.0, 3.0]
        @test descs[2].extract_posterior(θ_mat) == [10.0, 20.0, 30.0]
    end

    @testset "NamedScalars restricts to listed syms" begin
        n = 4
        lgm = latte_from_dppl(sbc_pois_beta(rand(Int, n), n); random = (:x,))
        descs = resolve_targets(NamedScalars(:τ), lgm)

        @test length(descs) == 1
        @test descs[1].label == :τ
    end

    @testset "unknown sym errors" begin
        n = 3
        lgm = latte_from_dppl(sbc_pois(rand(Int, n), n); random = (:x,))
        @test_throws ArgumentError resolve_targets(NamedScalars(:nonexistent), lgm)
    end
end
