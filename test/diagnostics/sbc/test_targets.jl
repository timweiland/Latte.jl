using Test
using Latte
using Latte: resolve_targets, Hyperparameters, NamedScalars, TargetDescriptor
using DynamicPPL: @model
using Distributions
using GaussianMarkovRandomFields: IIDModel
using LinearAlgebra

@testset "resolve_targets" begin

    @testset "single scalar hyperparameter" begin
        @model function m1(y, n)
            τ ~ PCPrior.Precision(1.0, α = 0.01)
            x ~ IIDModel(n)(τ = τ)
            for i in eachindex(y)
                y[i] ~ Poisson(exp(x[i]); check_args = false)
            end
        end
        n = 5
        lgm = latte_from_dppl(m1(rand(Int, n), n); random = (:x,))
        descs = resolve_targets(Hyperparameters(), lgm)

        @test length(descs) == 1
        d = descs[1]
        @test d isa TargetDescriptor
        @test d.label == :τ
        @test d.sym == :τ
        @test d.index === nothing
    end

    @testset "extract_truth/extract_posterior round-trips correctly" begin
        @model function m2(y, n)
            τ ~ PCPrior.Precision(1.0, α = 0.01)
            x ~ IIDModel(n)(τ = τ)
            for i in eachindex(y)
                y[i] ~ Poisson(exp(x[i]); check_args = false)
            end
        end
        n = 5
        lgm = latte_from_dppl(m2(rand(Int, n), n); random = (:x,))
        descs = resolve_targets(Hyperparameters(), lgm)
        d = descs[1]

        truth_nt = (τ = 2.5, x = randn(n))
        @test d.extract_truth(truth_nt) ≈ 2.5

        θ_mat = reshape(collect(1.0:10.0), 10, 1)  # 10 draws, 1 hp
        @test d.extract_posterior(θ_mat) == collect(1.0:10.0)
    end

    @testset "two scalar hyperparameters, preserved order" begin
        @model function m3(y, n)
            τ ~ PCPrior.Precision(1.0, α = 0.01)
            β ~ Normal(0, 1)
            x ~ IIDModel(n)(τ = τ)
            for i in eachindex(y)
                y[i] ~ Poisson(exp(β + x[i]); check_args = false)
            end
        end
        n = 4
        lgm = latte_from_dppl(m3(rand(Int, n), n); random = (:x,))
        descs = resolve_targets(Hyperparameters(), lgm)

        @test length(descs) == 2
        @test [d.label for d in descs] == [:τ, :β]

        # Posterior extraction maps to correct columns
        θ_mat = [1.0 10.0; 2.0 20.0; 3.0 30.0]  # 3 draws
        @test descs[1].extract_posterior(θ_mat) == [1.0, 2.0, 3.0]
        @test descs[2].extract_posterior(θ_mat) == [10.0, 20.0, 30.0]
    end

    @testset "NamedScalars restricts to listed syms" begin
        @model function m4(y, n)
            τ ~ PCPrior.Precision(1.0, α = 0.01)
            β ~ Normal(0, 1)
            x ~ IIDModel(n)(τ = τ)
            for i in eachindex(y)
                y[i] ~ Poisson(exp(β + x[i]); check_args = false)
            end
        end
        n = 4
        lgm = latte_from_dppl(m4(rand(Int, n), n); random = (:x,))
        descs = resolve_targets(NamedScalars(:τ), lgm)

        @test length(descs) == 1
        @test descs[1].label == :τ
    end

    @testset "unknown sym errors" begin
        @model function m5(y, n)
            τ ~ PCPrior.Precision(1.0, α = 0.01)
            x ~ IIDModel(n)(τ = τ)
            for i in eachindex(y)
                y[i] ~ Poisson(exp(x[i]); check_args = false)
            end
        end
        n = 3
        lgm = latte_from_dppl(m5(rand(Int, n), n); random = (:x,))
        @test_throws ArgumentError resolve_targets(NamedScalars(:nonexistent), lgm)
    end
end
