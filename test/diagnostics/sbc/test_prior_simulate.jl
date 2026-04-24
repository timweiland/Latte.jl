using Test
using Latte
using Latte: _prior_simulate, SBCReplicate
using DynamicPPL: @model
using Distributions
using StableRNGs
using GaussianMarkovRandomFields: IIDModel

@testset "_prior_simulate" begin

    @testset "scalar y, scalar params" begin
        @model function demo(y, n)
            τ ~ Gamma(2, 1)
            μ ~ Normal(0, 1 / sqrt(τ))
            for i in eachindex(y)
                y[i] ~ Poisson(exp(μ); check_args = false)
            end
        end
        n = 5
        y_proto = Vector{Missing}(missing, n)
        build = y -> demo(y, n)
        rep = _prior_simulate(build, y_proto, StableRNG(42); replicate_id = 7)

        @test rep isa SBCReplicate
        @test rep.replicate_id == 7
        @test length(rep.y) == n
        @test eltype(rep.y) <: Integer
        @test haskey(rep.truth, :τ)
        @test haskey(rep.truth, :μ)
        @test !haskey(rep.truth, :y)  # y lives on the field, not the truth
    end

    @testset "determinism: same seed → same draw" begin
        @model function tiny(y, n)
            τ ~ Gamma(2, 1)
            for i in eachindex(y)
                y[i] ~ Normal(0, 1 / sqrt(τ))
            end
        end
        build = y -> tiny(y, 3)
        y_proto = Vector{Missing}(missing, 3)

        r1 = _prior_simulate(build, y_proto, StableRNG(9); replicate_id = 1)
        r2 = _prior_simulate(build, y_proto, StableRNG(9); replicate_id = 1)
        @test r1.truth == r2.truth
        @test r1.y == r2.y
    end

    @testset "latent field is preserved in truth" begin
        @model function lgm_model(y, n)
            τ ~ Gamma(2, 1)
            x ~ IIDModel(n)(τ = τ)
            for i in eachindex(y)
                y[i] ~ Poisson(exp(x[i]); check_args = false)
            end
        end
        n = 4
        build = y -> lgm_model(y, n)
        y_proto = Vector{Missing}(missing, n)
        rep = _prior_simulate(build, y_proto, StableRNG(1); replicate_id = 1)

        @test haskey(rep.truth, :τ)
        @test haskey(rep.truth, :x)
        @test length(rep.truth.x) == n
    end
end
