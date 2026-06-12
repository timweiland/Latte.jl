using Test
using Latte
using GaussianMarkovRandomFields
using Distributions
using DynamicPPL
using LinearAlgebra
using Random

# `base_latent_marginals` and `linear_predictor_marginals` must work uniformly
# in BOTH the compact (default) and augmented (`augment = true`) modes, and the
# two modes must agree: compact derives the predictor from the latent posterior
# via the design map, augmented slices the η-block of the augmented latent.
@testset "Marginal accessors: compact and augmented agree" begin
    @latte function hier_poisson(y, X, group)
        τ_u ~ Gamma(2, 1)
        β ~ MvNormal(zeros(size(X, 2)), 100.0 * I(size(X, 2)))
        u ~ IIDModel(maximum(group))(τ = τ_u)
        for i in eachindex(y)
            y[i] ~ Poisson(exp(X[i, :] ⋅ β + u[group[i]]))
        end
    end

    Random.seed!(11)
    n, p, G = 50, 2, 6
    X = [ones(n) randn(n)]
    group = rand(1:G, n)
    β_true = [0.2, 0.4]
    u_true = randn(G) ./ 2
    y = [rand(Poisson(exp(X[i, :] ⋅ β_true + u_true[group[i]]))) for i in 1:n]

    r_compact = inla(hier_poisson(y, X, group), y; progress = false)
    r_aug = inla(hier_poisson(y, X, group; augment = true), y; progress = false)

    @testset "linear_predictor_marginals works in both modes" begin
        lp_c = linear_predictor_marginals(r_compact)
        lp_a = linear_predictor_marginals(r_aug)
        @test length(lp_c) == n
        @test length(lp_a) == n
        @test all(d -> d isa Distribution, lp_c)
        # Compact (VBC) and augmented (SLA) are different marginalizations of the
        # same posterior, so the predictor moments agree closely but not exactly.
        @test maximum(abs.(mean.(lp_c) .- mean.(lp_a))) < 0.15
        @test std.(lp_c) ≈ std.(lp_a) rtol = 1.0e-1
    end

    @testset "base_latent_marginals works in both modes" begin
        b_c = base_latent_marginals(r_compact)
        b_a = base_latent_marginals(r_aug)
        @test length(b_c) == p + G
        @test length(b_a) == p + G
        @test mean.(b_c) ≈ mean.(b_a) rtol = 5.0e-2
    end

    @testset "observation_marginals still consistent with the accessor" begin
        # observation_marginals = link(linear_predictor_marginals); both modes work.
        @test length(observation_marginals(r_compact)) == n
        @test length(observation_marginals(r_aug)) == n
    end
end
