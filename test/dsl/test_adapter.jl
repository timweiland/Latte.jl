using Test
using Latte
using DynamicPPL: @model
using Distributions
using LinearAlgebra
using SparseArrays
using Random
using GaussianMarkovRandomFields: precision_matrix

@testset "latte_from_dppl — structural correctness" begin
    # Hierarchical Poisson: τ_u is a hyperparameter; β and u are random.
    # β is unaffected by τ_u (independent prior), u's prior scale depends on
    # τ_u — so the DAG has no edges (both atomic Gaussians with no linear
    # coupling from each other).
    @model function hier_poisson(y, X, group)
        n = length(y)
        p = size(X, 2)
        G = maximum(group)
        τ_u ~ Gamma(2, 1)
        β ~ MvNormal(zeros(p), 100.0 * I(p))
        u ~ MvNormal(zeros(G), (1 / τ_u) * I(G))
        η = X * β .+ u[group]
        for i in 1:n
            y[i] ~ Poisson(exp(η[i]); check_args = false)
        end
    end

    Random.seed!(2026)
    n, p, G = 40, 2, 5
    X = [ones(n) randn(n)]
    group = rand(1:G, n)
    β_true = [0.3, 0.5]
    u_true = randn(G) ./ sqrt(4.0)
    η_true = X * β_true .+ u_true[group]
    y_obs = [rand(Poisson(exp(η))) for η in η_true]

    dppl = hier_poisson(y_obs, X, group)

    @testset "Adapter produces a LatentGaussianModel" begin
        model = latte_from_dppl(dppl; random = (:β, :u))
        @test model isa LatentGaussianModel
        @test keys(model.hyperparameter_spec.free) == (:τ_u,)
        # Fast path triggers LGM auto-augmentation → latent = [η; β; u],
        # length n + p + G. Base (p + G) is recoverable via augmentation_info.
        @test length(model.latent_prior) == n + p + G
        @test model.augmentation_info !== nothing
        @test length(model.augmentation_info.base_latent_indices) == p + G
    end

    @testset "Latent prior has correct sparsity at τ_u = 4" begin
        # The fast path (auto-detected for Poisson+LogLink) wraps the
        # latent in an AugmentedLatentModel; the base model carries the
        # `latent_fn` we want to probe. Force the AD path for this
        # structural check to keep the assertions about the base prior
        # focused on what `build_latent_model` produces.
        model = latte_from_dppl(dppl; random = (:β, :u), force_ad_obs_model = true)
        μ = mean(model.latent_prior; τ_u = 4.0)
        Q = precision_matrix(model.latent_prior; τ_u = 4.0)

        @test length(μ) == p + G

        # β and u are independent a priori so Q should be block-diagonal
        # (the likelihood Hessian pattern gets unioned in as structural
        # zeros, so we check numeric values rather than raw nnz).
        β_idx = 1:p
        u_idx = (p + 1):(p + G)
        @test all(iszero, Q[β_idx, u_idx])
        @test all(iszero, Q[u_idx, β_idx])

        # Diagonal entries: β uses prior precision 1/100 = 0.01
        for i in β_idx
            @test Q[i, i] ≈ 0.01 rtol = 1.0e-10
        end
        # Diagonal entries for u: τ_u = 4
        for i in u_idx
            @test Q[i, i] ≈ 4.0 rtol = 1.0e-10
        end
    end

    @testset "Hyperparameter prior round-trips working ↔ natural" begin
        model = latte_from_dppl(dppl; random = (:β, :u))
        # The prior may be wrapped by the working-space transform; we don't
        # pin its exact type. What matters is that working → natural inverts
        # the log transform correctly.
        wh = WorkingHyperparameters([log(4.0)], model.hyperparameter_spec)
        nat = convert(NaturalHyperparameters, wh)
        @test convert(NamedTuple, nat).τ_u ≈ 4.0 rtol = 1.0e-10
    end
end
