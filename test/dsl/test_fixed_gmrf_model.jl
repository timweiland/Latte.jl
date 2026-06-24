using Test
using Latte: FixedGMRFModel
using GaussianMarkovRandomFields:
    GMRF, ConstrainedGMRF, LatentModel,
    precision_matrix, hyperparameters, model_name, constraints, mean
using LinearAlgebra: SymTridiagonal
using SparseArrays: sparse

# `FixedGMRFModel` wraps a hyperparameter-independent GMRF as a `LatentModel`, so
# a fixed Gaussian prior can be embedded in an `@latte` model. These are pure
# contract tests on the wrapper — no inference, no model-specific constructions.
@testset "FixedGMRFModel (fixed GMRF latent prior adapter)" begin
    n = 4
    Q = sparse(SymTridiagonal(fill(2.0, n), fill(-0.5, n - 1)))  # SPD, diagonally dominant
    μ = collect(1.0:n)
    g = GMRF(μ, Q)

    @testset "unconstrained: LatentModel contract" begin
        m = FixedGMRFModel(g)
        @test m isa LatentModel
        @test length(m) == n
        @test hyperparameters(m) == NamedTuple()
        @test model_name(m) == :fixed_gmrf
        @test constraints(m) === nothing
        @test mean(m) == mean(g)
        @test precision_matrix(m) == precision_matrix(g)
        # Hyperparameter-independent: any kwargs are ignored.
        @test precision_matrix(m; τ = 99.0) == precision_matrix(g)
        @test mean(m; τ = 99.0) == mean(g)
    end

    @testset "unconstrained: materialization returns the wrapped GMRF" begin
        m = FixedGMRFModel(g)
        # Identity, not a re-wrap: preserves the concrete type and never trips
        # the dense-cov guard the MvNormal coercion path hit.
        @test m() === g
        @test m(; τ = 99.0) === g
    end

    @testset "constrained: the linear constraint is threaded through" begin
        A = reshape(fill(1.0, n), 1, n)   # sum-to-zero
        e = [0.0]
        cg = ConstrainedGMRF(g, A, e)
        m = FixedGMRFModel(cg)

        @test length(m) == n
        # The constraint is reported via `constraints`, not folded into mean/Q.
        @test constraints(m) == (A, e)
        # mean / precision are the *base* (unconstrained) values, matching the
        # LatentModel contract — not the constrained marginal mean.
        @test mean(m) == mean(g)
        @test mean(m) != mean(cg)
        @test precision_matrix(m) == precision_matrix(g)
        # Materialization preserves the ConstrainedGMRF itself.
        @test m() === cg
    end
end
