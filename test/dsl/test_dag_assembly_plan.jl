using Test
using Latte
using Latte: DAGAssemblyPlan, build_dag_assembly_plan, assemble_values!, assemble_mean
using Latte: assemble_joint, CachedDAGLatentModel
using Distributions
using DynamicPPL
using LinearAlgebra
using SparseArrays
using Random
import ForwardDiff
import GaussianMarkovRandomFields as GMRF

# Equivalence between the cached `DAGAssemblyPlan`-based path and the
# original `assemble_joint`. The plan caches the joint sparsity pattern
# at LGM build time; per-call assembly fills values into a pre-allocated
# nzval buffer instead of doing CSC `setindex!` storms.

@testset "DAG assembly plan" begin
    Random.seed!(20260513)

    @testset "Plan reproduces assemble_joint for single block (no edges)" begin
        # `β ~ MvNormal(zeros(3), 1.0)` — single random sym, no DAG edges.
        random_syms = (:β,)
        dims = Dict(:β => 3)
        edges = Dict(:β => Symbol[])
        linear_maps = Dict{Tuple{Symbol, Symbol}, NamedTuple}()
        cond_Qs = Dict(:β => sparse(Matrix(I(3) * 2.0)))
        intercepts = Dict(:β => [0.1, 0.2, 0.3])

        legacy = assemble_joint(random_syms, dims, edges, linear_maps, intercepts, cond_Qs)

        plan = build_dag_assembly_plan(
            random_syms, dims, edges, linear_maps, cond_Qs; lik_pattern = nothing,
        )
        nzval = zeros(plan.nnz)
        assemble_values!(nzval, plan, cond_Qs, linear_maps)
        Q_new = SparseMatrixCSC(plan.n_total, plan.n_total, plan.colptr, plan.rowval, nzval)
        μ_new = assemble_mean(plan, intercepts, linear_maps)

        @test μ_new ≈ legacy.μ
        @test Matrix(Q_new) ≈ Matrix(legacy.Q)
    end

    @testset "Plan reproduces assemble_joint with one parent-child edge" begin
        # `u` (parent), `v = A·u + noise` (child).
        random_syms = (:u, :v)
        dims = Dict(:u => 2, :v => 3)
        edges = Dict(:u => Symbol[], :v => [:u])
        A = sparse([1.0 0.5; 0.2 1.0; 0.7 0.3])
        linear_maps = Dict((:v, :u) => (A = A, b = zeros(3), linear = true))
        cond_Qs = Dict(
            :u => sparse(Matrix(I(2) * 1.0)),
            :v => sparse(Matrix(I(3) * 4.0)),
        )
        intercepts = Dict(:u => [0.5, -0.3], :v => [0.0, 0.0, 0.0])

        legacy = assemble_joint(random_syms, dims, edges, linear_maps, intercepts, cond_Qs)

        plan = build_dag_assembly_plan(
            random_syms, dims, edges, linear_maps, cond_Qs; lik_pattern = nothing,
        )
        nzval = zeros(plan.nnz)
        assemble_values!(nzval, plan, cond_Qs, linear_maps)
        Q_new = SparseMatrixCSC(plan.n_total, plan.n_total, plan.colptr, plan.rowval, nzval)
        μ_new = assemble_mean(plan, intercepts, linear_maps)

        @test μ_new ≈ legacy.μ
        @test Matrix(Q_new) ≈ Matrix(legacy.Q)
    end

    @testset "Plan unions with lik_pattern (no runtime augment)" begin
        # Latent: single block. lik_pattern adds a cross-coupling that
        # wouldn't otherwise exist in Q. The plan should pre-allocate
        # those entries (as zeros) so runtime augment_pattern is a no-op.
        random_syms = (:β,)
        dims = Dict(:β => 3)
        edges = Dict(:β => Symbol[])
        linear_maps = Dict{Tuple{Symbol, Symbol}, NamedTuple}()
        cond_Qs = Dict(:β => sparse(Diagonal([1.0, 2.0, 3.0])))
        intercepts = Dict(:β => zeros(3))

        # Mock obs Hessian pattern: a [1,3] coupling not in cond_Qs.
        lik_pattern = sparse([1, 3], [3, 1], ones(2), 3, 3)

        plan = build_dag_assembly_plan(
            random_syms, dims, edges, linear_maps, cond_Qs;
            lik_pattern = lik_pattern,
        )
        nzval = zeros(plan.nnz)
        assemble_values!(nzval, plan, cond_Qs, linear_maps)
        Q_new = SparseMatrixCSC(plan.n_total, plan.n_total, plan.colptr, plan.rowval, nzval)

        # Pattern must include the lik_pattern entries (with value 0).
        @test Q_new[1, 3] == 0.0
        @test Q_new[3, 1] == 0.0
        # Stored entries — must include (1,3) and (3,1) in the structure.
        nz_pairs = [(Q_new.rowval[k], j) for j in 1:3 for k in Q_new.colptr[j]:(Q_new.colptr[j + 1] - 1)]
        @test (1, 3) in nz_pairs
        @test (3, 1) in nz_pairs
        # And the diagonal entries are correct (matches cond_Qs[:β]).
        @test Q_new[1, 1] == 1.0
        @test Q_new[2, 2] == 2.0
        @test Q_new[3, 3] == 3.0
    end

    @testset "assemble_values! propagates ForwardDiff Duals" begin
        # Smoke: cond_Qs with Dual eltype → nzval(Dual) → resulting Q has
        # Dual values; matches what assemble_joint would compute.
        random_syms = (:u, :v)
        dims = Dict(:u => 2, :v => 2)
        edges = Dict(:u => Symbol[], :v => [:u])
        A = sparse([1.0 0.0; 0.0 1.0])
        linear_maps = Dict((:v, :u) => (A = A, b = zeros(2), linear = true))

        τ_u = ForwardDiff.Dual{:tag}(1.5, 1.0, 0.0)
        τ_v = ForwardDiff.Dual{:tag}(2.5, 0.0, 1.0)
        cond_Qs = Dict(
            :u => sparse(τ_u * Matrix{typeof(τ_u)}(I, 2, 2)),
            :v => sparse(τ_v * Matrix{typeof(τ_v)}(I, 2, 2)),
        )
        intercepts = Dict(:u => zeros(typeof(τ_u), 2), :v => zeros(typeof(τ_v), 2))

        # Build plan at probe time using primal patterns.
        cond_Qs_probe = Dict(
            :u => sparse(Matrix{Float64}(I, 2, 2)),
            :v => sparse(Matrix{Float64}(I, 2, 2)),
        )
        plan = build_dag_assembly_plan(
            random_syms, dims, edges, linear_maps, cond_Qs_probe; lik_pattern = nothing,
        )

        T = ForwardDiff.Dual{:tag, Float64, 2}
        nzval = zeros(T, plan.nnz)
        assemble_values!(nzval, plan, cond_Qs, linear_maps)
        Q_new = SparseMatrixCSC(plan.n_total, plan.n_total, plan.colptr, plan.rowval, nzval)

        legacy = assemble_joint(random_syms, dims, edges, linear_maps, intercepts, cond_Qs)
        @test eltype(Q_new) == eltype(legacy.Q)
        @test Matrix(ForwardDiff.value.(Q_new)) ≈ Matrix(ForwardDiff.value.(legacy.Q))
        @test Matrix(ForwardDiff.partials.(Q_new)) ≈ Matrix(ForwardDiff.partials.(legacy.Q))
    end

    @testset "CachedDAGLatentModel matches FunctionLatentModel end-to-end" begin
        @latte function hierarchical_regression(y, X)
            σ ~ Gamma(2.0, 1.0)
            τ ~ Gamma(2.0, 1.0)
            β ~ MvNormal(zeros(size(X, 2)), 1.0 / τ)
            for i in eachindex(y)
                y[i] ~ Normal(dot(X[i, :], β), σ)
            end
        end

        n, p = 8, 3
        X = randn(n, p)
        y = randn(n)
        lgm = hierarchical_regression(y, X)
        # Fast-path augmentation wraps the base latent prior in an
        # `AugmentedLatentModel`. Underneath should be our new
        # `CachedDAGLatentModel`.
        base = lgm.latent_prior isa CachedDAGLatentModel ?
            lgm.latent_prior : lgm.latent_prior.base_model
        @test base isa CachedDAGLatentModel

        # Pick a hp configuration and check (μ, Q) on the base latent prior.
        σ_val, τ_val = 0.7, 1.3
        μ_new = mean(base; σ = σ_val, τ = τ_val)
        Q_new = GMRF.precision_matrix(base; σ = σ_val, τ = τ_val)

        # Hand-compute expected Q. `MvNormal(μ, σ::Real)` treats σ as
        # standard deviation, so cov = σ² I = (1/τ)² I and precision = τ² I.
        @test μ_new ≈ zeros(p) atol = 1.0e-12
        @test Matrix(Q_new) ≈ (τ_val^2) .* Matrix(I, p, p) atol = 1.0e-12
    end

    @testset "Pattern invariance check fires when cond_Qs pattern changes" begin
        # Build a plan against a tridiagonal probe, then call with a
        # densified cond_Qs — the assemble path should detect the mismatch.
        random_syms = (:u,)
        dims = Dict(:u => 3)
        edges = Dict(:u => Symbol[])
        linear_maps = Dict{Tuple{Symbol, Symbol}, NamedTuple}()
        tridiag = sparse([1, 1, 2, 2, 2, 3, 3], [1, 2, 1, 2, 3, 2, 3], ones(7), 3, 3)
        cond_Qs_probe = Dict(:u => tridiag)
        plan = build_dag_assembly_plan(
            random_syms, dims, edges, linear_maps, cond_Qs_probe; lik_pattern = nothing,
        )

        # Densified version (4 extra stored entries) — pattern mismatch.
        dense = sparse(Matrix(I, 3, 3) .+ ones(3, 3))
        cond_Qs_bad = Dict(:u => dense)
        nzval = zeros(plan.nnz)
        @test_throws Exception assemble_values!(
            nzval, plan, cond_Qs_bad, linear_maps;
            check_pattern_invariance = true,
        )
    end
end
