using Test
using Latte
using Latte: adaptive_negative_hessian, pmap_executor
using LinearAlgebra

@testset "Parallel Hessian" begin

    # Test function with known Hessian: f(x) = -0.5 * x' * A * x
    # Hessian = -A, so negative Hessian = A
    A = [
        4.0 1.0 0.5;
        1.0 3.0 0.2;
        0.5 0.2 2.0
    ]
    f(x) = -0.5 * dot(x, A * x)
    x0 = zeros(3)

    @testset "Sequential matches known Hessian" begin
        H = adaptive_negative_hessian(f, x0)
        @test H ≈ A atol = 0.01
    end

    @testset "Parallel matches sequential" begin
        H_seq = adaptive_negative_hessian(f, x0; executor = SequentialExecutor())
        H_par = adaptive_negative_hessian(f, x0; executor = ThreadedExecutor(nworkers = 2))
        @test H_par ≈ H_seq atol = 1.0e-10
    end

    @testset "Parallel Hessian on 4D problem" begin
        B = Diagonal([5.0, 3.0, 2.0, 1.0]) + 0.1 * ones(4, 4)
        g(x) = -0.5 * dot(x, B * x)
        x1 = zeros(4)

        H_seq = adaptive_negative_hessian(g, x1; executor = SequentialExecutor())
        H_par = adaptive_negative_hessian(g, x1; executor = ThreadedExecutor(nworkers = 4))
        @test H_par ≈ H_seq atol = 1.0e-10
        @test H_par ≈ B atol = 0.01
    end

    @testset "compute_reparameterization passes executor" begin
        # Just verify the kwarg is accepted without error
        using Latte: compute_reparameterization, find_hyperparameter_mode
        using GaussianMarkovRandomFields
        using SparseArrays
        using Distributions

        n = 6
        spec = @hyperparams begin
            (σ ~ InverseGamma(2, 1), transform = log, space = natural)
        end
        function latent(; σ, kwargs...)
            Q = spdiagm(0 => fill(1 / σ^2, n))
            return (zeros(n), Q)
        end
        model = LatentGaussianModel(spec, FunctionLatentModel(latent, n), ExponentialFamily(Normal))
        y = randn(n)
        θ_star, _, _ = find_hyperparameter_mode(model, y)

        pool_seq = make_workspace_pool(model.latent_prior; size = 1, σ = 1.0)
        pool_par = make_workspace_pool(model.latent_prior; size = 2, σ = 1.0)
        t_seq = compute_reparameterization(model, y, θ_star; pool = pool_seq, executor = SequentialExecutor())
        t_par = compute_reparameterization(model, y, θ_star; pool = pool_par, executor = ThreadedExecutor(nworkers = 2))

        @test t_seq.H ≈ t_par.H atol = 1.0e-10
        @test t_seq.V ≈ t_par.V atol = 1.0e-10
    end

    @testset "Full inla() — Threaded matches Sequential" begin
        # Phase 2 contract: pool-aware pmap_executor gives numerically
        # identical results under SequentialExecutor vs ThreadedExecutor.
        # Bit-for-bit equality is the expected behavior since all per-task
        # work is deterministic given the same inputs.
        using Random
        using SparseArrays
        using GaussianMarkovRandomFields
        using Distributions

        Random.seed!(2026)
        n = 20
        spec = @hyperparams begin
            (τ ~ Gamma(2, 1), transform = log, space = natural)
            (α ~ Beta(2, 2), transform = logit, space = natural)
        end
        function latent_2hp(; τ, α, kwargs...)
            ρ = 2 * α - 1
            Q = spdiagm(-1 => fill(-ρ * τ, n - 1), 0 => fill((1 + ρ^2) * τ, n), 1 => fill(-ρ * τ, n - 1))
            Q[1, 1] = τ
            Q[n, n] = τ
            (zeros(n), Q)
        end
        model = LatentGaussianModel(spec, FunctionLatentModel(latent_2hp, n), ExponentialFamily(Poisson))
        y = PoissonObservations(rand(Poisson(2.0), n))

        # FiniteDiffStrategy: AD through `latent_2hp` produces a Q whose
        # sparsity pattern can drift from the workspace's Float-pattern at
        # Dual evaluation points (the user's `spdiagm`-based Q construction
        # depends on Dual coefficients), tripping the WorkspaceGMRF pattern
        # check. Same nested-AD limitation as in test_fast_path_agrees.jl.
        # Sidesteps it via finite-difference outer differentiation; this
        # test is about parallel-vs-sequential equivalence, not AD.
        r_seq = inla(
            model, y; progress = false,
            latent_marginalization_method = SimplifiedLaplace(),
            hyperparameter_marginalization_method = AutoHyperparameterMarginal(),
            diff_strategy = FiniteDiffStrategy(),
            executor = SequentialExecutor(),
        )
        r_thr = inla(
            model, y; progress = false,
            latent_marginalization_method = SimplifiedLaplace(),
            hyperparameter_marginalization_method = AutoHyperparameterMarginal(),
            diff_strategy = FiniteDiffStrategy(),
            executor = ThreadedExecutor(nworkers = max(2, min(4, Threads.nthreads()))),
        )

        @test r_seq.exploration.log_normalization_constant ≈ r_thr.exploration.log_normalization_constant atol = 1.0e-10
        @test r_seq.hyperparameter_mode ≈ r_thr.hyperparameter_mode atol = 1.0e-10
        @test length(r_seq.latent_marginals) == length(r_thr.latent_marginals)
        for i in eachindex(r_seq.latent_marginals)
            @test mean(r_seq.latent_marginals[i]) ≈ mean(r_thr.latent_marginals[i]) atol = 1.0e-10
            @test std(r_seq.latent_marginals[i]) ≈ std(r_thr.latent_marginals[i]) atol = 1.0e-10
        end
    end
end
