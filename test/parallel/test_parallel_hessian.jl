using Test
using IntegratedNestedLaplace
using IntegratedNestedLaplace: adaptive_negative_hessian, pmap_executor
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
        using IntegratedNestedLaplace: compute_reparameterization, find_hyperparameter_mode
        using GaussianMarkovRandomFields
        using SparseArrays
        using Distributions

        n = 6
        spec = @hyperparams begin
            (σ ~ InverseGamma(2, 1), transform = log, space = natural)
        end
        function latent(; σ, kwargs...)
            Q = spdiagm(0 => fill(1 / σ^2, n))
            return GMRF(zeros(n), Q)
        end
        model = INLAModel(spec, FunctionLatentModel(latent, n), ExponentialFamily(Normal))
        y = randn(n)
        θ_star, _, _ = find_hyperparameter_mode(model, y)

        t_seq = compute_reparameterization(model, y, θ_star; executor = SequentialExecutor())
        t_par = compute_reparameterization(model, y, θ_star; executor = ThreadedExecutor(nworkers = 2))

        @test t_seq.H ≈ t_par.H atol = 1.0e-10
        @test t_seq.V ≈ t_par.V atol = 1.0e-10
    end
end
