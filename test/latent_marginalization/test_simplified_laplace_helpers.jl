using Test
using IntegratedNestedLaplace
using LinearAlgebra
using SparseArrays
using Random

# Access the internal function for testing
using IntegratedNestedLaplace: _compute_tr

@testset "Simplified Laplace Helper Functions" begin

    Random.seed!(42)

    @testset "_compute_tr with Diagonal" begin
        n = 10
        Σ = Diagonal(rand(n))
        v_i = randn(n)
        σ_i = 2.0
        dMdxi = Diagonal(randn(n))

        result = _compute_tr(Σ, v_i, σ_i, dMdxi)

        # Manual computation
        τ_i = 1 / σ_i^2
        expected = sum(dMdxi.diag .* (Σ.diag .- τ_i .* v_i .^ 2))

        @test result ≈ expected rtol = 1.0e-12
    end

    @testset "_compute_tr with Sparse Diagonal" begin
        n = 10
        Σ = spdiagm(0 => rand(n))
        v_i = randn(n)
        σ_i = 2.0
        dMdxi_diag = Diagonal(randn(n))
        dMdxi_sparse = sparse(dMdxi_diag)

        result_diag = _compute_tr(Σ, v_i, σ_i, dMdxi_diag)
        result_sparse = _compute_tr(Σ, v_i, σ_i, dMdxi_sparse)

        @test result_diag ≈ result_sparse rtol = 1.0e-12
    end

    @testset "_compute_tr with Sparse Matrix (off-diagonal)" begin
        n = 8
        # Create a symmetric sparse covariance matrix
        Random.seed!(123)
        A = sprandn(n, n, 0.3)
        Σ = A * A' + 0.1 * I  # Make it positive definite
        Σ = (Σ + Σ') / 2  # Ensure symmetry

        v_i = randn(n)
        σ_i = 1.5

        # Create a sparse dMdxi with off-diagonal entries
        dMdxi = sprandn(n, n, 0.2)

        result_sparse = _compute_tr(Σ, v_i, σ_i, dMdxi)

        # Manual computation using dense matrices
        τ_i = 1 / σ_i^2
        correction = Σ - τ_i * (v_i * v_i')
        expected = sum((Matrix(dMdxi) .* Matrix(correction)))

        @test result_sparse ≈ expected rtol = 1.0e-10
    end

    @testset "_compute_tr consistency: sparse vs dense" begin
        n = 6
        Random.seed!(456)

        # Symmetric sparse Σ
        A = sprandn(n, n, 0.4)
        Σ = A * A' + 0.1 * I
        Σ = (Σ + Σ') / 2

        v_i = randn(n)
        σ_i = 0.8

        # Sparse dMdxi
        dMdxi_sparse = sprandn(n, n, 0.3)

        # Compute with sparse version
        result_sparse = _compute_tr(Σ, v_i, σ_i, dMdxi_sparse)

        # Compute with dense matrices manually
        τ_i = 1 / σ_i^2
        Σ_dense = Matrix(Σ)
        dMdxi_dense = Matrix(dMdxi_sparse)
        correction = Σ_dense - τ_i * (v_i * v_i')
        expected = sum(dMdxi_dense .* correction)

        @test result_sparse ≈ expected rtol = 1.0e-10
    end

    @testset "_compute_tr with different sparsity patterns" begin
        n = 10
        Random.seed!(789)

        Σ = spdiagm(0 => rand(n), 1 => rand(n - 1), -1 => rand(n - 1))
        Σ = (Σ + Σ') / 2  # Symmetrize

        v_i = randn(n)
        σ_i = 1.0

        # Tridiagonal sparse matrix
        dMdxi = spdiagm(0 => randn(n), 1 => randn(n - 1), -1 => randn(n - 1))

        result = _compute_tr(Σ, v_i, σ_i, dMdxi)

        # Reference computation
        τ_i = 1 / σ_i^2
        correction = Matrix(Σ) - τ_i * (v_i * v_i')
        expected = sum(Matrix(dMdxi) .* correction)

        @test result ≈ expected rtol = 1.0e-10
    end

    @testset "_compute_tr edge cases" begin
        # Single element
        Σ = sparse([1.0;;])
        v_i = [0.5]
        σ_i = 2.0
        dMdxi = sparse([1.5;;])

        result = _compute_tr(Σ, v_i, σ_i, dMdxi)
        τ_i = 1 / σ_i^2
        expected = 1.5 * (1.0 - τ_i * 0.5^2)

        @test result ≈ expected rtol = 1.0e-12

        # Zero sparse matrix
        n = 5
        Σ = spdiagm(0 => ones(n))
        v_i = ones(n)
        σ_i = 1.0
        dMdxi = spzeros(n, n)

        result = _compute_tr(Σ, v_i, σ_i, dMdxi)
        @test result ≈ 0.0
    end
end
