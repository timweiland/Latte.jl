using Test
using StatsModels
using DataFrames
using LinearAlgebra

@testset "RandomWalkTerm Tests" begin

    # Create test data
    df = DataFrame(
        y = randn(20),
        time = repeat(1:5, 4),  # 5 time points, 4 obs each
        x = randn(20)
    )

    @testset "Term Construction" begin
        # Test direct construction
        rw1_term = RandomWalkTerm{1}(:time)
        @test rw1_term.variable == :time
        @test repr(rw1_term) == "RandomWalk{1}(time)"

        rw2_term = RandomWalkTerm{2}(:time)
        @test repr(rw2_term) == "RandomWalk{2}(time)"
    end

    @testset "StatsModels Integration" begin
        # Test termvars
        rw_term = RandomWalkTerm{1}(:time)
        @test StatsModels.termvars(rw_term) == [:time]

        # Test modelcols
        design_cols = StatsModels.modelcols(rw_term, df)
        @test size(design_cols) == (20, 5)  # 20 obs, 5 time points
        @test sum(design_cols, dims = 2) == ones(20, 1)  # Each row sums to 1
        @test sum(design_cols, dims = 1) == [4 4 4 4 4]  # Each time point has 4 obs

        # Test specific mapping
        @test design_cols[1, 1] == 1.0  # First obs maps to first time
        @test design_cols[6, 1] == 1.0  # Sixth obs maps to first time (repeating pattern)
        @test design_cols[2, 2] == 1.0  # Second obs maps to second time
    end

    @testset "Formula Parsing" begin
        # Test that RandomWalk function creates the right arguments
        args = RandomWalk(1, :time)
        @test args == (1, :time)

        # Test formula parsing (this tests the apply_schema method)
        f = @formula(y ~ x + RandomWalk(1, time))

        # Check that we have the right terms
        @test length(f.rhs) == 2  # x + RandomWalk(1, time)

        # Find the RandomWalk function term
        rw_function_term = nothing
        for term in f.rhs
            if isa(term, StatsModels.FunctionTerm{typeof(RandomWalk)})
                rw_function_term = term
                break
            end
        end
        @test rw_function_term !== nothing

        # Test apply_schema transformation
        schema = StatsModels.Schema()
        transformed = StatsModels.apply_schema(rw_function_term, schema, StatsModels.MatrixTerm)
        @test isa(transformed, RandomWalkTerm{1})
        @test transformed.variable == :time
    end

    @testset "GMRF Block Construction" begin
        rw1_term = RandomWalkTerm{1}(:time)
        rw2_term = RandomWalkTerm{2}(:time)

        θ_test = Dict(:τ_rw => 2.0)

        # Test RW1 precision matrix
        Q1 = gmrf_block(rw1_term, df, θ_test)
        @test size(Q1) == (5, 5)  # 5 time points
        @test Q1[1, 1] == 4.0  # 2 * τ = 2 * 2.0 = 4.0
        @test Q1[1, 2] == -2.0  # -τ = -2.0
        @test Q1[2, 1] == -2.0  # Symmetric
        @test Q1[3, 3] == 4.0  # Interior point: 2 * τ
        @test Q1[1, 3] == 0.0  # Non-adjacent elements are zero

        # Verify it's tridiagonal
        @test all(Q1[i, j] == 0.0 for i in 1:5, j in 1:5 if abs(i - j) > 1)

        # Test RW2 precision matrix
        Q2 = gmrf_block(rw2_term, df, θ_test)
        @test size(Q2) == (5, 5)
        @test Q2[1, 1] == 2.0  # τ = 2.0
        @test Q2[1, 2] == -4.0  # -2 * τ = -4.0
        @test Q2[1, 3] == 2.0  # τ = 2.0

        # Test eigenvalues (should all be non-negative for valid precision matrix)
        eigs1 = eigvals(Matrix(Q1))
        @test all(eigs1 .>= -1.0e-10)  # Allow for small numerical errors

        eigs2 = eigvals(Matrix(Q2))
        @test all(eigs2 .>= -1.0e-10)

        # Test hyperparameters function
        @test hyperparameters(rw1_term) == [:τ_rw]
        @test hyperparameters(rw2_term) == [:τ_rw]
    end

    @testset "Edge Cases" begin
        # Test RW2 with insufficient time points
        small_df = DataFrame(time = [1, 2], y = [1.0, 2.0])
        rw2_term = RandomWalkTerm{2}(:time)
        θ_test = Dict(:τ_rw => 1.0)

        @test_throws ErrorException gmrf_block(rw2_term, small_df, θ_test)

        # Test missing hyperparameter (should use default)
        rw1_term = RandomWalkTerm{1}(:time)
        Q_default = gmrf_block(rw1_term, df, Dict())
        @test Q_default[1, 1] == 2.0  # Default τ = 1.0, so 2 * 1.0 = 2.0
    end

end
