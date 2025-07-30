using Test
using IntegratedNestedLaplace
using IntegratedNestedLaplace: CompositeObservations

@testset "CompositeObservations" begin
    @testset "Constructor and basic properties" begin
        # Test basic construction
        y1 = [1.0, 2.0, 3.0]
        y2 = [4.0, 5.0]
        y_composite = CompositeObservations((y1, y2))

        @test length(y_composite) == 5
        @test eltype(y_composite) == Float64
        @test y_composite isa AbstractVector{Float64}
    end

    @testset "AbstractVector interface" begin
        y1 = [1.0, 2.0, 3.0]
        y2 = [4.0, 5.0]
        y_composite = CompositeObservations((y1, y2))

        # Test indexing
        @test y_composite[1] == 1.0
        @test y_composite[2] == 2.0
        @test y_composite[3] == 3.0
        @test y_composite[4] == 4.0
        @test y_composite[5] == 5.0

        # Test iteration
        collected = collect(y_composite)
        @test collected == [1.0, 2.0, 3.0, 4.0, 5.0]

        # Test size
        @test size(y_composite) == (5,)
    end

    @testset "Edge cases" begin
        # Single component
        y_single = CompositeObservations(([1.0, 2.0],))
        @test length(y_single) == 2
        @test y_single[1] == 1.0

        # Empty components should error
        @test_throws ArgumentError CompositeObservations(())

        # Mixed numeric types should convert to Float64
        y_mixed = CompositeObservations(([1, 2], [3.0, 4.0]))
        @test eltype(y_mixed) == Float64
        @test y_mixed[1] == 1.0
    end

    @testset "Component access" begin
        y1 = [1.0, 2.0]
        y2 = [3.0, 4.0, 5.0]
        y_composite = CompositeObservations((y1, y2))

        # Should be able to access components
        @test length(y_composite.components) == 2
        @test y_composite.components[1] == y1
        @test y_composite.components[2] == y2
    end
end
