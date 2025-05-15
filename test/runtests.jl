using IntegratedNestedLaplace
using Test
using Aqua

@testset "IntegratedNestedLaplace.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(IntegratedNestedLaplace)
    end
    # Write your tests here.
end
