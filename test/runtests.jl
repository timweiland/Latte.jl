using IntegratedNestedLaplace
using Test
using Aqua

@testset "IntegratedNestedLaplace.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(IntegratedNestedLaplace)
    end
    
    include("hyperparameters/runtests.jl")
    include("observation_models/runtests.jl")
end
