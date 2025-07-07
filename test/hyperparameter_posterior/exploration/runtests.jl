using Test

@testset "Exploration Module Tests" begin
    include("test_transformation.jl")
    include("test_utils.jl")
    include("test_algorithms.jl")
end
