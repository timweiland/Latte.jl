using Test

@testset "Parallel" begin
    include("test_executors.jl")
    include("test_parallel_hessian.jl")
end
