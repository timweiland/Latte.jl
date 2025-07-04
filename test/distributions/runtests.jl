# Test runner for custom distributions

using Test

@testset "Distributions Tests" begin
    include("test_weighted_mixture.jl")

    # Future distribution tests can be added here
end
