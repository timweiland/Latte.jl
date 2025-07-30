using Test

@testset "Composite Likelihoods" begin
    include("test_composite_observations.jl")
    include("test_composite_model.jl")
    include("test_composite_likelihood.jl")
    include("test_integration.jl")
end
