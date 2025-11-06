using Test

@testset "Observation Marginals" begin
    include("test_link_to_bijector.jl")
    include("test_transformed_weighted_mixture.jl")
    include("test_observation_marginals_api.jl")
end
