using Test

@testset "Observation Models" begin
    include("test_link_functions.jl")
    include("test_exponential_family.jl")
    include("test_custom_models.jl")
    include("test_type_stability.jl")
    include("test_likelihood.jl")
    include("test_linearly_transformed.jl")
    include("composite/runtests.jl")
end
