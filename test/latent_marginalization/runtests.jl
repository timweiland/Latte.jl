using Test

@testset "Latent Marginalization" begin
    include("test_simplified_laplace_helpers.jl")
    include("test_simplified_laplace.jl")
    include("test_spline_augmented_gaussian.jl")
end
