using Test

@testset "Latent Marginalization" begin
    include("test_simplified_laplace_helpers.jl")
    include("test_simplified_laplace.jl")
    include("test_spline_augmented_gaussian.jl")
    include("test_kld_integration.jl")
    include("test_adaptive_marginal.jl")
    include("test_fixed_tau_quadrature.jl")
end
