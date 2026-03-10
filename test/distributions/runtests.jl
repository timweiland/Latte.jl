# Test runner for custom distributions

using Test

@testset "Distributions Tests" begin
    include("test_skew_normal_ext.jl")
    include("test_weighted_mixture.jl")
    include("test_bijectors.jl")
    include("test_pc_prior_precision.jl")
    include("test_pc_prior_sigma.jl")
    include("test_pc_prior_ar1.jl")
    include("test_pc_prior_bym.jl")
end
