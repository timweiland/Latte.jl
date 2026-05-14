using Test

@testset "Hyperparameter Posterior" begin
    include("test_mode_finding.jl")
    include("test_mode_init.jl")
    include("test_exploration.jl")
    include("test_type_stability.jl")
    include("test_ccd_interpolant.jl")
    include("test_spline_marginals.jl")
end
