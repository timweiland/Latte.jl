using Test

@testset "Hyperparameter Posterior" begin
    include("test_mode_finding.jl")
    include("test_exploration.jl")
    include("test_interpolation.jl")
    include("test_hyperparameter_marginals.jl")
    include("test_hyperparameter_marginal_distribution.jl")
    include("test_type_stability.jl")
    include("test_ccd_interpolant.jl")
    include("test_spline_marginals.jl")
end
