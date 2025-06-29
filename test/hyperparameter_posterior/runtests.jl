using Test

@testset "Hyperparameter Posterior" begin
    include("test_mode_finding.jl")
    include("test_exploration.jl") 
    include("test_interpolation.jl")
    include("test_marginals.jl")
    include("test_type_stability.jl")
end