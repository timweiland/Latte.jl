using IntegratedNestedLaplace
using Test
using Aqua

@testset "IntegratedNestedLaplace.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(IntegratedNestedLaplace)
    end
    
    include("hyperparameters/runtests.jl")
    include("observation_models/runtests.jl")
    include("gaussian_approximation/runtests.jl")
    include("distributions/runtests.jl")
    include("test_inla_model.jl")
    include("hyperparameter_posterior/runtests.jl")
    include("test_marginalization.jl")
end
