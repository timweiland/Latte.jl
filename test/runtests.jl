using IntegratedNestedLaplace
using Test
using Aqua

@testset "IntegratedNestedLaplace.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(IntegratedNestedLaplace)
    end

    include("hyperparameters/runtests.jl")
    include("distributions/runtests.jl")
    include("test_inla_model.jl")
    include("hyperparameter_posterior/runtests.jl")
    include("test_latent_marginalization.jl")
    include("end_to_end/runtests.jl")
end
