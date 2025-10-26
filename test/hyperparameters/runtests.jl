# Test files organized to mirror src/hyperparameters structure

@testset "Hyperparameters" begin
    include("test_hyperparameter.jl")
    include("test_hyperparameter_spec.jl")
    include("test_working_and_natural.jl")
    include("test_logpdf.jl")
    include("test_hyperparams_macro.jl")
end
