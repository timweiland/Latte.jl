using IntegratedNestedLaplace
using Test
using Aqua

@testset "IntegratedNestedLaplace.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        # piracies=false: we intentionally extend Distributions.cdf/quantile for SkewNormal
        # (Distributions.jl v0.25 lacks these; see StatsFuns.jl#99 for upstream discussion)
        Aqua.test_all(IntegratedNestedLaplace; persistent_tasks = false, piracies = false)
    end

    include("utils/runtests.jl")
    include("hyperparameters/runtests.jl")
    include("distributions/runtests.jl")
    include("test_inla_model.jl")
    include("hyperparameter_posterior/runtests.jl")
    include("latent_marginalization/runtests.jl")
    include("latent_augmentation/runtests.jl")
    include("prediction/runtests.jl")
    include("observation_marginals/runtests.jl")
    include("posterior_accumulators/runtests.jl")
    include("posterior_sampling/runtests.jl")
    include("linear_combinations/runtests.jl")
    include("test_latent_marginalization.jl")
    include("end_to_end/runtests.jl")
end
