using Latte
using Test
using Aqua

# DynamicPPL (loaded by the Turing-based suites included below) also exports `marginalize`,
# which collides with Latte's in this shared test module. An explicit import pins the name to
# Latte's binding regardless of include order, so the unqualified calls in the marginalization
# tests resolve unambiguously.
using Latte: marginalize

@testset "Latte.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        # piracies=false: we intentionally extend Distributions.cdf/quantile for SkewNormal
        # (Distributions.jl v0.25 lacks these; see StatsFuns.jl#99 for upstream discussion)
        # stale_deps ignore: BenchmarkTools is a benchmark-only dependency, used by
        # the scripts under benchmark/ (which run in the package's own environment),
        # not by src/. (A cleaner home is the LatteBench benchmark/Project.toml.)
        Aqua.test_all(
            Latte; persistent_tasks = false, piracies = false,
            stale_deps = (ignore = [:BenchmarkTools],),
        )
    end

    include("parallel/runtests.jl")
    include("differentiation/runtests.jl")
    include("utils/runtests.jl")
    include("test_query_interface.jl")
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
    include("model_averaging/runtests.jl")
    include("inference/tmb/runtests.jl")
    include("inference/hmc_laplace/runtests.jl")
    include("diagnostics/runtests.jl")
    include("dsl/runtests.jl")
    include("test_latent_marginalization.jl")
    include("end_to_end/runtests.jl")
end
