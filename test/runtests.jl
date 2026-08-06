using Latte
using Test
using Aqua

# DynamicPPL (loaded by the Turing-based suites included below) also exports `marginalize`,
# which collides with Latte's in this shared test module. An explicit import pins the name to
# Latte's binding regardless of include order, so the unqualified calls in the marginalization
# tests resolve unambiguously.
using Latte: marginalize

# CI shards the suite across parallel jobs via LATTE_TEST_GROUP (see .github/workflows/CI.yml).
# Groups are balanced by measured runtime, not by theme; without the env var everything runs.
# The dsl suite is split at file granularity (it dominates the runtime), so its files are listed
# here individually; `dsl/runtests.jl` remains the standalone entry point for local dsl-only runs.
const TEST_GROUP = get(ENV, "LATTE_TEST_GROUP", "all")
ingroup(g::String) = TEST_GROUP == "all" || TEST_GROUP == g

# Keep dsl files inside dsl shards: they amortize the DSL/non-Gaussian pipeline compile
# across the group's process, so a dsl file moved into core costs far more there than its
# in-shard marginal time.
const DSL_GROUPED_FILES = (
    dsl1 = [
        "test_adapter.jl", "test_end_to_end.jl", "test_fast_path_detection.jl",
        "test_constraints.jl", "test_fixed_gmrf_model.jl", "test_latent_layout.jl",
        "test_named_marginals.jl", "test_prelude_lift.jl", "test_prelude_lift_codegen.jl",
        "test_prelude_lift_end_to_end.jl", "test_dag_assembly_plan.jl",
        "test_dag_sparse_ad_plan.jl", "test_dotted_prior.jl", "test_landing_example.jl",
    ],
    dsl2 = [
        "test_fast_path_agrees.jl", "test_nlsq_recognition.jl", "test_nlsq_composite.jl",
        "test_obs_groups.jl", "test_latte_macro.jl", "test_turing_handoff.jl",
    ],
    dsl3 = [
        "test_recognition.jl", "test_matrix_latents.jl", "test_factor_extraction.jl",
        "test_structured_guard.jl", "test_structured_macro.jl", "test_block_latent.jl",
        "test_vector_hyperparameters.jl",
    ],
)

@testset "Latte.jl" begin
    # Guard against drift: every dsl file included by the standalone dsl/runtests.jl must appear
    # in exactly one shard group, so adding a dsl test file cannot silently drop it from CI.
    @testset "shard groups cover dsl/runtests.jl" begin
        standalone = [
            m[1] for m in
                eachmatch(r"include\(\"([^\"]+)\"\)", read(joinpath(@__DIR__, "dsl", "runtests.jl"), String))
        ]
        grouped = reduce(vcat, collect(DSL_GROUPED_FILES))
        @test sort(standalone) == sort(grouped)
    end

    if ingroup("core")
        @testset "Code quality (Aqua.jl)" begin
            # piracies=false: we intentionally extend Distributions.cdf/quantile for SkewNormal
            # (Distributions.jl v0.25 lacks these; see StatsFuns.jl#99 for upstream discussion)
            Aqua.test_all(
                Latte; persistent_tasks = false, piracies = false,
                undocumented_names = true,
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
    end

    if ingroup("dsl1")
        include("inference/tmb/runtests.jl")
        include("inference/hmc_laplace/runtests.jl")
        include("diagnostics/runtests.jl")
        for f in DSL_GROUPED_FILES.dsl1
            include(joinpath("dsl", f))
        end
        include("test_latent_marginalization.jl")
        include("end_to_end/runtests.jl")
    end

    if ingroup("dsl2")
        for f in DSL_GROUPED_FILES.dsl2
            include(joinpath("dsl", f))
        end
    end

    if ingroup("dsl3")
        for f in DSL_GROUPED_FILES.dsl3
            include(joinpath("dsl", f))
        end
    end
end
