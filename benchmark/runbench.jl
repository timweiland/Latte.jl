# CLI entry point for the Latte benchmark suite.
#
# Usage:
#
#   julia --project=benchmark benchmark/runbench.jl --suite quick
#   julia --project=benchmark benchmark/runbench.jl --scenario toy_iid_poisson --engine latte_inla
#   julia --project=benchmark benchmark/runbench.jl --suite full --reps 10
#
# The runner is deliberately small: scenarios + engines are plain files
# the runner `include`s into anonymous modules. This keeps coupling low —
# adding a new scenario or engine is "drop a file in, list it in the
# manifest below."

using Pkg
Pkg.activate(@__DIR__)

using Dates
using UUIDs: uuid4
using Logging

# ─── Load shared infrastructure ───────────────────────────────────────

include(joinpath(@__DIR__, "utils", "reporting.jl"))
include(joinpath(@__DIR__, "utils", "environment.jl"))
include(joinpath(@__DIR__, "utils", "timing.jl"))
include(joinpath(@__DIR__, "utils", "reference_store.jl"))
include(joinpath(@__DIR__, "utils", "accuracy.jl"))

# ─── Manifest ─────────────────────────────────────────────────────────
#
# Adding a scenario or engine: drop the file in, add an entry here.
# The runner does not auto-discover — explicit lists keep the suite
# definition diff-visible.

const SCENARIO_FILES = Dict{Symbol, String}(
    :toy_iid_poisson => joinpath(@__DIR__, "scenarios", "toy_iid_poisson.jl"),
    :tokyo_rainfall => joinpath(@__DIR__, "scenarios", "tokyo_rainfall.jl"),
    :seeds => joinpath(@__DIR__, "scenarios", "seeds.jl"),
    :scotland => joinpath(@__DIR__, "scenarios", "scotland.jl"),
    :nhtemp => joinpath(@__DIR__, "scenarios", "nhtemp.jl"),
    :epil => joinpath(@__DIR__, "scenarios", "epil.jl"),
)

const ENGINE_FILES = Dict{Symbol, String}(
    :latte_inla => joinpath(@__DIR__, "engines", "latte_inla.jl"),
    :latte_inla_gaussian => joinpath(@__DIR__, "engines", "latte_inla_gaussian.jl"),
    :latte_inla_simplified => joinpath(@__DIR__, "engines", "latte_inla_simplified.jl"),
    :latte_inla_full => joinpath(@__DIR__, "engines", "latte_inla_full.jl"),
    :latte_tmb => joinpath(@__DIR__, "engines", "latte_tmb.jl"),
    :latte_hmc_laplace => joinpath(@__DIR__, "engines", "latte_hmc_laplace.jl"),
    :nuts_reference => joinpath(@__DIR__, "engines", "nuts_reference.jl"),
    :quadrature_oracle => joinpath(@__DIR__, "engines", "quadrature_oracle.jl"),
)

# Suites are convenience groupings.
#   :quick     — fast smoke run across cheap engines (no NUTS).
#   :full      — same engines, bigger data + more reps.
#   :reference — only the NUTS reference engine; produces / refreshes
#                ReferenceSummary on disk for cross-engine accuracy.
const SUITES = Dict{Symbol, NamedTuple}(
    :quick => (
        mode = :quick, repetitions = 3,
        scenarios = [:toy_iid_poisson],
        engines = [:latte_inla, :latte_tmb, :latte_hmc_laplace],
    ),
    :full => (
        mode = :full, repetitions = 5,
        scenarios = [:toy_iid_poisson],
        engines = [:latte_inla, :latte_tmb, :latte_hmc_laplace],
    ),
    :reference => (
        mode = :full, repetitions = 1,
        scenarios = [:toy_iid_poisson],
        engines = [:nuts_reference],
    ),
    :oracle => (
        mode = :full, repetitions = 1,
        scenarios = [:toy_iid_poisson],
        engines = [:quadrature_oracle],
    ),
    :inla_strategy_sweep => (
        mode = :full, repetitions = 1,
        scenarios = [:toy_iid_poisson],
        engines = [
            :latte_inla_gaussian, :latte_inla_simplified,
            :latte_inla, :latte_inla_full,
        ],
    ),
)

# ─── Module loading ───────────────────────────────────────────────────

# Each scenario file defines top-level `scenario()`, `generate_data`,
# `build_model`, and `RANDOM_SYMS`. We `include` into a fresh anonymous
# module so two scenarios can't accidentally share globals.
function _load_scenario(id::Symbol)
    path = get(SCENARIO_FILES, id, nothing)
    path === nothing && error("unknown scenario $(id) (not in SCENARIO_FILES)")
    isfile(path) || error("scenario file missing: $(path)")
    mod = Module(Symbol("Scenario_", id))
    Core.eval(mod, :(using Latte))
    Core.eval(mod, :(import Main: Scenario))
    Base.include(mod, path)
    return mod
end

# Engine files reference shared types (Result, Scenario, PhaseTimings,
# EngineDiagnostics, summarize, has_reference, …). We bring those into
# the engine's anonymous module via `using ..Main` semantics — easiest
# is to evaluate the include in Main so engine code sees everything.
function _load_engine(id::Symbol)
    path = get(ENGINE_FILES, id, nothing)
    path === nothing && error("unknown engine $(id) (not in ENGINE_FILES)")
    isfile(path) || error("engine file missing: $(path)")
    mod = Module(Symbol("Engine_", id))
    # Re-export everything the engine needs from this runner's scope.
    Core.eval(mod, :(using Pkg))
    Core.eval(
        mod, :(
            import Main: Result, PhaseTimings, AccuracyMetrics,
                EngineDiagnostics, Scenario, ReferenceSummary,
                write_result, summarize, has_reference, load_reference,
                save_reference, reference_path, accuracy_against_reference,
                user_named_latents, capture_environment, git_sha
        )
    )
    Base.include(mod, path)
    return mod
end

# ─── Argument parsing ─────────────────────────────────────────────────

# Tiny hand-rolled parser. ArgParse.jl would be overkill here and adds
# precompile cost.
function _parse_args(args::Vector{String})
    opts = Dict{Symbol, Any}(
        :suite => :quick,
        :scenario => nothing,
        :engine => nothing,
        :reps => nothing,
        :seed => UInt64(0x0badcafe),
        :results_dir => nothing,
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--suite"
            opts[:suite] = Symbol(args[i + 1]); i += 2
        elseif a == "--scenario"
            opts[:scenario] = Symbol(args[i + 1]); i += 2
        elseif a == "--engine"
            opts[:engine] = Symbol(args[i + 1]); i += 2
        elseif a == "--reps"
            opts[:reps] = parse(Int, args[i + 1]); i += 2
        elseif a == "--seed"
            opts[:seed] = parse(UInt64, args[i + 1]); i += 2
        elseif a == "--results-dir"
            opts[:results_dir] = args[i + 1]; i += 2
        elseif a in ("-h", "--help")
            _print_help(); exit(0)
        else
            error("unknown argument: $a (try --help)")
        end
    end
    return opts
end

function _print_help()
    return println(
        """
        Latte benchmark runner.

        Options:
          --suite NAME           One of: $(join(keys(SUITES), ", ")). Default: quick.
          --scenario ID          Single scenario (overrides --suite scenarios).
          --engine ID            Single engine (overrides --suite engines).
          --reps N               Override warm-mode repetitions.
          --seed HEX             Override the PRNG seed (UInt64).
          --results-dir PATH     Override the results output directory.
          -h, --help             Show this message.

        Available scenarios: $(join(keys(SCENARIO_FILES), ", "))
        Available engines:   $(join(keys(ENGINE_FILES), ", "))
        """
    )
end

# ─── Result path layout ───────────────────────────────────────────────

# results/<date>-<host>/<scenario>/<engine>.json
function _result_path(root::String, host::String, scenario_id::String, engine_id::String)
    date = Dates.format(now(), "yyyy-mm-dd")
    return joinpath(root, "$(date)-$(host)", scenario_id, "$(engine_id).json")
end

# ─── Main ─────────────────────────────────────────────────────────────

function main(args::Vector{String} = ARGS)
    opts = _parse_args(args)

    # Resolve which (scenario, engine) pairs to run.
    suite = get(SUITES, opts[:suite], nothing)
    suite === nothing && error("unknown suite: $(opts[:suite])")
    scenarios = opts[:scenario] === nothing ? suite.scenarios : [opts[:scenario]]
    engines = opts[:engine] === nothing ? suite.engines : [opts[:engine]]
    repetitions = opts[:reps] === nothing ? suite.repetitions : opts[:reps]
    mode = suite.mode

    # Snapshot the environment once per invocation.
    env = capture_environment()
    host = get(env, :hostname, "unknown")
    run_id = string(uuid4())
    timestamp = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ")
    results_root = something(opts[:results_dir], joinpath(@__DIR__, "results"))

    @info "starting benchmark run" run_id mode scenarios engines reps = repetitions

    n_ok = 0
    n_fail = 0

    for scenario_id in scenarios
        scen_module = _load_scenario(scenario_id)
        scen_obj = Base.invokelatest(scen_module.scenario)

        eligible = engines  # we trust the user not to ask for incompatible pairs
        for engine_id in eligible
            if !(engine_id in scen_obj.engines)
                @warn "scenario does not list engine; running anyway" scenario = scenario_id engine = engine_id
            end

            eng_module = _load_engine(engine_id)
            @info "running" scenario = scenario_id engine = engine_id

            local result
            try
                result = Base.invokelatest(
                    eng_module.run!,
                    scen_module, scen_obj, mode;
                    repetitions = repetitions,
                    seed = opts[:seed],
                    env = env,
                    run_id = run_id,
                    timestamp_iso = timestamp,
                    timeout_seconds = scen_obj.timeout_seconds,
                )
            catch e
                @error "engine threw outside fit_once" exception = (e, catch_backtrace())
                n_fail += 1
                continue
            end

            path = _result_path(results_root, host, string(scenario_id), string(engine_id))
            write_result(result, path)
            if result.status === :success
                n_ok += 1
                @info "ok" path total_s = result.timings.total
            else
                n_fail += 1
                @warn "not ok" path status = result.status err = result.error_message
            end
        end
    end

    @info "done" ok = n_ok fail = n_fail
    return (ok = n_ok, fail = n_fail)
end

# Run only when executed as a script (not when included).
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
