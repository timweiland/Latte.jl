# Quadrature-oracle engine. Dispatches on `scenario.id` to a per-scenario
# `oracle_summary` defined in `benchmark/oracles/<scenario_id>.jl`.
# Writes a `ReferenceSummary` JSON in the same place a NUTS reference
# would go.

using Latte
using Dates

const ENGINE_ID = "quadrature_oracle"

engine_id() = ENGINE_ID
engine_version() = "1.0.0"

const _ORACLE_FILES = Dict{String, String}(
    "toy_iid_poisson" => joinpath(@__DIR__, "..", "oracles", "toy_iid_poisson.jl"),
)

# Lazy include of the per-scenario oracle. Tracked so we don't
# re-include on every fit.
const _LOADED_ORACLES = Set{String}()

function _ensure_oracle_loaded(scenario_id::String)
    scenario_id in _LOADED_ORACLES && return
    path = get(_ORACLE_FILES, scenario_id, nothing)
    path === nothing && error(
        "no quadrature oracle registered for scenario $(scenario_id); add one to benchmark/oracles/",
    )
    isfile(path) || error("oracle file missing: $(path)")
    Base.include(@__MODULE__, path)
    push!(_LOADED_ORACLES, scenario_id)
    return
end

function fit_once(scenario_module, scenario, data; kwargs...)
    _ensure_oracle_loaded(scenario.id)
    t_build = @elapsed nothing
    # `invokelatest` because `oracle_summary` is defined by the lazy
    # include — newer world age than this function.
    t_fit = @elapsed primary = Base.invokelatest(oracle_summary, data; kwargs...)
    return (primary, (build = t_build, fit = t_fit))
end

function run!(
        scenario_module, scenario, mode::Symbol; repetitions::Int = 1,
        seed::UInt64 = UInt64(0x0badcafe), env, run_id::String, timestamp_iso::String,
        timeout_seconds::Float64 = scenario.timeout_seconds,
    )
    n = mode === :quick ? scenario.quick_n : scenario.full_n
    data = scenario_module.generate_data(n; seed = seed)
    data_id = string(hash(data.y), base = 16)

    local primary, phases
    try
        primary, phases = fit_once(scenario_module, scenario, data)
    catch e
        return _failed_result(
            scenario, scenario_module, mode, repetitions, seed,
            data_id, env, run_id, timestamp_iso, e,
        )
    end

    n_params = length(primary.parameter_names)
    summary = ReferenceSummary(
        scenario_id = scenario.id,
        data_id = data_id,
        parameter_names = primary.parameter_names,
        posterior_cdf_grids = primary.cdf_grids,
        posterior_cdf_values = primary.cdf_values,
        posterior_q025 = primary.q025,
        posterior_q25 = primary.q25,
        posterior_median = primary.median,
        posterior_q75 = primary.q75,
        posterior_q975 = primary.q975,
        posterior_q99 = primary.q99,
        # NUTS-only diagnostics — placeholders for an oracle reference.
        ess = zeros(n_params),
        rhat = ones(n_params),
        n_chains = 0,
        n_samples_per_chain = 0,
        n_warmup = 0,
        seed = seed,
        timestamp_iso = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
        notes = String[
            "Deterministic quadrature oracle: 64-node Gauss-Hermite for inner conditionals + trapezoidal log-τ grid + per-latent x-grid CDF.",
        ],
    )
    save_reference(summary)

    cold_phases = PhaseTimings(
        model_construction = phases.build,
        sampling = phases.fit,
        total = phases.build + phases.fit,
    )

    diagnostics = EngineDiagnostics(
        :n_grid_τ => primary.n_grid_τ,
        :n_grid_x => primary.n_grid_x,
        :reference_path => reference_path(scenario.id, data_id),
        :n_obs => length(data.y),
    )

    return Result(
        scenario_id = scenario.id,
        scenario_version = string(hash((scenario.id, scenario.full_n, scenario.quick_n)), base = 16),
        engine_id = ENGINE_ID,
        engine_version = engine_version(),
        run_id = run_id,
        seed = seed,
        data_id = data_id,
        mode = mode,
        status = :success,
        timings = cold_phases,
        repetitions = 1,
        timeout_seconds = timeout_seconds,
        accuracy = nothing,
        diagnostics = diagnostics,
        environment = env,
        timestamp_iso = timestamp_iso,
        git_sha = git_sha(),
    )
end

function _failed_result(
        scenario, scenario_module, mode, repetitions, seed,
        data_id, env, run_id, timestamp_iso, err,
    )
    return Result(
        scenario_id = scenario.id,
        scenario_version = string(hash((scenario.id, scenario.full_n)), base = 16),
        engine_id = ENGINE_ID,
        engine_version = engine_version(),
        run_id = run_id,
        seed = seed,
        data_id = data_id,
        mode = mode,
        status = :failed,
        error_type = string(typeof(err)),
        error_message = sprint(showerror, err),
        timings = PhaseTimings(),
        repetitions = repetitions,
        timeout_seconds = scenario.timeout_seconds,
        diagnostics = EngineDiagnostics(),
        environment = env,
        timestamp_iso = timestamp_iso,
        git_sha = git_sha(),
    )
end
