# Shared core for the Latte INLA engine variants.
#
# The wrapping engine file (e.g. `latte_inla.jl`,
# `latte_inla_gaussian.jl`, ...) defines:
#   - `ENGINE_ID :: String`
#   - `_LATENT_METHOD()` — a zero-arg factory returning a fresh
#                         `MarginalApproximation` instance per fit
# and then `include`s this file. Everything below is the same across
# strategies; the only difference is the marginalization method we
# pass to `inla()`.

using Statistics: mean, std

engine_id() = ENGINE_ID
engine_version() = string(Pkg.dependencies()[Pkg.project().dependencies["Latte"]].version)

function fit_once(scenario_module, data)
    t_build = @elapsed begin
        dppl_model = scenario_module.build_model(data)
        lgm = latte_from_dppl(dppl_model; random = scenario_module.RANDOM_SYMS)
    end
    t_fit = @elapsed result = inla(
        lgm, data.y;
        latent_marginalization_method = _LATENT_METHOD(),
        progress = false,
    )
    return (lgm, result, (build = t_build, fit = t_fit))
end

function run!(
        scenario_module, scenario, mode::Symbol; repetitions::Int = 5,
        seed::UInt64 = UInt64(0x0badcafe), env, run_id::String, timestamp_iso::String,
        timeout_seconds::Float64 = scenario.timeout_seconds,
    )
    n = mode === :quick ? scenario.quick_n : scenario.full_n
    data = scenario_module.generate_data(n; seed = seed)
    data_id = string(hash(data.y), base = 16)

    local _lgm, _result, _phases
    try
        _lgm, _result, _phases = fit_once(scenario_module, data)
    catch e
        return _failed_result(
            scenario, scenario_module, mode, repetitions, seed,
            data_id, env, run_id, timestamp_iso, e,
        )
    end

    cold_phases = PhaseTimings(
        model_construction = _phases.build,
        sampling = _phases.fit,
        total = _phases.build + _phases.fit,
    )

    warm_durations = Float64[]
    for _ in 1:repetitions
        t = @elapsed fit_once(scenario_module, data)
        push!(warm_durations, t)
    end

    warm = summarize(warm_durations)
    diagnostics = EngineDiagnostics(
        :latent_method => string(typeof(_LATENT_METHOD())),
        :converged => Latte.converged(_result),
        :elapsed_total => Latte.time_elapsed(_result),
        :n_grid_points => length(_result.exploration.grid_points),
        :warm_median_seconds => warm.median,
        :warm_iqr_lo => warm.iqr_lo,
        :warm_iqr_hi => warm.iqr_hi,
        :warm_repetitions => repetitions,
    )

    accuracy = _accuracy_vs_reference(scenario_module, scenario, _result, data_id)

    return Result(
        scenario_id = scenario.id,
        scenario_version = string(hash((scenario.id, scenario.full_n, scenario.quick_n)), base = 16),
        engine_id = ENGINE_ID,
        engine_version = _safe_engine_version(),
        run_id = run_id,
        seed = seed,
        data_id = data_id,
        mode = mode,
        status = :success,
        timings = cold_phases,
        repetitions = repetitions,
        timeout_seconds = timeout_seconds,
        accuracy = accuracy,
        diagnostics = diagnostics,
        environment = env,
        timestamp_iso = timestamp_iso,
        git_sha = git_sha(),
    )
end

function _safe_engine_version()
    try
        return engine_version()
    catch
        return "dev"
    end
end

function _failed_result(
        scenario, scenario_module, mode, repetitions, seed,
        data_id, env, run_id, timestamp_iso, err,
    )
    return Result(
        scenario_id = scenario.id,
        scenario_version = string(hash((scenario.id, scenario.full_n)), base = 16),
        engine_id = ENGINE_ID,
        engine_version = _safe_engine_version(),
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

function _accuracy_vs_reference(scenario_module, scenario, inla_result, data_id)
    has_reference(scenario.id, data_id) || return nothing
    ref = load_reference(scenario.id, data_id)
    ref === nothing && return nothing
    try
        user_latents = user_named_latents(inla_result, scenario_module.RANDOM_SYMS)
        return accuracy_against_reference(
            inla_result.hyperparameter_marginals, user_latents, ref,
        )
    catch
        return nothing
    end
end
