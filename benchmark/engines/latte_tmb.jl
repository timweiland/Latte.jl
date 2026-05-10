# Latte TMB engine.
#
# Wraps `tmb(lgm, y)` to produce a Result. The TMB pipeline is:
# (1) MAP for hyperparameters, (2) Gaussian posterior for θ via Hessian
# at the MAP, (3) inner Laplace at the MAP for the latent field. Cheap
# and fast, but the hyperparameter posterior is forced to be Gaussian.
#
# DPPL-built LGMs currently require `FiniteDiffStrategy()` for the
# Hessian (Dual-degradation in the latent-prior closure) — see
# tasks/dppl-adapter-outer-ad-closure.org. This engine passes that
# through unconditionally.

using Latte
using Statistics: mean

const ENGINE_ID = "latte_tmb"

engine_id() = ENGINE_ID
engine_version() = string(Pkg.dependencies()[Pkg.project().dependencies["Latte"]].version)

function fit_once(scenario_module, data)
    t_build = @elapsed begin
        dppl_model = scenario_module.build_model(data)
        lgm = latte_from_dppl(dppl_model; random = scenario_module.RANDOM_SYMS)
    end
    t_fit = @elapsed result = tmb(lgm, data.y; diff_strategy = FiniteDiffStrategy())
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
            data_id, env, run_id, timestamp_iso, e
        )
    end

    cold_total = _phases.build + _phases.fit
    cold_phases = PhaseTimings(
        model_construction = _phases.build,
        optimisation = _phases.fit,
        total = cold_total,
    )

    warm_durations = Float64[]
    for _ in 1:repetitions
        t = @elapsed begin
            _, _, _ = fit_once(scenario_module, data)
        end
        push!(warm_durations, t)
    end

    warm_summary = summarize(warm_durations)
    diagnostics = EngineDiagnostics(
        :converged => Latte.converged(_result),
        :elapsed_total => Latte.time_elapsed(_result),
        :warm_median_seconds => warm_summary.median,
        :warm_iqr_lo => warm_summary.iqr_lo,
        :warm_iqr_hi => warm_summary.iqr_hi,
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

# ─── helpers ───────────────────────────────────────────────────────────

function _safe_engine_version()
    try
        return engine_version()
    catch
        return "dev"
    end
end

function _failed_result(
        scenario, scenario_module, mode, repetitions, seed,
        data_id, env, run_id, timestamp_iso, err
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

function _accuracy_vs_reference(scenario_module, scenario, result, data_id)
    has_reference(scenario.id, data_id) || return nothing
    ref = load_reference(scenario.id, data_id)
    ref === nothing && return nothing
    try
        user_latents = user_named_latents(result, scenario_module.RANDOM_SYMS)
        return accuracy_against_reference(
            result.hyperparameter_marginals, user_latents, ref,
        )
    catch
        return nothing
    end
end
