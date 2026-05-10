# Latte HMC-Laplace engine.
#
# Wraps `hmc_laplace(lgm, y)` — NUTS over the Laplace marginal `p(θ | y)`,
# with the inner Laplace `q(x | θ)` reconstructed at each draw. This is
# tmbstan in spirit: an exact-as-MCMC posterior on θ, paired with a
# Laplace surrogate on x. Slower than INLA/TMB but gives a fair
# "what would NUTS-on-Laplace say?" reading.
#
# DPPL-built LGMs require `FiniteDiffStrategy()` for the TMB warm-start.

using Latte
using Random: MersenneTwister
using Statistics: mean

const ENGINE_ID = "latte_hmc_laplace"

engine_id() = ENGINE_ID
engine_version() = string(Pkg.dependencies()[Pkg.project().dependencies["Latte"]].version)

# n_samples / n_warmup are kept modest in `:quick` mode for a cheap smoke
# fit; `:full` mode bumps them so the chain has enough mixing for
# meaningful diagnostics.
function _hmc_kwargs(mode::Symbol)
    return mode === :quick ?
        (n_samples = 200, n_warmup = 100) :
        (n_samples = 1000, n_warmup = 500)
end

function fit_once(scenario_module, data; rng, mode::Symbol)
    t_build = @elapsed begin
        dppl_model = scenario_module.build_model(data)
        lgm = latte_from_dppl(dppl_model; random = scenario_module.RANDOM_SYMS)
    end
    kwargs = _hmc_kwargs(mode)
    t_fit = @elapsed result = hmc_laplace(
        lgm, data.y;
        rng = rng,
        n_samples = kwargs.n_samples,
        n_warmup = kwargs.n_warmup,
        diff_strategy = FiniteDiffStrategy(),
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

    rng = MersenneTwister(seed)

    local _lgm, _result, _phases
    try
        _lgm, _result, _phases = fit_once(scenario_module, data; rng = rng, mode = mode)
    catch e
        return _failed_result(
            scenario, scenario_module, mode, repetitions, seed,
            data_id, env, run_id, timestamp_iso, e
        )
    end

    cold_total = _phases.build + _phases.fit
    cold_phases = PhaseTimings(
        model_construction = _phases.build,
        sampling = _phases.fit,
        total = cold_total,
    )

    warm_durations = Float64[]
    for r in 1:repetitions
        rng_r = MersenneTwister(seed + UInt64(r))
        t = @elapsed begin
            _, _, _ = fit_once(scenario_module, data; rng = rng_r, mode = mode)
        end
        push!(warm_durations, t)
    end

    warm_summary = summarize(warm_durations)
    diagnostics = EngineDiagnostics(
        :converged => Latte.converged(_result),
        :elapsed_total => Latte.time_elapsed(_result),
        :divergences => Latte.divergences(_result),
        :mean_tree_depth => Latte.mean_tree_depth(_result),
        :acceptance_rate => Latte.acceptance_rate(_result),
        :mean_step_size => Latte.mean_step_size(_result),
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
