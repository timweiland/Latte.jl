# NUTS reference engine. Runs Turing.NUTS on the full @model, writes a
# `ReferenceSummary` JSON for downstream accuracy comparisons, and a
# normal `Result` so the cost is recorded. Only invoked when the
# runner is asked for `--suite reference` or `--engine nuts_reference`.

using Turing: Turing, NUTS, sample, MCMCThreads, MCMCSerial
using MCMCChains: Chains, namesingroup
import MCMCChains
using Random: MersenneTwister
using Statistics: mean, std, quantile
using Dates

const ENGINE_ID = "nuts_reference"

engine_id() = ENGINE_ID
engine_version() = string(Pkg.dependencies()[Pkg.project().dependencies["Turing"]].version)

# `:quick` is for smoke tests; `:full` reads per-scenario knobs
# (`Scenario.nuts_full_*`) so heavy-tailed posteriors can be bumped
# without editing this file.
function _nuts_kwargs(mode::Symbol, scenario)
    if mode === :quick
        return (
            n_samples = 500, n_warmup = 250, n_chains = 2,
            target_accept = 0.8,
        )
    else
        return (
            n_samples = scenario.nuts_full_samples,
            n_warmup = scenario.nuts_full_warmup,
            n_chains = scenario.nuts_full_chains,
            target_accept = scenario.nuts_target_accept,
        )
    end
end

"""
    fit_once(scenario_module, data; rng, mode)

One Turing NUTS fit. Returns the chain plus per-phase timings.
"""
function fit_once(scenario_module, scenario, data; rng, mode::Symbol)
    t_build = @elapsed model = scenario_module.build_model(data)
    kw = _nuts_kwargs(mode, scenario)

    # Multi-chain via MCMCThreads if available, else MCMCSerial.
    backend = Threads.nthreads() > 1 ? MCMCThreads() : MCMCSerial()
    t_fit = @elapsed chain = sample(
        rng, model, NUTS(kw.n_warmup, kw.target_accept), backend,
        kw.n_samples, kw.n_chains;
        progress = false, verbose = false,
    )
    return (model, chain, (build = t_build, fit = t_fit), kw)
end

function run!(
        scenario_module, scenario, mode::Symbol; repetitions::Int = 1,
        seed::UInt64 = UInt64(0x0badcafe), env, run_id::String, timestamp_iso::String,
        timeout_seconds::Float64 = scenario.timeout_seconds,
    )
    n = mode === :quick ? scenario.quick_n : scenario.full_n
    data = scenario_module.generate_data(n; seed = seed)
    data_id = string(hash(data.y), base = 16)

    rng = MersenneTwister(seed)

    local _model, _chain, _phases, _kw
    try
        _model, _chain, _phases, _kw = fit_once(
            scenario_module, scenario, data; rng = rng, mode = mode,
        )
    catch e
        return _failed_result(
            scenario, scenario_module, mode, repetitions, seed,
            data_id, env, run_id, timestamp_iso, e
        )
    end

    # ── Build ReferenceSummary from the chain ────────────────────────────
    hp_syms = scenario_module.HP_SYMS
    summary = _chain_to_reference(_chain, scenario, hp_syms, _kw, seed, data_id)

    # Persist as the canonical reference for this (scenario, dataset).
    # Different `(n, seed)` produces a different `data_id`, so coexisting
    # references are fine and never overwrite each other.
    save_reference(summary)

    cold_total = _phases.build + _phases.fit
    cold_phases = PhaseTimings(
        model_construction = _phases.build,
        sampling = _phases.fit,
        total = cold_total,
    )

    # NUTS is too expensive to repeat — `repetitions` is honoured but
    # almost always 1 for reference runs.
    warm_durations = Float64[]
    for r in 1:max(0, repetitions - 1)
        rng_r = MersenneTwister(seed + UInt64(r + 1))
        t = @elapsed begin
            _, _, _, _ = fit_once(
                scenario_module, scenario, data; rng = rng_r, mode = mode,
            )
        end
        push!(warm_durations, t)
    end

    diagnostics = EngineDiagnostics(
        :n_chains => _kw.n_chains,
        :n_samples_per_chain => _kw.n_samples,
        :n_warmup => _kw.n_warmup,
        :min_ess => minimum(summary.ess),
        :max_rhat => maximum(summary.rhat),
        :reference_path => reference_path(scenario.id, data_id),
        :warm_repetitions => length(warm_durations),
    )
    if !isempty(warm_durations)
        ws = summarize(warm_durations)
        diagnostics[:warm_median_seconds] = ws.median
        diagnostics[:warm_iqr_lo] = ws.iqr_lo
        diagnostics[:warm_iqr_hi] = ws.iqr_hi
    end

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
        accuracy = nothing,  # NUTS *is* the reference
        diagnostics = diagnostics,
        environment = env,
        timestamp_iso = timestamp_iso,
        git_sha = git_sha(),
    )
end

# ─── chain → ReferenceSummary ────────────────────────────────────────

# Pull a per-parameter posterior summary out of the chain. Vector
# parameters (e.g. `x[1]`, `x[2]`, …) are flattened in order. We
# summarise hyperparameters first (in `hp_syms` order) so the slice
# `[1:n_hp]` contract used by other engines holds.
function _chain_to_reference(
        chain::Chains, scenario, hp_syms, kw,
        seed::UInt64, data_id::AbstractString,
    )
    # Order: hyperparameters first (in HP_SYMS order, expanded for
    # vector-valued params), then every other model parameter. HMC
    # internal stats are excluded by the `:parameters` section filter.
    hp_names_sym = Symbol[]
    for sym in hp_syms
        for vname in namesingroup(chain, sym)
            push!(hp_names_sym, vname)
        end
    end
    all_param_syms = MCMCChains.names(chain, :parameters)
    other_syms = [n for n in all_param_syms if !(n in hp_names_sym)]
    ordered = vcat(hp_names_sym, other_syms)

    summ = MCMCChains.summarize(chain).nt
    quants = MCMCChains.quantile(chain; q = [0.025, 0.25, 0.5, 0.75, 0.975, 0.99]).nt

    sidx = Dict(summ.parameters[i] => i for i in eachindex(summ.parameters))
    qidx = Dict(quants.parameters[i] => i for i in eachindex(quants.parameters))

    q_col(p_str) = getproperty(quants, Symbol(p_str))
    q025_col = q_col("2.5%")
    q25_col = q_col("25.0%")
    q50_col = q_col("50.0%")
    q75_col = q_col("75.0%")
    q975_col = q_col("97.5%")
    q99_col = q_col("99.0%")

    ess_vals = [summ.ess_bulk[sidx[n]] for n in ordered]
    rhat_vals = [summ.rhat[sidx[n]] for n in ordered]
    q025 = [q025_col[qidx[n]] for n in ordered]
    q25 = [q25_col[qidx[n]] for n in ordered]
    q50 = [q50_col[qidx[n]] for n in ordered]
    q75 = [q75_col[qidx[n]] for n in ordered]
    q975 = [q975_col[qidx[n]] for n in ordered]
    q99 = [q99_col[qidx[n]] for n in ordered]

    # ── Empirical CDFs from pooled chain samples for KS-based
    # accuracy. Sort-and-rank gives a cdf grid (sorted samples) and
    # cdf values ((1:n)/n). Pooled across chains; mixing diagnostics
    # (rhat, ess) tell consumers whether that's defensible.
    cdf_grids, cdf_values = _empirical_cdfs(chain, ordered)

    return ReferenceSummary(
        scenario_id = scenario.id,
        data_id = data_id,
        parameter_names = string.(ordered),
        posterior_cdf_grids = cdf_grids,
        posterior_cdf_values = cdf_values,
        posterior_q025 = q025,
        posterior_q25 = q25,
        posterior_median = q50,
        posterior_q75 = q75,
        posterior_q975 = q975,
        posterior_q99 = q99,
        ess = ess_vals,
        rhat = rhat_vals,
        n_chains = kw.n_chains,
        n_samples_per_chain = kw.n_samples,
        n_warmup = kw.n_warmup,
        seed = seed,
        timestamp_iso = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
        notes = String[
            "Reference produced by Turing.NUTS via benchmark/engines/nuts_reference.jl.",
        ],
    )
end

# Build an empirical CDF for each parameter from pooled chain samples.
# Returns a (cdf_grid_per_param, cdf_values_per_param) pair where each
# grid is the sorted draws and the values are (1:n) / n.
function _empirical_cdfs(chain::Chains, ordered::Vector{Symbol})
    grids = Vector{Vector{Float64}}(undef, length(ordered))
    values = Vector{Vector{Float64}}(undef, length(ordered))
    K = size(chain, 3)
    for (i, name) in enumerate(ordered)
        # Pool all chains and samples into a single sorted vector.
        per_chain = [vec(chain[:, name, k:k]) for k in 1:K]
        all = sort!(reduce(vcat, per_chain))
        n = length(all)
        grids[i] = all
        values[i] = collect(range(1, n) ./ n)
    end
    return grids, values
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
