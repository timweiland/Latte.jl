using Random
using StableRNGs: StableRNG

export sbc_run, sbc_coverage, sbc_quantile_position

"""
    sbc_run(build_model, y_prototype;
            n_attempted = 200,
            n_posterior = 1000,
            engine = :inla,
            engine_kwargs = (;),
            targets = Hyperparameters(),
            random,
            base_seed = UInt64(0xbadc0de),
            failure_policy = SBCFailurePolicy(),
            obs_name = :y,
            progress = true) -> SBCResult

Run Simulation-Based Calibration on a DPPL model.

`build_model(y)` is a user-provided zero- (or one-) argument closure
returning a `DynamicPPL.Model`. SBC calls it first with `y_prototype`
(a `Vector{Missing}` sized to the observations) to draw a prior
replicate, and then with the simulated `y` to run inference.

`random` is the tuple of syms Latte should treat as random effects —
the same kwarg you'd pass to `latte_from_dppl(model; random = …)`.

`n_attempted` fixes the number of replicate attempts. `n_success` is
derived and reported separately on the result.

For publishable calibration claims, `n_attempted = 200` is a smoke
test. Aim for 1000–5000.
"""
function sbc_run(
        build_model, y_prototype;
        n_attempted::Int = 200,
        n_posterior::Int = 1000,
        engine::Symbol = :inla,
        engine_kwargs::NamedTuple = (;),
        targets::SBCTarget = Hyperparameters(),
        random,
        base_seed::Unsigned = UInt64(0x0badc0de),
        failure_policy::SBCFailurePolicy = SBCFailurePolicy(),
        obs_name::Symbol = :y,
        executor::ParallelExecutor = SequentialExecutor(),
        progress::Bool = true,
    )
    engine_fn = _engine_fn(engine)

    # Resolve target descriptors up-front against a probe LGM. The
    # probe model uses a concrete y (zeros of the right eltype) so
    # latte_from_dppl succeeds; we only use it to read the
    # hyperparameter ordering.
    probe_y = _probe_y_from_prototype(build_model, y_prototype, base_seed, obs_name)
    probe_lgm = latte_from_dppl(build_model(probe_y); random = random)
    descriptors = resolve_targets(targets, probe_lgm)
    isempty(descriptors) && throw(
        ArgumentError("SBC: no targets resolved from $(typeof(targets)).")
    )

    # Fan out replicates via Latte's executor. Results come back in
    # order (pmap_executor preserves indexing), so determinism is
    # preserved across sequential/threaded runs as long as each
    # replicate uses its own `StableRNG(hash((base_seed, i)))` —
    # which is independent of scheduling.
    t_start = time()
    per_rep = pmap_executor(1:n_attempted, executor) do i
        rng_i = StableRNG(hash((base_seed, UInt(i))))
        return _run_one_replicate(
            build_model, y_prototype, rng_i, i,
            engine_fn, engine_kwargs, random,
            descriptors, n_posterior, obs_name,
        )
    end
    elapsed = time() - t_start

    ranks_rows = Vector{Vector{Int}}()
    truths_rows = Vector{Vector{Float64}}()
    failures = SBCFailure[]
    diagnostics = ReplicateDiagnostics[]
    for result in per_rep
        if result.kind === :success
            push!(ranks_rows, result.ranks)
            push!(truths_rows, result.truths)
            push!(diagnostics, result.diagnostics)
        else
            push!(failures, result.failure)
            if failure_policy.on_failure === :error
                error(
                    "SBC replicate $(result.failure.replicate_id) failed at " *
                        "stage $(result.failure.stage): $(result.failure.message)"
                )
            end
        end
    end

    n_success = length(ranks_rows)
    n_failures = length(failures)
    status = _determine_status(n_attempted, n_failures, failure_policy)

    ranks_mat = _rows_to_matrix(Int, ranks_rows, length(descriptors))
    truths_mat = _rows_to_matrix(Float64, truths_rows, length(descriptors))

    return SBCResult(
        descriptors, ranks_mat, truths_mat,
        n_posterior, n_attempted, n_success, n_failures,
        failures, diagnostics, status, UInt64(base_seed),
        engine, engine_kwargs, elapsed,
    )
end

# ─── Internal helpers ─────────────────────────────────────────────────

# Engine-specific dispatch. Each engine receives the per-replicate RNG
# plus a user kwargs bag. Engines that accept neither `rng` nor
# `progress` (currently `tmb`) get a trimmed call so we don't error on
# unknown kwargs.
function _engine_fn(engine::Symbol)
    engine === :inla && return (lgm, y, rng; kwargs...) ->
    inla(lgm, y; progress = false, kwargs...)
    engine === :tmb && return (lgm, y, rng; kwargs...) ->
    tmb(lgm, y; kwargs...)
    engine === :hmc_laplace && return (lgm, y, rng; kwargs...) ->
    hmc_laplace(lgm, y; rng = rng, progress = false, kwargs...)
    return throw(ArgumentError("SBC: unknown engine `:$(engine)`. Must be one of :inla, :tmb, :hmc_laplace."))
end

# Build a probe `y` of the right shape + eltype so we can stand up a
# probe LGM for target resolution. We just draw one prior replicate and
# use its simulated `y`.
function _probe_y_from_prototype(build_model, y_prototype, base_seed, obs_name)
    rng = StableRNG(hash((base_seed, 0x0070_72_6f_62_65)))  # "probe"
    rep = _prior_simulate(build_model, y_prototype, rng; obs_name = obs_name)
    return rep.y
end

# ── Per-replicate execution ─────────────────────────────────────────
# Returns a NamedTuple with either `kind = :success` and the per-rep
# data, or `kind = :failure` and an SBCFailure.

function _run_one_replicate(
        build_model, y_prototype, rng, replicate_id,
        engine_fn, engine_kwargs, random,
        descriptors, n_posterior, obs_name,
    )
    # Stage 1: prior simulate
    local rep
    try
        rep = _prior_simulate(build_model, y_prototype, rng; replicate_id = replicate_id, obs_name = obs_name)
    catch e
        return (;
            kind = :failure, failure = SBCFailure(
                replicate_id, :prior_simulate,
                string(typeof(e)), sprint(showerror, e), nothing,
            ),
        )
    end
    truth_nt = rep.truth

    # Stage 2: build LGM
    local lgm
    try
        inf_model = build_model(rep.y)
        lgm = latte_from_dppl(inf_model; random = random)
    catch e
        return (;
            kind = :failure, failure = SBCFailure(
                replicate_id, :model_build,
                string(typeof(e)), sprint(showerror, e), truth_nt,
            ),
        )
    end

    # Stage 3: inference
    local inf_result
    t_inf = 0.0
    try
        t0 = time()
        inf_result = engine_fn(lgm, rep.y, rng; engine_kwargs...)
        t_inf = time() - t0
    catch e
        return (;
            kind = :failure, failure = SBCFailure(
                replicate_id, :inference,
                string(typeof(e)), sprint(showerror, e), truth_nt,
            ),
        )
    end

    # Stage 4: posterior sample
    local θ_mat
    try
        samples = rand(rng, inf_result, n_posterior)
        θ_mat = samples.θ
    catch e
        return (;
            kind = :failure, failure = SBCFailure(
                replicate_id, :posterior_sample,
                string(typeof(e)), sprint(showerror, e), truth_nt,
            ),
        )
    end

    # Stage 5: rank
    local ranks_row, truths_row
    try
        ranks_row = Vector{Int}(undef, length(descriptors))
        truths_row = Vector{Float64}(undef, length(descriptors))
        for (j, d) in enumerate(descriptors)
            truth_val = d.extract_truth(truth_nt)
            post = d.extract_posterior(θ_mat)
            ranks_row[j] = _rank(post, truth_val, rng)
            truths_row[j] = truth_val
        end
    catch e
        return (;
            kind = :failure, failure = SBCFailure(
                replicate_id, :rank,
                string(typeof(e)), sprint(showerror, e), truth_nt,
            ),
        )
    end

    diag = ReplicateDiagnostics(
        replicate_id = replicate_id,
        inference_time = t_inf,
        convergence_ok = _convergence(inf_result),
    )

    return (;
        kind = :success,
        ranks = ranks_row,
        truths = truths_row,
        diagnostics = diag,
    )
end

"""Rank of `truth` in `posterior_draws`. Talts et al. style:
`r = #{ l : draws[l] < truth } + tie_break` where ties are broken
uniformly at random to avoid degenerate rank distributions at discrete
ties."""
function _rank(posterior_draws::AbstractVector{<:Real}, truth::Real, rng::AbstractRNG)
    below = 0
    ties = 0
    for v in posterior_draws
        if v < truth
            below += 1
        elseif v == truth
            ties += 1
        end
    end
    return ties == 0 ? below : below + rand(rng, 0:ties)
end

function _convergence(r::InferenceResult)
    try
        return converged(r)
    catch
        return nothing
    end
end

function _determine_status(n_attempted, n_failures, policy)
    n_failures == 0 && return :valid
    rate = n_failures / n_attempted
    return rate > policy.max_failure_rate ? :invalid : :completed_with_failures
end

function _rows_to_matrix(::Type{T}, rows::Vector{Vector{T}}, ncol::Int) where {T}
    nrow = length(rows)
    mat = Matrix{T}(undef, nrow, ncol)
    for (i, row) in enumerate(rows)
        mat[i, :] = row
    end
    return mat
end

# ─── Summary helpers ───────────────────────────────────────────────────

"""
    sbc_coverage(r::SBCResult, target_index::Int; levels = (0.5, 0.8, 0.95))

Empirical coverage for the given target (column of `r.ranks` /
`r.truths`) at each credible-interval level. Returns a NamedTuple
keyed by level. Computed purely from ranks (no re-draw of posterior
samples), so it's cheap and deterministic.

At level `α`, the posterior CI of width `α` covers `θ_true` exactly
when the rank falls in the central `α·(n_posterior + 1)` of the
rank range. Estimator: `count(central_band) / n_success`.
"""
function sbc_coverage(r::SBCResult, target_index::Int; levels = (0.5, 0.8, 0.95))
    1 <= target_index <= size(r.ranks, 2) ||
        throw(ArgumentError("target_index $(target_index) out of range (1:$(size(r.ranks, 2)))"))
    L = r.n_posterior
    ranks_j = view(r.ranks, :, target_index)
    n = length(ranks_j)
    out = Pair{Symbol, Float64}[]
    for α in levels
        half = α / 2
        lo = (0.5 - half) * (L + 1)
        hi = (0.5 + half) * (L + 1)
        cnt = count(r -> lo <= r <= hi, ranks_j)
        push!(out, Symbol("cov_", replace(string(α), "." => "_")) => cnt / n)
    end
    return NamedTuple(out)
end

"""
    sbc_quantile_position(r::SBCResult, target_index::Int) -> Vector{Float64}

Per-replicate quantile position `q = (rank + 0.5) / (n_posterior + 1)`
for the given target. A calibrated procedure has `q` uniform on
`(0, 1)`; the mean should be close to 0.5 and the variance close to
1/12.
"""
function sbc_quantile_position(r::SBCResult, target_index::Int)
    1 <= target_index <= size(r.ranks, 2) ||
        throw(ArgumentError("target_index $(target_index) out of range (1:$(size(r.ranks, 2)))"))
    L = r.n_posterior
    return [(rk + 0.5) / (L + 1) for rk in view(r.ranks, :, target_index)]
end

# ─── Pretty printing ───────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", r::SBCResult)
    status_color = r.status === :valid ? :green :
        r.status === :completed_with_failures ? :yellow : :red
    println(io, "SBCResult:")
    println(io, "  Engine:      :$(r.engine)")
    println(io, "  Replicates:  $(r.n_success) / $(r.n_attempted) successful")
    print(io, "  Status:      ")
    printstyled(io, r.status; color = status_color, bold = true)
    println(io)
    if r.n_failures > 0
        println(io, "  Failures:    $(r.n_failures) total")
        stage_counts = _count_by_stage(r.failures)
        for (stage, n) in stage_counts
            println(io, "    $(stage): $(n)")
        end
    end
    println(io, "  Targets:     $([d.label for d in r.targets])")
    println(io, "  Elapsed:     $(round(r.elapsed, digits = 1)) s")
    if r.n_success > 0
        println(io, "  Per-target summaries:")
        for (j, d) in enumerate(r.targets)
            q = sbc_quantile_position(r, j)
            cov = sbc_coverage(r, j)
            @printf(
                io,
                "    %-10s  mean q = %.3f    coverage 50/80/95 = %.2f / %.2f / %.2f\n",
                String(d.label), Statistics.mean(q),
                cov.cov_0_5, cov.cov_0_8, cov.cov_0_95,
            )
        end
    end
    println(io, "  Base seed:   0x$(string(r.base_seed; base = 16))")
    if r.n_attempted < 1000
        printstyled(
            io,
            "  ⚠  n_attempted=$(r.n_attempted) is a smoke-test size. Aim for ≥ 1000 for calibration claims.\n";
            color = :yellow,
        )
    end
    return nothing
end

function _count_by_stage(failures::Vector{SBCFailure})
    counts = Dict{Symbol, Int}()
    for f in failures
        counts[f.stage] = get(counts, f.stage, 0) + 1
    end
    return sort(collect(counts), by = first)
end
