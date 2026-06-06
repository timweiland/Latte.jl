# Offline SBC matrix for the validation report.
#
#   julia --project=benchmark benchmark/sbc/sbc_matrix.jl
#
# Runs Simulation-Based Calibration across the cross product
#   {Poisson, Bernoulli, Normal} × {IID, RW1} × {inla, tmb, hmc_laplace}
# and, per cell, ranks BOTH the free hyperparameters (Hyperparameters target)
# AND the joint observation log-likelihood (DataDependentQuantity) — the latter
# catches joint miscalibration the marginal ranks miss. Same base_seed for both
# target passes ⇒ identical prior draws ⇒ directly comparable ranks.
#
# This is an OFFLINE job: each replicate runs full inference, so a calibration
# claim needs n_attempted ≥ 1000 (env SBC_N). The default is a small smoke size
# that only exercises the pipeline — it is NOT a calibration claim, and the
# stored record says so.
#
# Knobs (all env vars):
#   SBC_N        n_attempted per cell           (default 64; use ≥1000 for claims)
#   SBC_NPOST    posterior draws per replicate  (default 256)
#   SBC_ENGINES  comma list of engines          (default inla,tmb,hmc_laplace)
#   SBC_CELLS    comma list of cell ids         (default all 6)
#   SBC_NODES    latent nodes per replicate     (default 30; e.g. 200 to identify)
#   SBC_PCU      PC-prior scale P(SD>pc_u)=α    (default 1.0; e.g. 3.0 stronger signal)
#   SBC_THREADS  ThreadedExecutor workers       (default 1; run julia -t N to match)
#   SBC_OUT      output JSON path               (default benchmark/results/sbc/sbc_matrix.json)
#
# DEFERRED — Binomial: SBC needs FORWARD sampling (prior_simulate draws y, and
# DataDependentQuantity re-evaluates the likelihood), but a Binomial conditional
# requires per-site trial counts `n` (GMRFs `_conditional_distribution_family`)
# that the LGM's `ExponentialFamily{Binomial}` descriptor does not carry — the
# trials live in the @latte closure, not in a forward-samplable slot of the LGM.
# Bernoulli (the 1-trial case) covers the binary likelihood here; multi-trial
# Binomial SBC is tracked as future work (tasks/feature-requests/).
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))      # the benchmark environment
using Latte, Distributions, DynamicPPL
using GaussianMarkovRandomFields: IIDModel, RW1Model
using StableRNGs: StableRNG
using Statistics, Printf
using JSON3

# ── Cells. Each `@latte` factory returns a LatentGaussianModel (the SBC LGM
# path, which records the latent truth that DataDependentQuantity ranks). ──

# `pc_u` is the PC-prior scale (P(latent SD > pc_u) = α): small pc_u ⇒ heavy-tailed
# prior favouring near-zero latent SD (the weak-identification stress regime);
# larger pc_u ⇒ stronger latent signal. Node count `n` sets how much data informs
# the shared variance component. Together they set the identification regime.

@latte function poisson_iid(y, n, pc_u)
    τ ~ PCPrior.Precision(pc_u, α = 0.01)
    x ~ IIDModel(n)(τ = τ)
    for i in eachindex(y)
        y[i] ~ Poisson(exp(x[i]); check_args = false)
    end
end
@latte function poisson_rw1(y, n, pc_u)
    τ ~ PCPrior.Precision(pc_u, α = 0.01)
    x ~ RW1Model(n)(τ = τ)
    for i in eachindex(y)
        y[i] ~ Poisson(exp(x[i]); check_args = false)
    end
end
@latte function bernoulli_iid(y, n, pc_u)
    τ ~ PCPrior.Precision(pc_u, α = 0.01)
    x ~ IIDModel(n)(τ = τ)
    for i in eachindex(y)
        p_i = 1 / (1 + exp(-x[i]))
        y[i] ~ Bernoulli(p_i; check_args = false)
    end
end
@latte function bernoulli_rw1(y, n, pc_u)
    τ ~ PCPrior.Precision(pc_u, α = 0.01)
    x ~ RW1Model(n)(τ = τ)
    for i in eachindex(y)
        p_i = 1 / (1 + exp(-x[i]))
        y[i] ~ Bernoulli(p_i; check_args = false)
    end
end
@latte function normal_iid(y, n, pc_u)
    τ ~ PCPrior.Precision(pc_u, α = 0.01)
    σ ~ Exponential(0.5)
    x ~ IIDModel(n)(τ = τ)
    for i in eachindex(y)
        y[i] ~ Normal(x[i], σ)
    end
end
@latte function normal_rw1(y, n, pc_u)
    τ ~ PCPrior.Precision(pc_u, α = 0.01)
    σ ~ Exponential(0.5)
    x ~ RW1Model(n)(τ = τ)
    for i in eachindex(y)
        y[i] ~ Normal(x[i], σ)
    end
end

# Cell registry parameterized by the identification regime (n nodes, pc_u scale).
function _cells(; n_nodes::Int, pc_u::Float64)
    return [
        (id = "poisson_iid", build = y -> poisson_iid(y, n_nodes, pc_u), n = n_nodes),
        (id = "poisson_rw1", build = y -> poisson_rw1(y, n_nodes, pc_u), n = n_nodes),
        (id = "bernoulli_iid", build = y -> bernoulli_iid(y, n_nodes, pc_u), n = n_nodes),
        (id = "bernoulli_rw1", build = y -> bernoulli_rw1(y, n_nodes, pc_u), n = n_nodes),
        (id = "normal_iid", build = y -> normal_iid(y, n_nodes, pc_u), n = n_nodes),
        (id = "normal_rw1", build = y -> normal_rw1(y, n_nodes, pc_u), n = n_nodes),
    ]
end

# ── SBC summary statistics ────────────────────────────────────────────

# One-sample KS distance of the rank quantile positions to Uniform(0,1).
# Under perfect calibration this → 0; the 95% null band is ≈ 1.36/√n_success.
function _ks_uniform(ranks::AbstractVector{<:Integer}, L::Int)
    n = length(ranks)
    n == 0 && return NaN
    q = sort((ranks .+ 0.5) ./ (L + 1))
    d = 0.0
    for i in 1:n
        d = max(d, abs(i / n - q[i]), abs(q[i] - (i - 1) / n))
    end
    return d
end

# JSON has no NaN/Inf; an all-failed cell yields non-finite stats. Emit null.
_safe(x) = (x isa Real && !isfinite(x)) ? nothing : x
_r4(x) = _safe(x isa Real && isfinite(x) ? round(x, digits = 4) : x)

# Per-target record from one SBCResult, indexed by descriptor column.
function _target_records(r)
    L = r.n_posterior
    recs = Any[]
    for (j, d) in enumerate(r.targets)
        q = sbc_quantile_position(r, j)
        cov = sbc_coverage(r, j)
        push!(
            recs, (
                target = String(d.label),
                mean_q = _r4(mean(q)),
                var_q = _r4(var(q)),
                ks_uniform = _r4(_ks_uniform(view(r.ranks, :, j), L)),
                cov50 = _r4(cov.cov_0_5),
                cov80 = _r4(cov.cov_0_8),
                cov95 = _r4(cov.cov_0_95),
            )
        )
    end
    return recs
end

# Run both target passes for one (cell, engine) at a shared seed.
function run_cell_engine(cell, engine; n_attempted, n_posterior, base_seed, executor)
    y_proto = Vector{Missing}(missing, cell.n)
    common = (;
        n_attempted = n_attempted, n_posterior = n_posterior,
        engine = engine, base_seed = base_seed, progress = false, executor = executor,
    )
    r_hp = sbc_run(cell.build, y_proto; targets = Hyperparameters(), common...)
    r_dd = sbc_run(cell.build, y_proto; targets = DataDependentQuantity(), common...)
    return (;
        cell = cell.id, engine = String(engine),
        n_attempted = n_attempted, n_posterior = n_posterior,
        n_success = r_hp.n_success, n_failures = r_hp.n_failures,
        status = String(r_hp.status),
        ks_null_band_95 = round(1.36 / sqrt(max(r_hp.n_success, 1)), digits = 4),
        targets = vcat(_target_records(r_hp), _target_records(r_dd)),
    )
end

function run_sbc_matrix(; n_attempted, n_posterior, engines, cell_ids, executor, n_nodes, pc_u, base_seed = UInt64(0x5bc3a710))
    allcells = _cells(; n_nodes = n_nodes, pc_u = pc_u)
    cells = filter(c -> c.id in cell_ids, allcells)
    isempty(cells) && error("No cells matched $(cell_ids); available: $([c.id for c in allcells])")
    records = Any[]
    for cell in cells
        for engine in engines
            t0 = time()
            rec = run_cell_engine(cell, engine; n_attempted, n_posterior, base_seed, executor)
            elapsed = round(time() - t0, digits = 1)
            @printf(
                "%-14s %-12s  n=%4d  ok=%4d fail=%3d  %-22s  %.1fs\n",
                rec.cell, rec.engine, n_attempted, rec.n_success, rec.n_failures,
                rec.status, elapsed,
            )
            push!(records, rec)
        end
    end
    return records
end

function main()
    n_attempted = parse(Int, get(ENV, "SBC_N", "64"))
    n_posterior = parse(Int, get(ENV, "SBC_NPOST", "256"))
    engines = Symbol.(Base.split(get(ENV, "SBC_ENGINES", "inla,tmb,hmc_laplace"), ","))
    n_nodes = parse(Int, get(ENV, "SBC_NODES", "30"))
    pc_u = parse(Float64, get(ENV, "SBC_PCU", "1.0"))
    all_ids = [c.id for c in _cells(; n_nodes = n_nodes, pc_u = pc_u)]
    cell_ids = String.(Base.split(get(ENV, "SBC_CELLS", join(all_ids, ",")), ","))
    out = get(ENV, "SBC_OUT", joinpath(@__DIR__, "..", "results", "sbc", "sbc_matrix.json"))
    nthreads = parse(Int, get(ENV, "SBC_THREADS", "1"))
    executor = nthreads > 1 ? ThreadedExecutor(nworkers = nthreads) : SequentialExecutor()

    is_smoke = n_attempted < 1000
    regime = (n_nodes >= 200 && pc_u >= 3.0) ? "well-identified" :
        (n_nodes <= 50 && pc_u <= 1.0) ? "stress (weak-id)" : "custom"
    @info "SBC matrix" n_attempted n_posterior engines cells = cell_ids threads = nthreads n_nodes pc_u regime smoke = is_smoke
    is_smoke && @warn "n_attempted < 1000 — this is a SMOKE run, NOT a calibration claim. Use SBC_N≥1000 for claims."

    records = run_sbc_matrix(; n_attempted, n_posterior, engines, cell_ids, executor, n_nodes, pc_u)

    payload = (;
        kind = "sbc_matrix",
        n_attempted, n_posterior, n_nodes, pc_u, regime,
        engines = String.(engines),
        is_calibration_claim = !is_smoke,
        note = is_smoke ? "SMOKE run (n_attempted<1000): pipeline check only, not a calibration claim." :
            "Calibration run (n_attempted≥1000).",
        records,
    )
    mkpath(dirname(out))
    open(out, "w") do io
        JSON3.pretty(io, payload)
    end
    @info "wrote SBC matrix results" out n_records = length(records)
    return payload
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
