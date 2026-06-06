# Aggregate committed SBC calibration results
#   benchmark/results/sbc/*.json
# into the data file consumed by the validation docs page:
#   docs/src/data/validation_results.json
#
# Usage:
#   julia --project=benchmark benchmark/render_validation.jl
#
# Output is organized for ENGINE TABS: per engine, per identification regime, a
# table of (cell × quantity) PIT-KS values with a PASS / ≈band / FAIL verdict vs
# the 95% null band, plus a roll-up. Records from different result files are
# merged by (engine, regime) — e.g. the n=1000 INLA/TMB stress file and the
# leaner hmc_laplace stress file both contribute to the "stress" regime. This
# script does NO inference and runs NO MCMC; it only reads the committed JSONs.

using Pkg
Pkg.activate(@__DIR__)

using JSON3
using Printf
using Dates

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
const SBC_DIR = joinpath(REPO_ROOT, "benchmark", "results", "sbc")
const DATA_OUT = joinpath(REPO_ROOT, "docs", "src", "data", "validation_results.json")

const ENGINE_ORDER = ["inla", "tmb", "hmc_laplace"]
const REGIME_ORDER = ["well-identified", "stress (weak-id)", "custom"]

_band(n_attempted) = 1.36 / sqrt(n_attempted)

# Verdict vs the 95% KS null band (small slack absorbs multiple testing).
function _verdict(ks, band)
    ks === nothing && return "n/a"
    ks <= band * 1.1 && return "pass"
    ks <= band * 1.6 && return "border"
    return "fail"
end

# Identification regime, keyed on how much data informs the variance component.
function _regime(n_nodes)
    n_nodes >= 100 && return "well-identified"
    n_nodes <= 50 && return "stress (weak-id)"
    return "custom"
end

_order(x, order) = let i = findfirst(==(x), order)
    (i === nothing ? typemax(Int) : i, x)
end

function main()
    isdir(SBC_DIR) || error("No SBC results dir at $(relpath(SBC_DIR, REPO_ROOT)); run benchmark/sbc/sbc_matrix.jl first.")
    files = sort(filter(f -> endswith(f, ".json"), readdir(SBC_DIR; join = true)))

    # Flatten every (file → record → target) into one tidy row list.
    flat = NamedTuple[]
    for f in files
        r = JSON3.read(read(f, String))
        get(r, :kind, "") == "sbc_matrix" || continue
        n_attempted = Int(r.n_attempted)
        n_nodes = Int(get(r, :n_nodes, 30))
        pc_u = Float64(get(r, :pc_u, 1.0))
        regime = _regime(n_nodes)
        band = round(_band(n_attempted); digits = 4)
        for rec in r.records
            eng = String(rec.engine)
            for t in rec.targets
                ks = t.ks_uniform === nothing ? nothing : Float64(t.ks_uniform)
                push!(
                    flat, (;
                        engine = eng, regime = regime, n_attempted = n_attempted,
                        n_nodes = n_nodes, pc_u = pc_u, band = band,
                        cell = String(rec.cell), target = String(t.target),
                        ks = ks, verdict = _verdict(ks, band),
                    )
                )
            end
        end
    end
    isempty(flat) && error("No sbc_matrix records found in $(relpath(SBC_DIR, REPO_ROOT)).")

    # Pivot: engine → regime → rows.
    engines = Dict{String, Any}[]
    for eng in sort(unique(getfield.(flat, :engine)); by = e -> _order(e, ENGINE_ORDER))
        eng_rows = filter(x -> x.engine == eng, flat)
        regimes = Dict{String, Any}[]
        for reg in sort(unique(getfield.(eng_rows, :regime)); by = r -> _order(r, REGIME_ORDER))
            rr = filter(x -> x.regime == reg, eng_rows)
            npass = count(x -> x.verdict == "pass", rr)
            push!(
                regimes, Dict(
                    "regime" => reg,
                    "n_attempted" => first(rr).n_attempted,
                    "n_nodes" => first(rr).n_nodes, "pc_u" => first(rr).pc_u,
                    "band95" => first(rr).band,
                    "n" => length(rr), "n_pass" => npass,
                    "rows" => [
                        Dict("cell" => x.cell, "target" => x.target, "ks" => x.ks, "verdict" => x.verdict)
                            for x in rr
                    ],
                )
            )
        end
        push!(engines, Dict("engine" => eng, "regimes" => regimes))
    end

    payload = Dict(
        "generated_at" => string(now()),
        "rank_method" => "pit",
        "engines" => engines,
        "notes" => [
            "SBC ranked by PIT (cdf(marginal, truth)) for INLA/TMB; required because INLA's posterior θ is grid-quantized.",
            "Verdict vs the 95% KS null band 1.36/√n: pass ≤ band, ≈band ≤ 1.6× band, fail otherwise.",
            "hmc_laplace runs a leaner NUTS chain per replicate (offline cost), so its band is looser than INLA/TMB; the well-identified regime (n=100 nodes) is prohibitively slow for per-replicate NUTS and is omitted.",
            "Gaussian-IID is structurally non-identified (only σ²+1/τ is identified): INLA/TMB's grid/Gaussian-MAP miscalibrate on the ridge, while hmc_laplace's NUTS samples it more faithfully.",
        ],
    )

    mkpath(dirname(DATA_OUT))
    open(DATA_OUT, "w") do io
        JSON3.pretty(io, payload)
    end

    println("Wrote $(length(engines)) engine tab(s) → $(relpath(DATA_OUT, REPO_ROOT))")
    for e in engines
        regs = join(["$(r["regime"]) $(r["n_pass"])/$(r["n"])" for r in e["regimes"]], "  ·  ")
        println("  - $(e["engine"]):  ", regs)
    end
    return DATA_OUT
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
