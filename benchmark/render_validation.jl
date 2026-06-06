# Aggregate committed SBC calibration results
#   benchmark/results/sbc/*.json
# into the data file consumed by the validation docs page:
#   docs/src/data/validation_results.json
#
# Usage:
#   julia --project=benchmark benchmark/render_validation.jl
#
# Each SBC run becomes a structured table (one row per cell × engine × target,
# with KS-to-uniform, the 95% null band, and a PASS / ~border / FAIL verdict),
# plus a per-engine roll-up. A Validation.vue page renders these directly. This
# script does NO inference and runs NO MCMC — it only reads the committed JSONs,
# so it is fast and deterministic.

using Pkg
Pkg.activate(@__DIR__)

using JSON3
using Printf
using Dates

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
const SBC_DIR = joinpath(REPO_ROOT, "benchmark", "results", "sbc")
const DATA_OUT = joinpath(REPO_ROOT, "docs", "src", "data", "validation_results.json")

# Verdict vs the 95% KS null band (1.36/√n_success). A small slack over the band
# absorbs the multiple-testing across many cells; clear failures are ≫ band.
function _verdict(ks, band)
    ks === nothing && return "n/a"
    ks <= band * 1.1 && return "pass"
    ks <= band * 1.6 && return "border"
    return "fail"
end

_band(n_attempted) = 1.36 / sqrt(n_attempted)

# Identification regime, keyed on how much data informs the variance component
# (node count is the lever). Render is authoritative — it re-infers rather than
# trusting a possibly-stale stored label.
function _regime(n_nodes)
    n_nodes >= 100 && return "well-identified"
    n_nodes <= 50 && return "stress (weak-id)"
    return "custom"
end

# Turn one SBC-matrix JSON into a structured table + per-engine roll-up. Tolerant
# of older JSONs that predate the regime fields (defaults match those runs).
function _sbc_table(r)
    n_attempted = Int(r.n_attempted)
    n_nodes = Int(get(r, :n_nodes, 30))
    pc_u = Float64(get(r, :pc_u, 1.0))
    n_posterior = Int(get(r, :n_posterior, 0))
    regime = _regime(n_nodes)
    is_claim = Bool(get(r, :is_calibration_claim, n_attempted >= 1000))
    band = round(_band(n_attempted); digits = 4)
    rows = Dict{String, Any}[]
    # engine => [verdicts] for the roll-up
    by_engine = Dict{String, Vector{String}}()
    for rec in r.records
        eng = String(rec.engine)
        for t in rec.targets
            ks = t.ks_uniform === nothing ? nothing : Float64(t.ks_uniform)
            v = _verdict(ks, band)
            push!(
                rows, Dict(
                    "cell" => String(rec.cell), "engine" => eng,
                    "target" => String(t.target),
                    "ks" => ks, "verdict" => v,
                )
            )
            push!(get!(by_engine, eng, String[]), v)
        end
    end
    rollup = Dict{String, Any}[]
    for (eng, vs) in sort(collect(by_engine); by = first)
        npass = count(==("pass"), vs)
        push!(
            rollup, Dict(
                "engine" => eng, "n" => length(vs), "n_pass" => npass,
                "pass_frac" => round(npass / length(vs); digits = 3),
            )
        )
    end
    return Dict(
        "id" => regime == "custom" ?
            @sprintf("sbc_n%d_pcu%.1f", n_nodes, pc_u) :
            "sbc_" * replace(regime, r"[^a-z]" => ""),
        "regime" => regime,
        "n_nodes" => n_nodes, "pc_u" => pc_u,
        "n_attempted" => n_attempted, "n_posterior" => n_posterior,
        "band95" => band,
        "engines" => [String(e) for e in r.engines],
        "is_calibration_claim" => is_claim,
        "rollup" => rollup,
        "rows" => rows,
    )
end

function main()
    isdir(SBC_DIR) || error("No SBC results dir at $(relpath(SBC_DIR, REPO_ROOT)); run benchmark/sbc/sbc_matrix.jl first.")
    json_files = sort(filter(f -> endswith(f, ".json"), readdir(SBC_DIR; join = true)))
    isempty(json_files) && error("No SBC result JSONs in $(relpath(SBC_DIR, REPO_ROOT)).")

    sbc = Dict{String, Any}[]
    for f in json_files
        r = JSON3.read(read(f, String))
        get(r, :kind, "") == "sbc_matrix" || continue
        push!(sbc, _sbc_table(r))
    end
    isempty(sbc) && error("No sbc_matrix JSONs found in $(relpath(SBC_DIR, REPO_ROOT)).")

    # Stable order: well-identified first, then stress, then anything else.
    sort!(sbc, by = d -> (d["regime"] == "well-identified" ? 0 : d["regime"] == "stress (weak-id)" ? 1 : 2, d["id"]))

    payload = Dict(
        "generated_at" => string(now()),
        "rank_method" => "pit",   # SBC ranked by PIT (marginal CDF) for grid/Laplace engines
        "sbc" => sbc,
        "notes" => [
            "SBC ranked by PIT (cdf(marginal, truth)) for INLA/TMB; required because INLA's posterior θ is grid-quantized.",
            "Verdict vs the 95% KS null band 1.36/√n_success: pass ≤ band, border ≤ 1.6× band, fail otherwise.",
            "Gaussian-IID is structurally non-identified (only σ²+1/τ is identified) and is expected to fail.",
        ],
    )

    mkpath(dirname(DATA_OUT))
    open(DATA_OUT, "w") do io
        JSON3.pretty(io, payload)
    end

    println("Wrote $(length(sbc)) SBC table(s) → $(relpath(DATA_OUT, REPO_ROOT))")
    for d in sbc
        rollup = join(["$(r["engine"]) $(r["n_pass"])/$(r["n"])" for r in d["rollup"]], "  ")
        println("  - $(d["id"])  (band $(d["band95"]))  ", rollup)
    end
    return DATA_OUT
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
