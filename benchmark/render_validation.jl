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
using Distributions: Binomial, quantile
using StableRNGs: StableRNG

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
const SBC_DIR = joinpath(REPO_ROOT, "benchmark", "results", "sbc")
const DATA_OUT = joinpath(REPO_ROOT, "docs", "src", "data", "validation_results.json")

const ENGINE_ORDER = ["inla", "tmb", "hmc_laplace"]
const REGIME_ORDER = ["well-identified", "stress (weak-id)", "custom"]

_band(n_attempted) = 1.36 / sqrt(n_attempted)

# Fallback KS verdict (used only if a JSON predates the rank histogram).
function _ks_verdict(ks, band)
    ks === nothing && return "n/a"
    ks <= band * 1.1 && return "pass"
    ks <= band * 1.6 && return "border"
    return "fail"
end

# ── Säilynoja, Bürkner & Vehtari (2022): simultaneous-band ECDF test ──
# The principled SBC verdict. `adjust_gamma` finds the pointwise level γ such that
# an M-sample uniform rank ECDF stays inside the pointwise-γ binomial band at ALL
# B-1 bin edges *simultaneously* with probability 1-α (γ ≪ α, controlling family-
# wise error over the curve). A cell PASSES iff its observed ECDF never exits that
# γ-band. Seeded ⇒ deterministic; this is a critical-value calibration, not MCMC.
function adjust_gamma(M::Int; B::Int = 100, α::Float64 = 0.05, S::Int = 4000, seed::UInt64 = UInt64(0x5a17))
    rng = StableRNG(seed)
    zs = [k / B for k in 1:(B - 1)]
    sim = Matrix{Float64}(undef, S, B - 1)
    for s in 1:S
        u = sort(rand(rng, M))
        for k in 1:(B - 1)
            sim[s, k] = searchsortedlast(u, zs[k]) / M
        end
    end
    function coverage(γ)
        lo = [quantile(Binomial(M, z), γ / 2) / M for z in zs]
        hi = [quantile(Binomial(M, z), 1 - γ / 2) / M for z in zs]
        c = 0
        @inbounds for s in 1:S
            ok = true
            for k in 1:(B - 1)
                if sim[s, k] < lo[k] - 1.0e-9 || sim[s, k] > hi[k] + 1.0e-9
                    ok = false; break
                end
            end
            c += ok
        end
        return c / S
    end
    loγ, hiγ = 1.0e-4, α          # coverage decreases as γ grows
    for _ in 1:40
        mid = (loγ + hiγ) / 2
        coverage(mid) < 1 - α ? (hiγ = mid) : (loγ = mid)
    end
    return (loγ + hiγ) / 2
end

# PASS iff the observed rank ECDF (from the cumulative histogram) stays inside the
# γ-adjusted simultaneous band at every bin edge.
function _sailynoja_verdict(hist::Vector{Int}, M::Int, γ::Float64)
    B = length(hist)
    cum = cumsum(hist)
    for k in 1:(B - 1)
        z = k / B
        e = cum[k] / M
        lo = quantile(Binomial(M, z), γ / 2) / M
        hi = quantile(Binomial(M, z), 1 - γ / 2) / M
        (e < lo - 1.0e-9 || e > hi + 1.0e-9) && return "fail"
    end
    return "pass"
end

# ── ECDF-difference curve + band for the per-cell sparklines ──
# Evaluated at EVAL_K interior points z_k = k/EVAL_K (the 100-bin histogram makes
# these exact when EVAL_K divides 100). Curves are the *difference* ECDF(z) - z so
# a calibrated cell hugs zero; the band is the γ-adjusted simultaneous band, also
# centred on zero. Rounded for compact JSON.
const EVAL_K = 50
_eval_z() = [k / EVAL_K for k in 1:(EVAL_K - 1)]
function _ecdf_diff(hist::Vector{Int}, M::Int)
    cum = cumsum(hist)
    step = length(hist) ÷ EVAL_K
    return [round(cum[k * step] / M - k / EVAL_K, digits = 4) for k in 1:(EVAL_K - 1)]
end
function _ecdf_band(M::Int, γ::Float64)
    lo = Float64[]; hi = Float64[]
    for k in 1:(EVAL_K - 1)
        z = k / EVAL_K
        push!(lo, round(quantile(Binomial(M, z), γ / 2) / M - z, digits = 4))
        push!(hi, round(quantile(Binomial(M, z), 1 - γ / 2) / M - z, digits = 4))
    end
    return lo, hi
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

    # γ is calibrated per replicate count M (a few distinct values); cache it.
    γcache = Dict{Int, Float64}()
    γfor(M) = get!(() -> adjust_gamma(M), γcache, M)

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
                cell = String(rec.cell)
                hist = get(t, :rank_hist, nothing)
                M = Int(get(t, :n_rank, n_attempted))
                # Tiered verdict: the Säilynoja 95% simultaneous-band test defines
                # "within band" (indistinguishable from calibrated). Cells that fail
                # it are tiered by EFFECT SIZE (KS) — because at n≈10³ the test
                # detects even tiny approximation error, so significance alone would
                # flag a perfectly-usable approximation. KS ≤ 0.10 (≤10% max-CDF
                # deviation) = "minor"; larger = "substantial".
                verdict = if hist !== nothing && M > 0
                    if _sailynoja_verdict(Int.(collect(hist)), M, γfor(M)) == "pass"
                        "pass"
                    elseif ks !== nothing && ks <= 0.1
                        "minor"
                    else
                        "substantial"
                    end
                else
                    _ks_verdict(ks, band)
                end
                ecdf_diff = (hist !== nothing && M > 0) ? _ecdf_diff(Int.(collect(hist)), M) : nothing
                push!(
                    flat, (;
                        engine = eng, regime = regime, n_attempted = n_attempted,
                        n_nodes = n_nodes, pc_u = pc_u, band = band, M = M,
                        cell = cell, target = String(t.target),
                        ks = ks, verdict = verdict, ecdf_diff = ecdf_diff,
                        non_identified = cell == "normal_iid",
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
            bM = first(rr).M
            band_lo, band_hi = _ecdf_band(bM, γfor(bM))
            push!(
                regimes, Dict(
                    "regime" => reg,
                    "n_attempted" => first(rr).n_attempted,
                    "n_nodes" => first(rr).n_nodes, "pc_u" => first(rr).pc_u,
                    "band95" => first(rr).band,
                    "n" => length(rr),
                    "n_pass" => count(x -> x.verdict == "pass", rr),
                    "n_minor" => count(x -> x.verdict == "minor", rr),
                    "n_substantial" => count(x -> x.verdict == "substantial", rr),
                    "band_lo" => band_lo, "band_hi" => band_hi,
                    "rows" => [
                        Dict(
                                "cell" => x.cell, "target" => x.target, "ks" => x.ks,
                                "verdict" => x.verdict, "non_identified" => x.non_identified,
                                "ecdf_diff" => x.ecdf_diff,
                            )
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
        "verdict_test" => "sailynoja_ecdf_simultaneous",
        "band_z" => _eval_z(),
        "engines" => engines,
        "notes" => [
            "SBC ranked by PIT (cdf(marginal, truth)) for INLA/TMB; required because INLA's posterior θ is grid-quantized.",
            "Verdict is effect-size tiered. ‘within band’ = passes the Säilynoja, Bürkner & Vehtari (2022) ECDF test with 95% SIMULTANEOUS confidence bands (rank ECDF inside the band at every point at once). At n≈10³ replicates that test detects even tiny error, so a cell failing it is tiered by KS effect size: ‘minor’ = KS ≤ 0.10 (≤10% max-CDF deviation — approximation-level, practically fine), ‘substantial’ = KS > 0.10. This separates an approximate-but-usable method from a genuinely-off one; a pure significance verdict would flag every approximation at this n.",
            "hmc_laplace runs a leaner NUTS chain per replicate (offline cost); the well-identified regime (n=100 nodes) is prohibitively slow for per-replicate NUTS and is omitted.",
            "Gaussian-IID (tagged ‘non-identified’) is a deliberate stress case: y~N(x,σ), x~N(0,1/τ) ⇒ only σ²+1/τ is identified. SBC against the EXACT posterior here is uniform (reference + harness validated), and a faithful sampler (hmc_laplace) recovers it (KS ~0.06); INLA's grid integration of the degenerate ridge does not (a finer grid barely helps), an inherent limit of grid-based hp exploration, not an implementation error. RW1 structure breaks the degeneracy and all engines recover.",
        ],
    )

    mkpath(dirname(DATA_OUT))
    open(DATA_OUT, "w") do io
        JSON3.pretty(io, payload)
    end

    println("Wrote $(length(engines)) engine tab(s) → $(relpath(DATA_OUT, REPO_ROOT))")
    for e in engines
        regs = join(
            ["$(r["regime"]): $(r["n_pass"]) within / $(r["n_minor"]) minor / $(r["n_substantial"]) subst." for r in e["regimes"]],
            "  ·  ",
        )
        println("  - $(e["engine"]):  ", regs)
    end
    return DATA_OUT
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
