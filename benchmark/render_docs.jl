# Aggregate per-scenario `result.json` files (one per benchmark
# comparison runner under `benchmark/external/rinla/<id>/_workdir/`)
# into a single file consumed by the Vitepress benchmarks page:
#   `docs/src/data/benchmark_results.json`
#
# Usage:
#   julia --project=benchmark benchmark/render_docs.jl
#
# Each scenario produces a "receipt" — a small block with a title,
# scenario subtitle, comparability label, a list of rows (label / value
# / optional muted flag), and a short note. The Vue component renders
# these directly.

using Pkg
Pkg.activate(@__DIR__)

using JSON3
using Printf
using Dates

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
const RINLA_DIR = joinpath(REPO_ROOT, "benchmark", "external", "rinla")
const DATA_OUT = joinpath(REPO_ROOT, "docs", "src", "data", "benchmark_results.json")

# ── Per-scenario formatting ───────────────────────────────────────────

# Format a duration as a string with sane precision per magnitude.
function _fmt_secs(t::Real)
    if t < 0.01
        return @sprintf("%.0f ms", t * 1000)
    elseif t < 1.0
        return @sprintf("%.0f ms", t * 1000)
    elseif t < 10.0
        return @sprintf("%.2f s", t)
    else
        return @sprintf("%.1f s", t)
    end
end

# `r` is a JSON3.Object loaded from a scenario's `result.json`.
function _common_timing_rows(r)
    speedup = Float64(r.t_rinla) / Float64(r.t_latte_warm)
    return [
        Dict("label" => "Latte INLA · warm", "value" => _fmt_secs(Float64(r.t_latte_warm))),
        Dict("label" => "R-INLA", "value" => _fmt_secs(Float64(r.t_rinla)), "muted" => true),
        Dict("label" => "speedup (warm)", "value" => @sprintf("%.1fx", speedup), "muted" => true),
        Dict("label" => "Latte INLA · cold", "value" => _fmt_secs(Float64(r.t_latte_cold)), "muted" => true),
    ]
end

function _ks_summary(values, label::AbstractString)
    isempty(values) && return Dict("label" => "$label KS", "value" => "n/a")
    vmax = maximum(values)
    vmed = sort(collect(Float64, values))[div(length(values), 2) + 1]
    return Dict(
        "label" => "$label KS (max / median)",
        "value" => @sprintf("%.3f / %.3f", vmax, vmed),
    )
end

function _max_field(arr, name)
    isempty(arr) && return 0.0
    return maximum(Float64, arr)
end

# Numeric summary for the at-a-glance accuracy×speed scatter: warm speedup plus
# the KS spread of the benchmark's primary latent-marginal block (median =
# typical agreement, max = worst single component).
function _summary(r, ks_vec)
    v = sort(collect(Float64, ks_vec))
    sp = round(Float64(r.t_rinla) / Float64(r.t_latte_warm), digits = 2)
    isempty(v) && return Dict("speedup" => sp)
    return Dict(
        "speedup" => sp,
        "ks_median" => round(v[div(length(v), 2) + 1], digits = 4),
        "ks_max" => round(maximum(v), digits = 4),
    )
end

function _format_seeds(r)
    rows = [
        _ks_summary(Float64.(r.ks_fixed), "fixed (α, β1, β2, β12)"),
        _ks_summary(Float64.(r.ks_b), "plate b_i"),
        _common_timing_rows(r)...,
    ]
    return Dict(
        "id" => "seeds",
        "title" => "Crowder seeds",
        "scenario" => "Binomial GLMM · 21 plates · IID RE",
        "comparability" => "same posterior",
        "rows" => rows,
        "notes" => @sprintf(
            "fixed-effects max KS %.3f; plate REs max %.3f / median %.3f",
            _max_field(r.ks_fixed, "ks_fixed"),
            _max_field(r.ks_b, "ks_b"),
            sort(collect(Float64, r.ks_b))[div(length(r.ks_b), 2) + 1],
        ),
    )
end

function _format_scotland(r)
    rows = [
        _ks_summary(Float64.(r.ks_fixed), "fixed (α, β)"),
        _ks_summary(Float64.(r.ks_u), "district u_i"),
        _common_timing_rows(r)...,
    ]
    return Dict(
        "id" => "scotland",
        "title" => "Scottish lip cancer",
        "scenario" => "Poisson + Besag · 56 districts · log-offset",
        "comparability" => "same posterior",
        "rows" => rows,
        "notes" => @sprintf(
            "fixed-effects max KS %.3f; spatial RE max %.3f / median %.3f",
            _max_field(r.ks_fixed, "ks_fixed"),
            _max_field(r.ks_u, "ks_u"),
            sort(collect(Float64, r.ks_u))[div(length(r.ks_u), 2) + 1],
        ),
    )
end

function _format_nhtemp(r)
    rows = [
        Dict("label" => "intercept α KS", "value" => @sprintf("%.3f", Float64(r.ks_α))),
        _ks_summary(Float64.(r.ks_x), "x_t (RW2)"),
        _common_timing_rows(r)...,
    ]
    return Dict(
        "id" => "nhtemp",
        "title" => "New Haven temperature",
        "scenario" => "Normal + RW2 · 60 years · 1912–1971",
        "comparability" => "same posterior",
        "rows" => rows,
        "notes" => @sprintf(
            "α KS %.3f; RW2 max %.3f / median %.3f",
            Float64(r.ks_α),
            _max_field(r.ks_x, "ks_x"),
            sort(collect(Float64, r.ks_x))[div(length(r.ks_x), 2) + 1],
        ),
    )
end

function _format_epil(r)
    rows = [
        _ks_summary(Float64.(r.ks_fixed), "fixed (6 effects)"),
        _ks_summary(Float64.(r.ks_subj), "subject b_i"),
        _common_timing_rows(r)...,
    ]
    return Dict(
        "id" => "epil",
        "title" => "Epil (BUGS)",
        "scenario" => "Poisson + IID×IID · 59 subj × 4 visits",
        "comparability" => "same posterior",
        "rows" => rows,
        "notes" => @sprintf(
            "fixed max %.3f; subject RE max %.3f / median %.3f",
            _max_field(r.ks_fixed, "ks_fixed"),
            _max_field(r.ks_subj, "ks_subj"),
            sort(collect(Float64, r.ks_subj))[div(length(r.ks_subj), 2) + 1],
        ),
    )
end

function _format_tokyo(r)
    rows = [
        _ks_summary(Float64.(r.ks_x), "x_t (366-day RW2)"),
        _common_timing_rows(r)...,
    ]
    return Dict(
        "id" => "tokyo_rainfall",
        "title" => "Tokyo rainfall",
        "scenario" => "Binomial + RW2 · 366 days",
        "comparability" => "same posterior",
        "rows" => rows,
        "notes" => @sprintf(
            "x_t max KS %.3f / median %.3f",
            _max_field(r.ks_x, "ks_x"),
            sort(collect(Float64, r.ks_x))[div(length(r.ks_x), 2) + 1],
        ),
    )
end

function _format_spdetoy(r)
    hp = (Float64(r.ks_intercept), Float64(r.ks_sigma_obs), Float64(r.ks_range), Float64(r.ks_stdev))
    rows = [
        _ks_summary(Float64.(r.ks_field), "SPDE field node"),
        Dict(
            "label" => "hyperpar KS (β / obs SD / range / field SD)",
            "value" => @sprintf("%.3f / %.3f / %.3f / %.3f", hp...),
        ),
        _common_timing_rows(r)...,
    ]
    return Dict(
        "id" => "spdetoy",
        "title" => "SPDEtoy",
        "scenario" => "Gaussian + Matérn SPDE · 200 obs · 1680-node shared mesh",
        "comparability" => "same posterior",
        "rows" => rows,
        "notes" => @sprintf(
            "field max %.3f / median %.3f over 1680 nodes; every hyperparameter KS ≤ %.3f",
            Float64(r.ks_field_max), Float64(r.ks_field_median), maximum(hp),
        ),
    )
end

function _format_paranaprec(r)
    rw1 = sort(collect(Float64, r.ks_rw1))
    rw1_med = rw1[div(length(rw1), 2) + 1]
    rows = [
        _ks_summary(Float64.(r.ks_field), "SPDE field node"),
        Dict("label" => "RW1 node KS (max / median)", "value" => @sprintf("%.3f / %.3f", maximum(rw1), rw1_med)),
        Dict(
            "label" => "hyperpar KS (β / range / field SD / RW1 SD)",
            "value" => @sprintf(
                "%.3f / %.3f / %.3f / %.3f",
                Float64(r.ks_intercept), Float64(r.ks_range), Float64(r.ks_stdev), Float64(r.ks_rw1_sd),
            ),
        ),
        _common_timing_rows(r)...,
    ]
    return Dict(
        "id" => "paranaprec",
        "title" => "Paraná precipitation",
        "scenario" => "Gamma + RW1 + Matérn SPDE · 616 obs · 407-node mesh",
        "comparability" => "same posterior",
        "rows" => rows,
        "notes" => @sprintf(
            "RW1 and intercept agree closely; SPDE field KS ~%.2f is the weakly-identified-field floor (variance- not mean-limited)",
            Float64(r.ks_field_median),
        ),
    )
end

const FORMATTERS = Dict(
    "seeds" => _format_seeds,
    "scotland" => _format_scotland,
    "nhtemp" => _format_nhtemp,
    "epil" => _format_epil,
    "tokyo_rainfall" => _format_tokyo,
    "spdetoy" => _format_spdetoy,
    "paranaprec" => _format_paranaprec,
)

# Ordering for the receipts on the page (smallest → biggest, roughly).
const RECEIPT_ORDER = ["seeds", "scotland", "nhtemp", "tokyo_rainfall", "epil", "spdetoy", "paranaprec"]

# Primary latent-marginal KS block per scenario — drives the scatter's y-spread.
const PRIMARY_KS = Dict(
    "seeds" => :ks_b, "scotland" => :ks_u, "nhtemp" => :ks_x,
    "tokyo_rainfall" => :ks_x, "epil" => :ks_subj,
    "spdetoy" => :ks_field, "paranaprec" => :ks_field,
)

function _scenario_dirs()
    isdir(RINLA_DIR) || return String[]
    return [
        joinpath(RINLA_DIR, name) for name in readdir(RINLA_DIR)
            if isdir(joinpath(RINLA_DIR, name))
    ]
end

function main()
    receipts = Dict{String, Any}[]

    for sdir in _scenario_dirs()
        result_file = joinpath(sdir, "_workdir", "result.json")
        isfile(result_file) || continue
        r = JSON3.read(read(result_file, String))
        sid = String(r.scenario)
        fmt = get(FORMATTERS, sid, nothing)
        fmt === nothing && continue
        receipt = fmt(r)
        ksf = get(PRIMARY_KS, sid, nothing)
        if ksf !== nothing && haskey(r, ksf)
            receipt["summary"] = _summary(r, r[ksf])
        end
        push!(receipts, receipt)
    end

    # Stable, curated order; unknown receipts (none today) go last alphabetically.
    sort!(
        receipts,
        by = r -> let
            i = findfirst(==(r["id"]), RECEIPT_ORDER)
            (i === nothing ? typemax(Int) : i, r["id"])
        end,
    )

    payload = Dict(
        "generated_at" => string(now()),
        "receipts" => receipts,
    )

    mkpath(dirname(DATA_OUT))
    open(DATA_OUT, "w") do io
        JSON3.pretty(io, payload)
    end

    println("Wrote $(length(receipts)) receipt(s) → $(relpath(DATA_OUT, REPO_ROOT))")
    for r in receipts
        println("  - ", r["id"])
    end
    return DATA_OUT
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
