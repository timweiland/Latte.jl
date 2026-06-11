# Export a single posterior-overlay curve (median-KS latent component) for the
# Latte.jl docs benchmark page, Tokyo benchmark.
#
# Reuses the Tokyo comparison model (RW2SumOnly + tokyo_model) via the DEFAULT
# compact path. Runs Latte once for the per-day latent marginals, loads R-INLA's
# cached :x marginals, computes per-component KS with the shared `_ks_density`,
# picks the component whose KS is the block median, and writes
# _workdir/overlay.json with both posterior density curves on a shared grid.

include(joinpath(@__DIR__, "tokyo_compare.jl"))  # definitions only (main is guarded)

using CSV
using DataFrames
using Distributions
using JSON3
using Statistics

const REP_SYM = :x  # representative latent block: RW2 trend

# Hyperparameters with a cached R-INLA marginal, in the natural (precision)
# scale that both engines report. Each tuple is (Latte HP key, human label,
# R-INLA marginal CSV under WORKDIR). The τ precision marginal is dumped by
# tokyo_compare.R as `rinla_tau_marginal.csv` ("Precision for time").
const HYPER_REFS = [
    (:τ, "precision τ", "rinla_tau_marginal.csv"),
]

# Build a Latte-vs-R-INLA overlay entry for one hyperparameter marginal.
# Mirrors the latent path: KS is computed with the shared `_ks_density` on
# R-INLA's NATIVE grid (so it matches what the compare driver would report),
# while the 121-point padded grid carries the display curves only. The Latte
# spline density is clamped to 0 outside its support — the ±8% left pad of a
# precision marginal goes negative, where the cubic spline would extrapolate
# to NaN.
function _hyper_overlay(latte_d, label::String, xr::Vector{Float64}, dr::Vector{Float64})
    ks, _ = _ks_density(latte_d, xr, dr)

    span = xr[end] - xr[1]
    grid = collect(range(xr[1] - 0.08 * span, xr[end] + 0.08 * span; length = 121))
    lo, hi = minimum(latte_d), maximum(latte_d)
    latte_curve = map(grid) do xg
        (xg < lo || xg > hi) && return 0.0
        return pdf(latte_d, xg)
    end
    rinla_curve = map(grid) do xg
        (xg <= xr[1] || xg >= xr[end]) && return 0.0
        k = searchsortedlast(xr, xg)
        t = (xg - xr[k]) / (xr[k + 1] - xr[k])
        return (1 - t) * dr[k] + t * dr[k + 1]
    end

    return Dict(
        "name" => label,
        "ks" => round(ks, digits = 3),
        "grid" => round.(grid, digits = 4),
        "latte" => round.(latte_curve, digits = 5),
        "rinla" => round.(rinla_curve, digits = 5),
    )
end

function strip_overlay()
    data = load_tokyo()
    M = RW2SumOnly(data.n)
    # @latte builds a compact LGM (augment=false default); inla resolves the VBC
    # mean correction — the same default path the Benchmarks page reports.
    lgm = tokyo_model(data.y, data.n_trials, M)

    @info "running Latte INLA (for marginals)"
    result = inla(lgm, data.y; progress = false)
    latte_x = _user_x_marginals(result, REP_SYM)

    rinla_marg = CSV.read(joinpath(WORKDIR, "rinla_x_marginals.csv"), DataFrame)
    res = JSON3.read(read(joinpath(WORKDIR, "result.json"), String))
    t_latte_warm = Float64(res.t_latte_warm)
    t_rinla = Float64(res.t_rinla)

    n_comp = length(latte_x)

    # Per-component KS via the shared `_ks_density` helper.
    ks_per = Vector{Float64}(undef, n_comp)
    refs = Vector{Tuple{Vector{Float64}, Vector{Float64}}}(undef, n_comp)
    for c in 1:n_comp
        sub = sort!(filter(r -> r.i == c, rinla_marg), :x)
        xr = Vector{Float64}(sub.x)
        dr = Vector{Float64}(sub.density)
        refs[c] = (xr, dr)
        ks, _ = _ks_density(latte_x[c], xr, dr)
        ks_per[c] = ks
    end

    # Component whose KS is nearest the block median.
    med = median(ks_per)
    chosen = argmin(abs.(ks_per .- med))
    ks_chosen = ks_per[chosen]
    @info "chosen component" chosen ks_chosen median_ks = med

    xr, dr = refs[chosen]
    span = xr[end] - xr[1]
    grid = collect(range(xr[1] - 0.08 * span, xr[end] + 0.08 * span; length = 121))
    ld = latte_x[chosen]
    latte_d = pdf.(ld, grid)
    rinla_d = map(grid) do xg
        (xg <= xr[1] || xg >= xr[end]) && return 0.0
        k = searchsortedlast(xr, xg)
        t = (xg - xr[k]) / (xr[k + 1] - xr[k])
        return (1 - t) * dr[k] + t * dr[k + 1]
    end

    # Hyperparameter posterior overlays (additive `hypers` array). Each compared
    # hyperparameter with a cached R-INLA marginal contributes one entry on the
    # natural (precision) scale both engines report.
    hp_marg = result.hyperparameter_marginals
    hypers = Dict{String, Any}[]
    for (key, label, csv_name) in HYPER_REFS
        key in propertynames(hp_marg) || continue
        csv_path = joinpath(WORKDIR, csv_name)
        isfile(csv_path) || continue
        sub = sort!(CSV.read(csv_path, DataFrame), :x)
        xr_h = Vector{Float64}(sub.x)
        dr_h = Vector{Float64}(sub.density)
        entry = _hyper_overlay(getproperty(hp_marg, key), label, xr_h, dr_h)
        @info "hyper overlay" name = entry["name"] ks = entry["ks"]
        push!(hypers, entry)
    end

    out = Dict(
        "id" => "tokyo",
        "dataset" => "Tokyo rainfall",
        "scenario" => "Binomial + RW2 · 366 days",
        "label" => "day #$(chosen)",
        "ks" => round(ks_chosen, digits = 3),
        "speedup" => round(t_rinla / t_latte_warm, digits = 1),
        "grid" => round.(grid, digits = 4),
        "latte" => round.(latte_d, digits = 5),
        "rinla" => round.(rinla_d, digits = 5),
        "hypers" => hypers,
    )
    outpath = joinpath(WORKDIR, "overlay.json")
    open(outpath, "w") do io
        JSON3.write(io, out)
    end
    @info "wrote overlay" outpath chosen ks = out["ks"] speedup = out["speedup"]
    return out
end

strip_overlay()
