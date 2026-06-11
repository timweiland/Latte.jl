# Export the Latte-vs-R-INLA posterior overlay for the landing-page visual.
#
# Reuses the Tokyo comparison model (RW2SumOnly + tokyo_model). Runs Latte once
# to get the per-day latent marginals, loads R-INLA's cached marginals, and
# writes docs/src/data/landing_overlay.json: warm times plus both posterior
# density curves for two days — the median-KS day (typical agreement) and the
# worst-KS day (honest upper bound) — so the landing can toggle between them.

include(joinpath(@__DIR__, "tokyo_compare.jl"))  # definitions only (main is guarded)

using CSV
using DataFrames
using Distributions
using JSON3
using Statistics

# Both posterior density curves for one day, on a shared grid.
function _overlay_case(latte_x, rinla_marg, day::Int, ks::Float64, key::String, label::String)
    sub = sort!(filter(r -> r.i == day, rinla_marg), :x)
    xr = Vector{Float64}(sub.x)
    dr = Vector{Float64}(sub.density)
    ld = latte_x[day]
    span = xr[end] - xr[1]
    grid = collect(range(xr[1] - 0.08 * span, xr[end] + 0.08 * span; length = 121))
    latte_d = pdf.(ld, grid)
    rinla_d = map(grid) do xg
        (xg <= xr[1] || xg >= xr[end]) && return 0.0
        k = searchsortedlast(xr, xg)
        t = (xg - xr[k]) / (xr[k + 1] - xr[k])
        return (1 - t) * dr[k] + t * dr[k + 1]
    end
    return Dict(
        "key" => key, "label" => label, "day" => day, "ks" => round(ks, digits = 3),
        "grid" => round.(grid, digits = 4),
        "latte" => round.(latte_d, digits = 5),
        "rinla" => round.(rinla_d, digits = 5),
    )
end

function export_overlay()
    data = load_tokyo()
    M = RW2SumOnly(data.n)
    # @latte builds a compact LGM (augment=false default); inla resolves the VBC
    # mean correction — the same default path the Benchmarks page reports.
    lgm = tokyo_model(data.y, data.n_trials, M)

    # Latte run is only for the (deterministic) marginal curves. The displayed
    # timings come from the canonical result.json so they match the Benchmarks page.
    @info "running Latte INLA (for marginals)"
    result = inla(lgm, data.y; progress = false)
    latte_x = _user_x_marginals(result)

    rinla_marg = CSV.read(joinpath(WORKDIR, "rinla_x_marginals.csv"), DataFrame)
    res = JSON3.read(read(joinpath(WORKDIR, "result.json"), String))
    t_latte_warm = Float64(res.t_latte_warm)
    t_rinla = Float64(res.t_rinla)

    ks_x = Vector{Float64}(res.ks_x)
    worst_day = Int(res.worst_x)
    median_day = argmin(abs.(ks_x .- median(ks_x)))   # day whose KS is nearest the median

    out = Dict(
        "dataset" => "Tokyo rainfall",
        "scenario" => "366-day binomial · RW2 smoothing",
        "t_latte_warm" => round(t_latte_warm, digits = 3),
        "t_rinla" => round(t_rinla, digits = 3),
        "speedup" => round(t_rinla / t_latte_warm, digits = 1),
        "cases" => [
            _overlay_case(latte_x, rinla_marg, median_day, ks_x[median_day], "median", "median fit"),
            _overlay_case(latte_x, rinla_marg, worst_day, ks_x[worst_day], "worst", "worst fit"),
        ],
    )
    outpath = normpath(joinpath(@__DIR__, "..", "..", "..", "..", "docs", "src", "data", "landing_overlay.json"))
    open(outpath, "w") do io
        JSON3.write(io, out)
    end
    @info "wrote overlay" outpath median_day worst_day speedup = out["speedup"]
    return out
end

export_overlay()
