# Export a single posterior-overlay strip for the Latte.jl Benchmarks page.
#
# Reuses the Scotland comparison model (scotland_model + load_scotland). Runs
# Latte once via the DEFAULT compact path to get the per-district latent
# marginals (:u block), loads R-INLA's cached u_i marginals, KS-compares each
# district, picks the MEDIAN-KS district, and writes _workdir/overlay.json with
# both posterior density curves on a shared grid.

include(joinpath(@__DIR__, "scotland_compare.jl"))  # definitions only (main is guarded)

using CSV
using DataFrames
using Distributions
using JSON3
using Statistics

const REP_BLOCK = :u   # district spatial effects

# Build a Latte-vs-R-INLA overlay for one hyperparameter on a shared 121-point
# grid over R-INLA's support (padded ±8%). Mirrors the latent overlay exactly:
# Latte's density via `pdf`, R-INLA's via linear interpolation (0 outside
# support), and the KS via the compare script's `_ks_density`.
function _hyper_overlay(name, latte_marg, xr::Vector{Float64}, dr::Vector{Float64})
    span = xr[end] - xr[1]
    grid = collect(range(xr[1] - 0.08 * span, xr[end] + 0.08 * span; length = 121))
    latte_d = pdf.(latte_marg, grid)
    rinla_d = map(grid) do xg
        (xg <= xr[1] || xg >= xr[end]) && return 0.0
        k = searchsortedlast(xr, xg)
        t = (xg - xr[k]) / (xr[k + 1] - xr[k])
        return (1 - t) * dr[k] + t * dr[k + 1]
    end
    ks, _ = _ks_density(latte_marg, xr, dr)
    return Dict(
            "name" => name,
            "ks" => round(ks, digits = 3),
            "grid" => round.(grid, digits = 4),
            "latte" => round.(latte_d, digits = 5),
            "rinla" => round.(rinla_d, digits = 5),
        ), ks
end

function export_overlay()
    data = load_scotland()

    # Default (compact + VBC) path — same as main() without --augmented.
    lgm = scotland_model(data.y, data.log_E, data.x_scaled, data.W, data.n)
    @info "running Latte INLA (for marginals)"
    result = inla(lgm, data.y; progress = false)
    latte_u = _user_marginals(result, REP_BLOCK)

    rinla_u_df = CSV.read(joinpath(WORKDIR, "rinla_u_marginals.csv"), DataFrame)

    # Per-district KS vs cached R-INLA marginals.
    ks_u = Float64[]
    for i in 1:data.n
        sub = sort!(filter(row -> row.i == i, rinla_u_df), :x)
        ks, _ = _ks_density(
            latte_u[i], Vector{Float64}(sub.x), Vector{Float64}(sub.density),
        )
        push!(ks_u, ks)
    end

    # District whose KS is nearest the median of the block.
    median_i = argmin(abs.(ks_u .- median(ks_u)))
    chosen_ks = ks_u[median_i]

    # Shared grid over the chosen district's R-INLA support, padded ±8%.
    sub = sort!(filter(row -> row.i == median_i, rinla_u_df), :x)
    xr = Vector{Float64}(sub.x)
    dr = Vector{Float64}(sub.density)
    span = xr[end] - xr[1]
    grid = collect(range(xr[1] - 0.08 * span, xr[end] + 0.08 * span; length = 121))

    latte_d = pdf.(latte_u[median_i], grid)
    rinla_d = map(grid) do xg
        (xg <= xr[1] || xg >= xr[end]) && return 0.0
        k = searchsortedlast(xr, xg)
        t = (xg - xr[k]) / (xr[k + 1] - xr[k])
        return (1 - t) * dr[k] + t * dr[k + 1]
    end

    res = JSON3.read(read(joinpath(WORKDIR, "result.json"), String))
    t_rinla = Float64(res.t_rinla)
    t_latte_warm = Float64(res.t_latte_warm)

    # ── Hyperparameter overlays ─────────────────────────────────────────────
    # Only export hyperparameters that scotland_compare.jl actually validates
    # against R-INLA — i.e. ones for which result.json carries a reference KS.
    # Besag precision τ. With the R-INLA formula now using scale.model = TRUE
    # (matched to Latte's Sørbye–Rue normalization of the Besag structure), the
    # two engines parameterise τ on the same scale, so the marginals are directly
    # comparable (the prior mismatch that produced a spurious ≈0.75 KS is gone).
    latte_tau = result.hyperparameter_marginals.τ
    tau_df = sort!(CSV.read(joinpath(WORKDIR, "rinla_tau_marginal.csv"), DataFrame), :x)
    entry, ks_tau = _hyper_overlay(
        "precision τ", latte_tau,
        Vector{Float64}(tau_df.x), Vector{Float64}(tau_df.density),
    )
    hypers = Dict{String, Any}[entry]
    @info "hyper overlay" name = "precision τ" ks = round(ks_tau, digits = 4)

    out = Dict(
        "id" => "scotland",
        "dataset" => "Scottish lip cancer",
        "scenario" => "Poisson + Besag · offset",
        "label" => "district #$(median_i)",
        "ks" => round(chosen_ks, digits = 3),
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
    @info "wrote overlay" outpath median_i chosen_ks = round(chosen_ks, digits = 4) reference_ks = round(res.ks_u[median_i], digits = 4) speedup = out["speedup"] n_hypers = length(hypers)
    return out
end

export_overlay()
