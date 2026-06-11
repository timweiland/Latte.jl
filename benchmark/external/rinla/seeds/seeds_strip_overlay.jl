# Export the Latte-vs-R-INLA posterior overlay for the Benchmarks page strip.
#
# Reuses the Crowder seeds comparison model (Binomial GLMM + IID plate effect).
# Runs Latte once via the default compact path, slices the per-plate b_i latent
# marginals, loads R-INLA's cached b_i marginals, and writes _workdir/overlay.json
# with the posterior density curves for the median-KS plate (a representative fit).

include(joinpath(@__DIR__, "seeds_compare.jl"))  # definitions only (main is guarded)

using CSV
using DataFrames
using Distributions
using JSON3
using Statistics

function export_overlay()
    data = load_seeds()

    # Default path: @latte builds a compact LGM (augment=false) and inla resolves
    # the VBC mean correction — the same default the Benchmarks page reports.
    lgm = seeds_model(data.y, data.n_trials, data.x1, data.x2, data.n)

    @info "running Latte INLA (for marginals)"
    result = inla(lgm, data.y; progress = false)
    latte_b = _user_marginals(result, :b)

    # Cached R-INLA b_i marginals (long form: i, x, density).
    rinla_b_df = CSV.read(joinpath(WORKDIR, "rinla_b_marginals.csv"), DataFrame)

    # Per-plate KS via the shared helper, then pick the median-KS plate.
    n_plate = length(latte_b)
    ks_b = Float64[]
    for i in 1:n_plate
        sub = sort!(filter(row -> row.i == i, rinla_b_df), :x)
        ks, _ = _ks_density(
            latte_b[i], Vector{Float64}(sub.x), Vector{Float64}(sub.density),
        )
        push!(ks_b, ks)
    end
    median_plate = argmin(abs.(ks_b .- median(ks_b)))
    chosen_ks = ks_b[median_plate]

    # Shared grid over the chosen plate's R-INLA support, padded ±8%.
    sub = sort!(filter(row -> row.i == median_plate, rinla_b_df), :x)
    xr = Vector{Float64}(sub.x)
    dr = Vector{Float64}(sub.density)
    span = xr[end] - xr[1]
    grid = collect(range(xr[1] - 0.08 * span, xr[end] + 0.08 * span; length = 121))

    latte_d = pdf.(latte_b[median_plate], grid)
    rinla_d = map(grid) do xg
        (xg <= xr[1] || xg >= xr[end]) && return 0.0
        k = searchsortedlast(xr, xg)
        t = (xg - xr[k]) / (xr[k + 1] - xr[k])
        return (1 - t) * dr[k] + t * dr[k + 1]
    end

    # ── Hyperparameter overlays ──────────────────────────────────────
    # Mirror the latent recipe per compared hyperparameter: a 121-point grid
    # over R-INLA's marginal support (±8% pad), Latte's density on it, R-INLA's
    # density linearly interpolated (0 outside support), KS via _ks_density.
    # Both engines report τ on the precision scale, so no transform is applied.
    hyper_specs = [
        (name = "precision τ", sym = :τ, csv = "rinla_tau_marginal.csv"),
    ]
    hypers = Dict{String, Any}[]
    for spec in hyper_specs
        csvpath = joinpath(WORKDIR, spec.csv)
        isfile(csvpath) || continue  # no cached R-INLA marginal ⇒ skip
        haskey(result.hyperparameter_marginals, spec.sym) || continue
        latte_h = result.hyperparameter_marginals[spec.sym]

        rh = sort!(CSV.read(csvpath, DataFrame), :x)
        xh = Vector{Float64}(rh.x)
        dh = Vector{Float64}(rh.density)
        ks_h, _ = _ks_density(latte_h, xh, dh)

        span_h = xh[end] - xh[1]
        grid_h = collect(range(xh[1] - 0.08 * span_h, xh[end] + 0.08 * span_h; length = 121))
        # Precision marginals are positive; the spline returns NaN below its
        # support (e.g. the ±8% pad reaching τ ≤ 0), where density is 0.
        latte_dh = map(grid_h) do xg
            d = pdf(latte_h, xg)
            isfinite(d) ? d : 0.0
        end
        rinla_dh = map(grid_h) do xg
            (xg <= xh[1] || xg >= xh[end]) && return 0.0
            k = searchsortedlast(xh, xg)
            t = (xg - xh[k]) / (xh[k + 1] - xh[k])
            return (1 - t) * dh[k] + t * dh[k + 1]
        end

        push!(
            hypers, Dict(
                "name" => spec.name,
                "ks" => round(ks_h, digits = 3),
                "grid" => round.(grid_h, digits = 4),
                "latte" => round.(latte_dh, digits = 5),
                "rinla" => round.(rinla_dh, digits = 5),
            )
        )
        @info "hyper overlay" name = spec.name ks = round(ks_h, digits = 4)
    end

    # Warm-time speedup from the canonical result.json.
    res = JSON3.read(read(joinpath(WORKDIR, "result.json"), String))
    speedup = round(Float64(res.t_rinla) / Float64(res.t_latte_warm), digits = 1)

    out = Dict(
        "id" => "seeds",
        "dataset" => "Crowder seeds",
        "scenario" => "Binomial GLMM · IID plate",
        "label" => "plate $(median_plate)",
        "ks" => round(chosen_ks, digits = 3),
        "speedup" => speedup,
        "grid" => round.(grid, digits = 4),
        "latte" => round.(latte_d, digits = 5),
        "rinla" => round.(rinla_d, digits = 5),
        "hypers" => hypers,
    )
    outpath = joinpath(WORKDIR, "overlay.json")
    open(outpath, "w") do io
        JSON3.write(io, out)
    end
    @info "wrote overlay" outpath median_plate chosen_ks speedup
    return out
end

export_overlay()
