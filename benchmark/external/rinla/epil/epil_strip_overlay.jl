# Export the Latte-vs-R-INLA posterior overlay for the Benchmarks page strip.
#
# Reuses the Epil comparison model + helpers from epil_compare.jl. Runs Latte
# once via the DEFAULT compact path to get the per-subject latent marginals,
# loads R-INLA's cached subject marginals, computes per-component KS, and writes
# _workdir/overlay.json for the median-KS subject (a representative fit).

include(joinpath(@__DIR__, "epil_compare.jl"))  # definitions only (main is guarded)

using CSV
using DataFrames
using Distributions
using JSON3
using Statistics

function export_overlay()
    data = load_epil()
    # Default path: the @latte macro builds a compact LGM and inla resolves the
    # VBC mean correction — the same default path the Benchmarks page reports.
    lgm = epil_model(
        data.y, data.log_base4, data.trt, data.trt_logbase4, data.log_age, data.v4,
        data.ind, data.n_subject, data.n,
    )

    @info "running Latte INLA (for marginals)"
    result = inla(lgm, data.y; progress = false)
    latte_subj = _user_marginals(result, :b_subject)

    rinla_subj_df = CSV.read(joinpath(WORKDIR, "rinla_subj_marginals.csv"), DataFrame)

    # Per-subject KS against R-INLA's cached curve.
    ks_subj = Float64[]
    for i in 1:data.n_subject
        sub = filter(row -> row.i == i, rinla_subj_df)
        sort!(sub, :x)
        ks, _ = _ks_density(
            latte_subj[i], Vector{Float64}(sub.x), Vector{Float64}(sub.density),
        )
        push!(ks_subj, ks)
    end

    # Pick the subject whose KS is the MEDIAN of the block.
    med = median(ks_subj)
    chosen = argmin(abs.(ks_subj .- med))
    chosen_ks = ks_subj[chosen]
    @info "chosen median-KS subject" chosen chosen_ks median = med

    # Shared grid of 121 points over R-INLA support, padded ±8%.
    sub = sort!(filter(row -> row.i == chosen, rinla_subj_df), :x)
    xr = Vector{Float64}(sub.x)
    dr = Vector{Float64}(sub.density)
    span = xr[end] - xr[1]
    grid = collect(range(xr[1] - 0.08 * span, xr[end] + 0.08 * span; length = 121))

    ld = latte_subj[chosen]
    latte_d = pdf.(ld, grid)
    rinla_d = map(grid) do xg
        (xg <= xr[1] || xg >= xr[end]) && return 0.0
        k = searchsortedlast(xr, xg)
        t = (xg - xr[k]) / (xr[k + 1] - xr[k])
        return (1 - t) * dr[k] + t * dr[k + 1]
    end

    # ── Hyperparameter overlays ──────────────────────────────────────────────
    # R-INLA caches the two precision marginals in NATURAL (τ) space, the same
    # space Latte's hyperparameter_marginals return. Each entry maps a Latte
    # hyper group → its cached R-INLA CSV and a human label. KS is computed with
    # the identical `_ks_density` helper used for the latent curves above.
    hyper_specs = [
        (:τ_subj, "rinla_tau_subj_marginal.csv", "subject precision τ"),
        (:τ_obs, "rinla_tau_obs_marginal.csv", "obs precision τ"),
    ]
    hyper_marginals = Latte.hyperparameter_marginals(result)
    hyper_groups = Latte.hyperparameter_groups(result)
    hypers = Vector{Dict{String, Any}}()
    for (sym, csv, label) in hyper_specs
        path = joinpath(WORKDIR, csv)
        if !isfile(path) || !haskey(hyper_groups, sym)
            @info "skipping hyper overlay (no cached R-INLA marginal)" sym csv
            continue
        end
        hd = hyper_marginals[first(hyper_groups[sym])]

        m = CSV.read(path, DataFrame)
        sort!(m, :x)
        xr_h = Vector{Float64}(m.x)
        dr_h = Vector{Float64}(m.density)

        ks_h, _ = _ks_density(hd, xr_h, dr_h)

        span_h = xr_h[end] - xr_h[1]
        grid_h = collect(range(xr_h[1] - 0.08 * span_h, xr_h[end] + 0.08 * span_h; length = 121))
        latte_dh = pdf.(hd, grid_h)
        rinla_dh = map(grid_h) do xg
            (xg <= xr_h[1] || xg >= xr_h[end]) && return 0.0
            k = searchsortedlast(xr_h, xg)
            t = (xg - xr_h[k]) / (xr_h[k + 1] - xr_h[k])
            return (1 - t) * dr_h[k] + t * dr_h[k + 1]
        end

        push!(
            hypers, Dict{String, Any}(
                "name" => label,
                "ks" => round(ks_h, digits = 3),
                "grid" => round.(grid_h, digits = 4),
                "latte" => round.(latte_dh, digits = 5),
                "rinla" => round.(rinla_dh, digits = 5),
            ),
        )
        @info "hyper overlay" sym label ks = round(ks_h, digits = 3)
    end

    res = JSON3.read(read(joinpath(WORKDIR, "result.json"), String))
    t_latte_warm = Float64(res.t_latte_warm)
    t_rinla = Float64(res.t_rinla)
    speedup = round(t_rinla / t_latte_warm, digits = 1)

    out = Dict(
        "id" => "epil",
        "dataset" => "Epil (BUGS)",
        "scenario" => "Poisson + IID",
        "label" => "subject #$(chosen)",
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
    @info "wrote overlay" outpath chosen chosen_ks = out["ks"] speedup
    return out
end

export_overlay()
