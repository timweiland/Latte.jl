# Export the Latte-vs-R-INLA posterior overlay for the Benchmarks-page strip,
# for the "paranaprec" benchmark (Gamma + RW1 + SPDE, shared mesh).
#
# Reuses paranaprec_compare.jl (definitions only — main is guarded behind the
# PROGRAM_FILE check). Builds the LGM via the DEFAULT compact path, runs Latte
# once for the per-node SPDE field marginals, loads R-INLA's cached field
# marginals, picks the field node whose KS is the MEDIAN of the block, and
# writes _workdir/overlay.json with both posterior density curves on a shared
# grid.

include(joinpath(@__DIR__, "paranaprec_compare.jl"))  # definitions only (main is guarded)

using CSV
using DataFrames
using Distributions
using JSON3
using Statistics

# Build Latte-vs-R-INLA overlay curves for each hyperparameter the compare
# script cross-validates. `latte_density(s)` returns Latte's marginal density in
# the SAME scale as R-INLA's cached CSV; `latte_cdf(s)` returns the matching CDF
# so the reported KS reproduces paranaprec_compare.jl on R-INLA's native grid.
function _hyper_overlay(name, csvpath, latte_density, latte_cdf)
    g, d = _read_marg(csvpath)          # R-INLA grid + density (in the cached scale)

    # KS on R-INLA's native grid — identical to the compare script
    ks = _ks_cdf(latte_cdf, g, d)

    # shared 121-point grid over R-INLA support, padded ±8%
    span = g[end] - g[1]
    grid = collect(range(g[1] - 0.08 * span, g[end] + 0.08 * span; length = 121))

    # the hyperparameter scales (range, SD) live on the positive half-line; the
    # ±8% pad can dip ≤0 where the density is undefined → clamp those to 0
    latte_d = map(grid) do s
        s <= 0 && return 0.0
        v = latte_density(s)
        isfinite(v) ? v : 0.0
    end
    rinla_d = map(grid) do xg
        (xg <= g[1] || xg >= g[end]) && return 0.0
        k = searchsortedlast(g, xg)
        t = (xg - g[k]) / (g[k + 1] - g[k])
        return (1 - t) * d[k] + t * d[k + 1]
    end

    return Dict(
        "name" => name,
        "ks" => round(ks, digits = 3),
        "grid" => round.(grid, digits = 4),
        "latte" => round.(latte_d, digits = 5),
        "rinla" => round.(rinla_d, digits = 5),
    )
end

function build_hyper_overlays(result, res)
    rm = result.hyperparameter_marginals[:range_field]
    τm = result.hyperparameter_marginals[:τ_field]
    τr = result.hyperparameter_marginals[:τ_rw1]

    # SD overlay from a precision marginal: s = 1/sqrt(τ), τ = 1/s²,
    # f_S(s) = f_τ(1/s²)·|d(1/s²)/ds| = f_τ(1/s²)·2/s³, F_S(s) = 1 - F_τ(1/s²).
    sd_density(τmarg) = s -> pdf(τmarg, 1 / s^2) * 2 / s^3
    sd_cdf(τmarg) = s -> 1 - cdf(τmarg, 1 / s^2)

    overlays = Any[]
    push!(
        overlays, _hyper_overlay(
            "field range", joinpath(WORKDIR, "rinla_range_marginal.csv"),
            x -> pdf(rm, x), x -> cdf(rm, x),
        )
    )
    push!(
        overlays, _hyper_overlay(
            "field SD", joinpath(WORKDIR, "rinla_stdev_marginal.csv"),
            sd_density(τm), sd_cdf(τm),
        )
    )
    push!(
        overlays, _hyper_overlay(
            "RW1 SD", joinpath(WORKDIR, "rinla_rw1_sd_marginal.csv"),
            sd_density(τr), sd_cdf(τr),
        )
    )

    # correctness gate: each KS must match the cached benchmark value
    refs = Dict(
        "field range" => Float64(res.ks_range),
        "field SD" => Float64(res.ks_stdev),
        "RW1 SD" => Float64(res.ks_rw1_sd),
    )
    for h in overlays
        ref = refs[h["name"]]
        @info "hyper overlay" name = h["name"] ks = h["ks"] ks_ref = round(ref, digits = 4) Δ = round(abs(h["ks"] - ref), digits = 4)
    end
    return overlays
end

function export_overlay()
    # ── rebuild the model + data exactly as main() does (default compact path) ──
    pj = joinpath(WORKDIR, "params.json")
    p_raw = JSON3.read(read(pj, String))
    p = (
        sigma_field_U = Float64(p_raw.sigma_field_U), sigma_field_p = Float64(p_raw.sigma_field_p),
        range_U = Float64(p_raw.range_U), range_p = Float64(p_raw.range_p),
        rw1_sigma_U = Float64(p_raw.rw1_sigma_U), rw1_sigma_alpha = Float64(p_raw.rw1_sigma_alpha),
        prec_intercept = Float64(p_raw.prec_intercept),
    )

    data = CSV.read(joinpath(WORKDIR, "parana_data.csv"), DataFrame)
    coords = Matrix{Float64}(hcat(data.s1, data.s2))
    y = Vector{Float64}(data.y)
    grp = Vector{Int}(data.grp)
    n_groups = nrow(CSV.read(joinpath(WORKDIR, "rw1_groups.csv"), DataFrame))
    n = length(y)

    disc, n_nodes = load_discretization(WORKDIR)
    A_spde = evaluation_matrix(disc, coords)
    A_rw1 = sparse(1:n, grp, ones(n), n, n_groups)
    base_matern = MaternModel(disc; smoothness = 0)
    rw1 = RWModel{1}(n_groups; scale_model = true)
    perm = node_to_dof(disc, n_nodes)

    # default compact LGM (augment=false); inla resolves VBC marginalization
    lgm = parana_model(y, base_matern, A_spde, rw1, A_rw1, p)

    @info "running Latte INLA (for SPDE field marginals)"
    result = inla(lgm, y; progress = false)

    # ── slice the SPDE field block; R-INLA node k ↔ Latte field[perm[k]] ──
    latte_field = result.latent_marginals.field
    fmarg = CSV.read(joinpath(WORKDIR, "rinla_field_marginals.csv"), DataFrame)

    # cached benchmark KS for the field block (for the correctness gate + median)
    res = JSON3.read(read(joinpath(WORKDIR, "result.json"), String))
    ks_field_ref = Vector{Float64}(res.ks_field)

    # per-node KS we compute here (must match ks_field_ref)
    ks_field = Float64[
        _ks_density(
                latte_field[perm[k]],
                Vector{Float64}((s = fmarg[fmarg.node .== k, :]).x),
                Vector{Float64}(s.density),
            ) for k in 1:n_nodes
    ]

    # node whose KS is nearest the median of the block
    med = median(ks_field)
    node = argmin(abs.(ks_field .- med))
    ks_chosen = ks_field[node]

    @info "chosen field node" node ks_chosen ks_ref = ks_field_ref[node] median = med

    # ── shared grid over R-INLA support padded ±8% ──
    sub = sort!(fmarg[fmarg.node .== node, :], :x)
    xr = Vector{Float64}(sub.x)
    dr = Vector{Float64}(sub.density)
    span = xr[end] - xr[1]
    grid = collect(range(xr[1] - 0.08 * span, xr[end] + 0.08 * span; length = 121))

    ld = latte_field[perm[node]]
    latte_d = pdf.(ld, grid)
    rinla_d = map(grid) do xg
        (xg <= xr[1] || xg >= xr[end]) && return 0.0
        k = searchsortedlast(xr, xg)
        t = (xg - xr[k]) / (xr[k + 1] - xr[k])
        return (1 - t) * dr[k] + t * dr[k + 1]
    end

    speedup = Float64(res.t_rinla) / Float64(res.t_latte_warm)

    # ── hyperparameter overlays (mirror paranaprec_compare.jl exactly) ──
    hypers = build_hyper_overlays(result, res)

    out = Dict(
        "id" => "paranaprec",
        "dataset" => "Paraná precipitation",
        "scenario" => "Gamma + RW1 + SPDE",
        "label" => "field node #$(node)",
        "ks" => round(ks_chosen, digits = 3),
        "speedup" => round(speedup, digits = 1),
        "grid" => round.(grid, digits = 4),
        "latte" => round.(latte_d, digits = 5),
        "rinla" => round.(rinla_d, digits = 5),
        "hypers" => hypers,
    )
    outpath = joinpath(WORKDIR, "overlay.json")
    open(outpath, "w") do io
        JSON3.write(io, out)
    end
    @info "wrote overlay" outpath node ks = out["ks"] speedup = out["speedup"]
    return out
end

export_overlay()
