# Export a single Latte-vs-R-INLA posterior overlay for the SPDEtoy benchmark
# strip on the docs benchmark page.
#
# Reuses spdetoy_compare.jl (model, data + shared-mesh loaders, node→DOF bridge,
# cached R-INLA marginals, and _ks_density). Runs Latte once on the DEFAULT
# compact path, slices the SPDE field block, computes per-node KS against the
# cached R-INLA field marginals, picks the median-KS node, and writes both
# posterior density curves on a shared grid to _workdir/overlay.json.

include(joinpath(@__DIR__, "spdetoy_compare.jl"))  # definitions only (main is guarded)

using CSV
using DataFrames
using Distributions
using JSON3
using Statistics

# Build one Latte-vs-R-INLA hyperparameter overlay entry.
#
#   name          : human label (mirrors spdetoy_compare.jl's printout)
#   csv           : cached R-INLA marginal CSV in WORKDIR (x, density)
#   latte_pdf     : s -> Latte's posterior density on the *interpretable* scale
#   ks_cdf_fn     : s -> Latte's posterior CDF on that scale, passed to _ks_cdf
#                   verbatim with R-INLA's (x, density) so the KS reproduces
#                   spdetoy_compare.jl exactly.
function _hyper_overlay(name, csv, latte_pdf, ks_cdf_fn)
    path = joinpath(WORKDIR, csv)
    isfile(path) || return nothing
    xr, dr = _read_marg(path)
    # _ks_cdf expects ascending x; _read_marg returns R's emitted order.
    if !issorted(xr)
        o = sortperm(xr)
        xr, dr = xr[o], dr[o]
    end
    ks, _ = _ks_cdf(ks_cdf_fn, xr, dr)

    span = xr[end] - xr[1]
    grid = collect(range(xr[1] - 0.08 * span, xr[end] + 0.08 * span; length = 121))
    latte_d = latte_pdf.(grid)
    rinla_d = map(grid) do xg
        (xg <= xr[1] || xg >= xr[end]) && return 0.0
        k = searchsortedlast(xr, xg)
        t = (xg - xr[k]) / (xr[k + 1] - xr[k])
        return (1 - t) * dr[k] + t * dr[k + 1]
    end

    return Dict(
        "name" => name,
        "ks" => round(ks, digits = 3),
        "grid" => round.(grid, digits = 4),
        "latte" => round.(latte_d, digits = 5),
        "rinla" => round.(rinla_d, digits = 5),
    )
end

# Mirror the three hyperparameter comparisons in spdetoy_compare.jl: obs SD and
# field range are compared directly on Latte's marginals; field Stdev compares
# the τ marginal mapped to σ_field = 1/√τ. For the density curve that same map
# is a change of variables: pdf_σ(s) = pdf_τ(1/s²)·2/s³, consistent with the
# CDF 1 - cdf_τ(1/s²) used for the KS.
function export_hyper_overlays(result)
    σm = result.hyperparameter_marginals[:σ]
    rm = result.hyperparameter_marginals[:range_matern]
    τm = result.hyperparameter_marginals[:τ_matern]

    entries = Dict{String, Any}[]
    specs = (
        (
            "obs SD", "rinla_sigma_obs_marginal.csv",
            s -> pdf(σm, s), s -> cdf(σm, s),
        ),
        (
            "field range", "rinla_range_marginal.csv",
            s -> pdf(rm, s), s -> cdf(rm, s),
        ),
        (
            "field SD", "rinla_stdev_marginal.csv",
            s -> pdf(τm, 1 / s^2) * 2 / s^3, s -> 1 - cdf(τm, 1 / s^2),
        ),
    )
    for (name, csv, lpdf, lcdf) in specs
        e = _hyper_overlay(name, csv, lpdf, lcdf)
        if e === nothing
            @warn "no cached R-INLA marginal — skipping hyperparameter" name csv
        else
            @info "hyper overlay" name ks = e["ks"]
            push!(entries, e)
        end
    end
    return entries
end

function export_overlay()
    # ── params + shared mesh (must already exist; produced by the R run) ──
    pj = joinpath(WORKDIR, "params.json")
    isfile(pj) || error("missing $pj — run spdetoy_compare.jl first")
    p_raw = JSON3.read(read(pj, String))
    p = (
        sigma_obs_U = Float64(p_raw.sigma_obs_U), sigma_obs_alpha = Float64(p_raw.sigma_obs_alpha),
        sigma_field_U = Float64(p_raw.sigma_field_U), sigma_field_p = Float64(p_raw.sigma_field_p),
        range_U = Float64(p_raw.range_U), range_p = Float64(p_raw.range_p),
        prec_intercept = Float64(p_raw.prec_intercept),
    )

    data = load_spdetoy()
    disc, n_nodes = load_discretization(WORKDIR)
    A_obs = evaluation_matrix(disc, data.coords)
    base_matern = MaternModel(disc; smoothness = 0)
    perm = node_to_dof(disc, n_nodes)

    # @latte builds the compact LGM (augment=false default). Gaussian obs ⇒
    # GaussianMarginal is exact; this is the default path the Benchmarks page reports.
    lgm = spdetoy_model(data.y, base_matern, A_obs, p)
    @info "running Latte INLA (for marginals)"
    result = inla(lgm, data.y; progress = false, latent_marginalization_method = GaussianMarginal())
    latte_field = result.latent_marginals.field            # DOF order

    # ── per-node KS vs cached R-INLA field marginals ──
    field_marg = CSV.read(joinpath(WORKDIR, "rinla_field_marginals.csv"), DataFrame)
    ks_field = Vector{Float64}(undef, n_nodes)
    for k in 1:n_nodes
        sub = field_marg[field_marg.node .== k, :]
        ks_field[k], _ = _ks_density(latte_field[perm[k]], Vector{Float64}(sub.x), Vector{Float64}(sub.density))
    end

    # node whose KS is nearest the block median (typical agreement)
    med_node = argmin(abs.(ks_field .- median(ks_field)))
    ks_chosen = ks_field[med_node]
    @info "chosen median-KS field node" med_node ks_chosen block_median = median(ks_field)

    # ── shared grid over the chosen node's R-INLA support padded ±8% ──
    sub = sort!(field_marg[field_marg.node .== med_node, :], :x)
    xr = Vector{Float64}(sub.x)
    dr = Vector{Float64}(sub.density)
    span = xr[end] - xr[1]
    grid = collect(range(xr[1] - 0.08 * span, xr[end] + 0.08 * span; length = 121))

    ld = latte_field[perm[med_node]]
    latte_d = pdf.(ld, grid)
    rinla_d = map(grid) do xg
        (xg <= xr[1] || xg >= xr[end]) && return 0.0
        k = searchsortedlast(xr, xg)
        t = (xg - xr[k]) / (xr[k + 1] - xr[k])
        return (1 - t) * dr[k] + t * dr[k + 1]
    end

    # ── timings from the canonical result.json ──
    res = JSON3.read(read(joinpath(WORKDIR, "result.json"), String))
    t_latte_warm = Float64(res.t_latte_warm)
    t_rinla = Float64(res.t_rinla)

    # ── hyperparameter posterior overlays (additive) ──
    hypers = export_hyper_overlays(result)

    out = Dict(
        "id" => "spdetoy",
        "dataset" => "SPDEtoy",
        "scenario" => "Gaussian + Matérn SPDE",
        "label" => "field node $(med_node)",
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
    @info "wrote overlay" outpath label = out["label"] ks = out["ks"] speedup = out["speedup"]
    return out
end

export_overlay()
