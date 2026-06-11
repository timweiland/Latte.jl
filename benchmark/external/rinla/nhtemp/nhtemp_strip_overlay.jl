# Export the Latte-vs-R-INLA posterior overlay strip for the Benchmarks page.
#
# Reuses the nhtemp comparison model (RW2SumOnly + nhtemp_model). Runs Latte
# once via the DEFAULT compact + VBC path to get the per-year RW2 latent
# marginals, loads R-INLA's cached marginals, picks the median-KS year, and
# writes _workdir/overlay.json: both posterior density curves on a shared grid
# plus the chosen KS and the canonical speedup.
#
# Additively, it also exports the hyperparameter posterior overlays (obs SD and
# field SD) as a `hypers` array. R-INLA caches both hyperparameters on the
# PRECISION scale; Latte reports obs SD on the SD scale and the RW2 smoothing
# parameter on the PRECISION scale. We compare both on the interpretable SD
# scale (matching the spdetoy "obs SD" / "field Stdev" convention): R-INLA's
# precision density is pushed forward to SD via the change-of-variables
# σ = τ^(-1/2), f_σ(s) = f_τ(1/s²)·2/s³, and Latte's precision marginal is
# evaluated through the same transform.

include(joinpath(@__DIR__, "nhtemp_compare.jl"))  # definitions only (main is guarded)

using CSV
using DataFrames
using Distributions
using JSON3
using Statistics

# Engine wrapper that views a precision-scale marginal `m` on the SD scale
# s = τ^(-1/2). `cdf` flips because the map is decreasing; `pdf` carries the
# Jacobian |dτ/ds| = 2/s³.
struct PrecisionAsSD{M}
    m::M
end
Distributions.cdf(w::PrecisionAsSD, s::Real) = s <= 0 ? 0.0 : 1 - cdf(w.m, 1 / s^2)
Distributions.pdf(w::PrecisionAsSD, s::Real) = s <= 0 ? 0.0 : pdf(w.m, 1 / s^2) * (2 / s^3)
Base.Broadcast.broadcastable(w::PrecisionAsSD) = Ref(w)

# Push an R-INLA precision marginal (grid `xr`, density `dr`) forward to the SD
# scale, returning a sorted (grid, density) pair on s = τ^(-1/2).
function _precision_marginal_as_sd(xr::Vector{Float64}, dr::Vector{Float64})
    s = 1.0 ./ sqrt.(xr)
    ds = dr .* (2.0 ./ (s .^ 3))            # |dτ/ds| = 2/s³
    ord = sortperm(s)
    return s[ord], ds[ord]
end

# Build one hyperparameter overlay entry: 121-pt grid over R-INLA's (transformed)
# support padded ±8%, Latte density on the grid, R-INLA density linearly
# interpolated (0 outside support), and the KS via the same trapezoid-CDF rule
# as `_ks_density`.
function _hyper_overlay(name::String, latte_engine, xr::Vector{Float64}, dr::Vector{Float64})
    span = xr[end] - xr[1]
    # SD scale is non-negative; the ±8% pad must not push the floor below 0.
    lo = max(0.0, xr[1] - 0.08 * span)
    grid = collect(range(lo, xr[end] + 0.08 * span; length = 121))
    latte_d = pdf.(latte_engine, grid)
    rinla_d = map(grid) do xg
        (xg <= xr[1] || xg >= xr[end]) && return 0.0
        k = searchsortedlast(xr, xg)
        t = (xg - xr[k]) / (xr[k + 1] - xr[k])
        return (1 - t) * dr[k] + t * dr[k + 1]
    end
    ks, _ = _ks_density(latte_engine, xr, dr)
    return Dict(
        "name" => name,
        "ks" => round(ks, digits = 3),
        "grid" => round.(grid, digits = 4),
        "latte" => round.(latte_d, digits = 5),
        "rinla" => round.(rinla_d, digits = 5),
    )
end

function strip_overlay()
    data = load_nhtemp()
    # @latte builds a compact LGM (augment=false default); inla resolves the VBC
    # mean correction — the same default path the Benchmarks page reports.
    lgm = nhtemp_model(data.y, data.n, RW2SumOnly(data.n))

    # Latte run is only for the (deterministic) marginal curves. The displayed
    # timing/KS come from the canonical result.json so they match the page.
    @info "running Latte INLA (for marginals)"
    result = inla(lgm, data.y; progress = false)
    latte_x = _user_marginals(result, :x)

    rinla_x_df = CSV.read(joinpath(WORKDIR, "rinla_x_marginals.csv"), DataFrame)
    res = JSON3.read(read(joinpath(WORKDIR, "result.json"), String))
    t_latte_warm = Float64(res.t_latte_warm)
    t_rinla = Float64(res.t_rinla)
    ks_x_ref = Vector{Float64}(res.ks_x)

    # Recompute per-component KS from the live Latte marginals vs cached R-INLA.
    ks_x = Float64[]
    for i in 1:data.n
        sub = sort!(filter(row -> row.i == i, rinla_x_df), :x)
        ks, _ = _ks_density(
            latte_x[i], Vector{Float64}(sub.x), Vector{Float64}(sub.density),
        )
        push!(ks_x, ks)
    end

    # The component whose KS is nearest the block median (typical agreement).
    med_idx = argmin(abs.(ks_x .- median(ks_x)))
    ks_chosen = ks_x[med_idx]
    @info "median component" med_idx ks_chosen ref_ks = ks_x_ref[med_idx]

    # Shared grid over the chosen component's R-INLA support, padded ±8%.
    sub = sort!(filter(row -> row.i == med_idx, rinla_x_df), :x)
    xr = Vector{Float64}(sub.x)
    dr = Vector{Float64}(sub.density)
    span = xr[end] - xr[1]
    grid = collect(range(xr[1] - 0.08 * span, xr[end] + 0.08 * span; length = 121))

    latte_d = pdf.(latte_x[med_idx], grid)
    rinla_d = map(grid) do xg
        (xg <= xr[1] || xg >= xr[end]) && return 0.0
        k = searchsortedlast(xr, xg)
        t = (xg - xr[k]) / (xr[k + 1] - xr[k])
        return (1 - t) * dr[k] + t * dr[k + 1]
    end

    # ── Hyperparameter overlays (additive; both compared on the SD scale) ──
    hm = result.hyperparameter_marginals
    hypers = Vector{Any}()

    # obs SD: Latte's :σ is already on the SD scale; R-INLA caches the
    # observation precision τ_obs → push forward to SD.
    tau_obs_path = joinpath(WORKDIR, "rinla_tau_obs_marginal.csv")
    if isfile(tau_obs_path)
        df = sort!(CSV.read(tau_obs_path, DataFrame), :x)
        xr_s, dr_s = _precision_marginal_as_sd(
            Vector{Float64}(df.x), Vector{Float64}(df.density),
        )
        push!(hypers, _hyper_overlay("obs SD", hm[:σ], xr_s, dr_s))
    else
        @warn "no cached R-INLA obs-precision marginal; skipping obs SD" tau_obs_path
    end

    # field SD: Latte's :τ_x is on the PRECISION scale; view it on the SD scale.
    # R-INLA caches the RW2 precision τ_x → push forward to SD.
    tau_x_path = joinpath(WORKDIR, "rinla_tau_x_marginal.csv")
    if isfile(tau_x_path)
        df = sort!(CSV.read(tau_x_path, DataFrame), :x)
        xr_s, dr_s = _precision_marginal_as_sd(
            Vector{Float64}(df.x), Vector{Float64}(df.density),
        )
        push!(hypers, _hyper_overlay("field SD", PrecisionAsSD(hm[:τ_x]), xr_s, dr_s))
    else
        @warn "no cached R-INLA field-precision marginal; skipping field SD" tau_x_path
    end

    label = "year $(data.year[med_idx])"
    out = Dict(
        "id" => "nhtemp",
        "dataset" => "New Haven temperature",
        "scenario" => "Normal + RW2",
        "label" => label,
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
    @info "wrote overlay" outpath label ks = out["ks"] speedup = out["speedup"] n_hypers = length(hypers)
    for h in hypers
        @info "  hyper" name = h["name"] ks = h["ks"]
    end
    return out
end

strip_overlay()
