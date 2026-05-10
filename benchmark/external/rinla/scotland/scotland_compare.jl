# Scottish lip cancer: Latte INLA vs R-INLA, both with `simplified.laplace`.
#
# 56-district Besag-ICAR Poisson with log-offset and one fixed
# covariate. Per-fixed-effect and per-district u_i marginals are
# KS-compared against R-INLA's output. Inlines the model to avoid a
# world-age cascade through DPPL when including the scenario file.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using CSV
using DataFrames
using Distributions
using DynamicPPL: @model
using GaussianMarkovRandomFields
using GaussianMarkovRandomFields: BesagModel
using JSON3
using Latte
using LinearAlgebra
using Printf
using SparseArrays
using Statistics

const SCOT_DATA_CSV = joinpath(@__DIR__, "scotland_data.csv")
const SCOT_EDGES_CSV = joinpath(@__DIR__, "scotland_edges.csv")
const WORKDIR = joinpath(@__DIR__, "_workdir")

function _adjacency(n::Int, edges::DataFrame)
    Is = Int[]
    Js = Int[]
    for row in eachrow(edges)
        i, j = Int(row.i), Int(row.j)
        push!(Is, i)
        push!(Js, j)
        push!(Is, j)
        push!(Js, i)
    end
    return sparse(Is, Js, ones(Float64, length(Is)), n, n)
end

@model function scotland_model(y, log_E, x_scaled, W, n_d)
    τ ~ PCPrior.Precision(1.0, α = 0.01)
    fixed ~ MvNormal(zeros(2), 100.0 * I(2))
    u ~ BesagModel(W)(τ = τ)
    for i in eachindex(y)
        η_i = log_E[i] + fixed[1] + fixed[2] * x_scaled[i] + u[i]
        y[i] ~ Poisson(exp(η_i); check_args = false)
    end
end

function load_scotland()
    df = CSV.read(SCOT_DATA_CSV, DataFrame)
    edges = CSV.read(SCOT_EDGES_CSV, DataFrame)
    n_d = nrow(df)
    return (
        n = n_d,
        y = Vector{Int}(df.Counts),
        log_E = log.(Vector{Float64}(df.E)),
        x_scaled = Vector{Float64}(df.X) ./ 10,
        W = _adjacency(n_d, edges),
    )
end

function _user_marginals(result, sym::Symbol)
    groups = Latte.latent_groups(result)
    idx = groups[sym]
    return [result.latent_marginals[i] for i in idx]
end

function _ks_density(
        engine, ref_grid::Vector{Float64}, ref_density::Vector{Float64},
    )
    n = length(ref_grid)
    cdf_vals = Vector{Float64}(undef, n)
    cdf_vals[1] = 0.0
    for k in 2:n
        cdf_vals[k] = cdf_vals[k - 1] +
            0.5 * (ref_density[k - 1] + ref_density[k]) * (ref_grid[k] - ref_grid[k - 1])
    end
    Z = cdf_vals[end]
    Z > 0 && (cdf_vals ./= Z)

    best_abs = 0.0
    best_signed = 0.0
    for k in 1:n
        gap = cdf(engine, ref_grid[k]) - cdf_vals[k]
        if abs(gap) > best_abs
            best_abs = abs(gap)
            best_signed = gap
        end
    end
    return best_abs, best_signed
end

const RINLA_FIXED_NAMES = ["(Intercept)", "x_scaled"]

function main(args::Vector{String} = ARGS)
    mkpath(WORKDIR)
    data = load_scotland()
    @info "Scotland dataset" n = data.n sum_y = sum(data.y)

    @info "running Latte INLA (simplified.laplace)"
    dppl = scotland_model(data.y, data.log_E, data.x_scaled, data.W, data.n)
    lgm = latte_from_dppl(dppl; random = (:fixed, :u))
    t_latte_cold = @elapsed result = inla(
        lgm, data.y;
        latent_marginalization_method = SimplifiedLaplace(),
        progress = false,
    )
    warm_times = Float64[]
    for _ in 1:5
        push!(
            warm_times, @elapsed inla(
                lgm, data.y;
                latent_marginalization_method = SimplifiedLaplace(),
                progress = false,
            )
        )
    end
    t_latte_warm = median(warm_times)
    @info "Latte done" cold = round(t_latte_cold, digits = 2) warm_median = round(t_latte_warm, digits = 3)
    latte_fixed = _user_marginals(result, :fixed)
    latte_u = _user_marginals(result, :u)

    rinla_marker = joinpath(WORKDIR, "rinla_fixed_marginals.csv")
    if !isfile(rinla_marker) || "--refresh-rinla" in args
        @info "running R-INLA"
        cp(SCOT_DATA_CSV, joinpath(WORKDIR, "scotland_data.csv"); force = true)
        cp(SCOT_EDGES_CSV, joinpath(WORKDIR, "scotland_edges.csv"); force = true)
        open(joinpath(WORKDIR, "params.json"), "w") do io
            JSON3.write(
                io, Dict(
                    "pc_U" => 1.0, "pc_alpha" => 0.01,
                    "strategy" => "simplified.laplace",
                )
            )
        end
        rscript = joinpath(@__DIR__, "scotland_compare.R")
        t_rinla = @elapsed run(`Rscript $rscript $(WORKDIR) $(WORKDIR)`)
        @info "R-INLA done" elapsed = round(t_rinla, digits = 2)
    else
        @info "loading R-INLA cache" workdir = WORKDIR
    end

    rinla_meta = JSON3.read(read(joinpath(WORKDIR, "rinla_meta.json"), String))
    rinla_fixed_df = CSV.read(joinpath(WORKDIR, "rinla_fixed_marginals.csv"), DataFrame)
    rinla_u_df = CSV.read(joinpath(WORKDIR, "rinla_u_marginals.csv"), DataFrame)

    ks_fixed = Float64[]
    sgn_fixed = Float64[]
    for (k, nm) in enumerate(RINLA_FIXED_NAMES)
        sub = filter(row -> row.name == nm, rinla_fixed_df)
        sort!(sub, :x)
        ks, sgn = _ks_density(
            latte_fixed[k], Vector{Float64}(sub.x), Vector{Float64}(sub.density),
        )
        push!(ks_fixed, ks)
        push!(sgn_fixed, sgn)
    end

    ks_u = Float64[]
    sgn_u = Float64[]
    for i in 1:data.n
        sub = filter(row -> row.i == i, rinla_u_df)
        sort!(sub, :x)
        ks, sgn = _ks_density(
            latte_u[i], Vector{Float64}(sub.x), Vector{Float64}(sub.density),
        )
        push!(ks_u, ks)
        push!(sgn_u, sgn)
    end

    worst_u = argmax(ks_u)
    println()
    println("Scottish lip cancer — Latte vs R-INLA (simplified.laplace), n=$(data.n)")
    println("="^70)
    println("Fixed effects:")
    for (k, nm) in enumerate(("α", "β"))
        @printf "  %-4s (R-INLA: %-12s)  KS = %.4f   signed = %+.4f\n" nm RINLA_FIXED_NAMES[k] ks_fixed[k] sgn_fixed[k]
    end
    println()
    @printf "District u_i KS:  max %.4f (i*=%d)   median %.4f   count > 0.05: %d / %d\n" ks_u[worst_u] worst_u sort(ks_u)[div(length(ks_u), 2) + 1] count(>(0.05), ks_u) data.n
    println()
    @printf "Timing: Latte cold = %.2f s,  Latte warm (median of 5) = %.3f s\n" t_latte_cold t_latte_warm
    @printf "        R-INLA = %.2f s   (INLA version %s)\n" Float64(rinla_meta.elapsed_seconds) String(rinla_meta.inla_version)
    @printf "        Latte/R-INLA warm speedup: %.1fx\n" Float64(rinla_meta.elapsed_seconds) / t_latte_warm
    println()

    out = (
        scenario = "scotland",
        n = data.n,
        ks_fixed = ks_fixed, sgn_fixed = sgn_fixed,
        ks_u = ks_u, sgn_u = sgn_u, worst_u = worst_u,
        t_latte_cold = t_latte_cold,
        t_latte_warm = t_latte_warm,
        t_rinla = Float64(rinla_meta.elapsed_seconds),
        inla_version = String(rinla_meta.inla_version),
    )
    open(joinpath(WORKDIR, "result.json"), "w") do io
        JSON3.write(io, out)
    end
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
