# BUGS Epil: Latte INLA vs R-INLA, both with `simplified.laplace`.
#
# 59 patients × 4 visits = 236 obs hierarchical Poisson with subject +
# observation IID REs. Fixed-effect and per-subject RE marginals are
# KS-compared against R-INLA's output.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using CSV
using DataFrames
using Distributions
using DynamicPPL: @model
using GaussianMarkovRandomFields: IIDModel
using JSON3
using Latte
using LinearAlgebra
using Printf
using Statistics

const EPIL_CSV = joinpath(@__DIR__, "epil_data.csv")
const WORKDIR = joinpath(@__DIR__, "_workdir")

@model function epil_model(
        y, log_base4, trt, trt_logbase4, log_age, v4,
        ind, n_subject, n_obs,
    )
    τ_subj ~ PCPrior.Precision(1.0, α = 0.01)
    τ_obs ~ PCPrior.Precision(1.0, α = 0.01)
    fixed ~ MvNormal(zeros(6), 100.0 * I(6))
    b_subject ~ IIDModel(n_subject)(τ = τ_subj)
    b_obs ~ IIDModel(n_obs)(τ = τ_obs)
    for k in eachindex(y)
        η_k = fixed[1] + fixed[2] * log_base4[k] + fixed[3] * trt[k] +
            fixed[4] * trt_logbase4[k] + fixed[5] * log_age[k] + fixed[6] * v4[k] +
            b_subject[ind[k]] + b_obs[k]
        y[k] ~ Poisson(exp(η_k); check_args = false)
    end
end

function load_epil()
    df = CSV.read(EPIL_CSV, DataFrame)
    n_obs = nrow(df)
    log_base4 = log.(Vector{Float64}(df.Base) ./ 4)
    trt = Vector{Float64}(df.Trt)
    log_age = log.(Vector{Float64}(df.Age))
    v4 = Vector{Float64}(df.V4)
    ind = Vector{Int}(df.Ind)
    n_subject = maximum(ind)
    return (
        n = n_obs,
        n_subject = n_subject,
        y = Vector{Int}(df.y),
        log_base4 = log_base4,
        trt = trt,
        trt_logbase4 = trt .* log_base4,
        log_age = log_age,
        v4 = v4,
        ind = ind,
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

const RINLA_FIXED_NAMES = ["(Intercept)", "log_base4", "Trt", "trt_logbase4", "log_age", "V4"]

function main(args::Vector{String} = ARGS)
    mkpath(WORKDIR)
    data = load_epil()
    @info "Epil dataset" n = data.n n_subject = data.n_subject sum_y = sum(data.y)

    @info "running Latte INLA (simplified.laplace)"
    dppl = epil_model(
        data.y, data.log_base4, data.trt, data.trt_logbase4, data.log_age, data.v4,
        data.ind, data.n_subject, data.n,
    )
    lgm = latte_from_dppl(dppl; random = (:fixed, :b_subject, :b_obs))
    t_latte_cold = @elapsed result = inla(
        lgm, data.y;
        latent_marginalization_method = SimplifiedLaplace(),
        progress = false,
    )
    warm_times = Float64[]
    for _ in 1:3
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
    latte_subj = _user_marginals(result, :b_subject)

    rinla_marker = joinpath(WORKDIR, "rinla_fixed_marginals.csv")
    if !isfile(rinla_marker) || "--refresh-rinla" in args
        @info "running R-INLA"
        cp(EPIL_CSV, joinpath(WORKDIR, "epil_data.csv"); force = true)
        open(joinpath(WORKDIR, "params.json"), "w") do io
            JSON3.write(
                io, Dict(
                    "pc_U" => 1.0, "pc_alpha" => 0.01,
                    "strategy" => "simplified.laplace",
                )
            )
        end
        rscript = joinpath(@__DIR__, "epil_compare.R")
        t_rinla = @elapsed run(`Rscript $rscript $(WORKDIR) $(WORKDIR)`)
        @info "R-INLA done" elapsed = round(t_rinla, digits = 2)
    else
        @info "loading R-INLA cache" workdir = WORKDIR
    end

    rinla_meta = JSON3.read(read(joinpath(WORKDIR, "rinla_meta.json"), String))
    rinla_fixed_df = CSV.read(joinpath(WORKDIR, "rinla_fixed_marginals.csv"), DataFrame)
    rinla_subj_df = CSV.read(joinpath(WORKDIR, "rinla_subj_marginals.csv"), DataFrame)

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

    ks_subj = Float64[]
    sgn_subj = Float64[]
    for i in 1:data.n_subject
        sub = filter(row -> row.i == i, rinla_subj_df)
        sort!(sub, :x)
        ks, sgn = _ks_density(
            latte_subj[i], Vector{Float64}(sub.x), Vector{Float64}(sub.density),
        )
        push!(ks_subj, ks)
        push!(sgn_subj, sgn)
    end

    worst_subj = argmax(ks_subj)
    println()
    println("BUGS Epil — Latte vs R-INLA (simplified.laplace), n=$(data.n), n_subj=$(data.n_subject)")
    println("="^70)
    println("Fixed effects:")
    for (k, label) in enumerate(("α", "β_base", "β_trt", "β_int", "β_age", "β_v4"))
        @printf "  %-8s (R-INLA: %-14s)  KS = %.4f   signed = %+.4f\n" label RINLA_FIXED_NAMES[k] ks_fixed[k] sgn_fixed[k]
    end
    println()
    @printf "Subject b_i KS:  max %.4f (i*=%d)   median %.4f   count > 0.05: %d / %d\n" ks_subj[worst_subj] worst_subj sort(ks_subj)[div(length(ks_subj), 2) + 1] count(>(0.05), ks_subj) data.n_subject
    println()
    @printf "Timing: Latte cold = %.2f s,  Latte warm (median of 3) = %.3f s\n" t_latte_cold t_latte_warm
    @printf "        R-INLA = %.2f s   (INLA version %s)\n" Float64(rinla_meta.elapsed_seconds) String(rinla_meta.inla_version)
    @printf "        Latte/R-INLA warm speedup: %.1fx\n" Float64(rinla_meta.elapsed_seconds) / t_latte_warm
    println()

    out = (
        scenario = "epil",
        n = data.n,
        n_subject = data.n_subject,
        ks_fixed = ks_fixed, sgn_fixed = sgn_fixed,
        ks_subj = ks_subj, sgn_subj = sgn_subj, worst_subj = worst_subj,
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
