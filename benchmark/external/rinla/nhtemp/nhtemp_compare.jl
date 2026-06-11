# New Haven temperature: Latte INLA vs R-INLA, both with `simplified.laplace`.
#
# 60-year RW2 + Gaussian likelihood. Per-x_t and intercept marginals
# are KS-compared against R-INLA's output. Inlines the model to avoid
# a world-age cascade through DPPL when including the scenario file.
#
# `RW2SumOnly` drops the linear (slope) null-space constraint that
# Latte's stock `RW2Model` imposes. R-INLA's default `rw2` only
# constrains sum-to-zero; same wrapper as tokyo_compare.jl.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using CSV
using DataFrames
using Distributions
using DynamicPPL          # full module: the @latte macro's expansion references it
using GaussianMarkovRandomFields
using GaussianMarkovRandomFields: RWModel, LatentModel
using JSON3
using Latte
using LinearAlgebra
using Printf
using Statistics

struct RW2SumOnly{Inner <: RWModel{2}} <: LatentModel
    inner::Inner
end
RW2SumOnly(n::Int; kwargs...) = RW2SumOnly(RWModel{2}(n; kwargs...))
GaussianMarkovRandomFields.precision_matrix(m::RW2SumOnly; kwargs...) =
    GaussianMarkovRandomFields.precision_matrix(m.inner; kwargs...)
GaussianMarkovRandomFields.mean(m::RW2SumOnly; kwargs...) =
    GaussianMarkovRandomFields.mean(m.inner; kwargs...)
GaussianMarkovRandomFields.model_name(m::RW2SumOnly) =
    GaussianMarkovRandomFields.model_name(m.inner)
GaussianMarkovRandomFields.hyperparameters(m::RW2SumOnly) =
    GaussianMarkovRandomFields.hyperparameters(m.inner)
function GaussianMarkovRandomFields.constraints(m::RW2SumOnly; kwargs...)
    n = m.inner.n
    return (ones(1, n), zeros(1))
end
Base.length(m::RW2SumOnly) = length(m.inner)
function Base.getproperty(m::RW2SumOnly, s::Symbol)
    s === :inner && return getfield(m, :inner)
    s === :n && return m.inner.n
    s === :alg && return m.inner.alg
    return getfield(m, s)
end

const NHTEMP_CSV = joinpath(@__DIR__, "nhtemp_data.csv")
const WORKDIR = joinpath(@__DIR__, "_workdir")

@latte function nhtemp_model(y, n, M)
    τ_x ~ PCPrior.Precision(1.0, α = 0.01)
    σ ~ PCPrior.Sigma(1.0, α = 0.01)
    fixed ~ MvNormal(zeros(1), 100.0 * I(1))
    x ~ M(τ = τ_x)
    for t in eachindex(y)
        y[t] ~ Normal(fixed[1] + x[t], σ)
    end
end

function load_nhtemp()
    df = CSV.read(NHTEMP_CSV, DataFrame)
    return (
        n = nrow(df),
        y = Vector{Float64}(df.temp),
        year = Vector{Int}(df.year),
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

function main(args::Vector{String} = ARGS)
    mkpath(WORKDIR)
    data = load_nhtemp()
    @info "nhtemp dataset" n = data.n mean_y = round(mean(data.y), digits = 2)

    # Default: the @latte macro builds a compact LGM and inla resolves the VBC
    # mean correction. `--augmented` opts into the legacy augmented + SLA mode.
    augmented = "--augmented" in args
    marg = augmented ? SimplifiedLaplace() : nothing   # nothing ⇒ resolve (→ VBC, compact LTM)
    @info "running Latte INLA" mode = (augmented ? "augmented + simplified.laplace (legacy)" : "compact + VBC (default)")
    lgm = nhtemp_model(data.y, data.n, RW2SumOnly(data.n); augment = augmented)
    t_latte_cold = @elapsed result = inla(
        lgm, data.y;
        latent_marginalization_method = marg,
        progress = false,
    )
    warm_times = Float64[]
    for _ in 1:5
        push!(
            warm_times, @elapsed inla(
                lgm, data.y;
                latent_marginalization_method = marg,
                progress = false,
            )
        )
    end
    t_latte_warm = median(warm_times)
    @info "Latte done" cold = round(t_latte_cold, digits = 2) warm_median = round(t_latte_warm, digits = 3) augmented resolved = string(typeof(augmented ? SimplifiedLaplace() : Latte.default_marginalization(lgm)).name.name)
    latte_fixed = _user_marginals(result, :fixed)
    latte_x = _user_marginals(result, :x)

    rinla_marker = joinpath(WORKDIR, "rinla_x_marginals.csv")
    if !isfile(rinla_marker) || "--refresh-rinla" in args
        @info "running R-INLA"
        cp(NHTEMP_CSV, joinpath(WORKDIR, "nhtemp_data.csv"); force = true)
        open(joinpath(WORKDIR, "params.json"), "w") do io
            JSON3.write(
                io, Dict(
                    "pc_U_x" => 1.0, "pc_alpha_x" => 0.01,
                    "pc_U_obs" => 1.0, "pc_alpha_obs" => 0.01,
                    "strategy" => "simplified.laplace",
                )
            )
        end
        rscript = joinpath(@__DIR__, "nhtemp_compare.R")
        t_rinla = @elapsed run(`Rscript $rscript $(WORKDIR) $(WORKDIR)`)
        @info "R-INLA done" elapsed = round(t_rinla, digits = 2)
    else
        @info "loading R-INLA cache" workdir = WORKDIR
    end

    rinla_meta = JSON3.read(read(joinpath(WORKDIR, "rinla_meta.json"), String))
    rinla_fixed_df = CSV.read(joinpath(WORKDIR, "rinla_fixed_marginals.csv"), DataFrame)
    rinla_x_df = CSV.read(joinpath(WORKDIR, "rinla_x_marginals.csv"), DataFrame)

    # Intercept comparison
    sub_α = filter(row -> row.name == "(Intercept)", rinla_fixed_df)
    sort!(sub_α, :x)
    ks_α, sgn_α = _ks_density(
        latte_fixed[1], Vector{Float64}(sub_α.x), Vector{Float64}(sub_α.density),
    )

    # Per-year x_t
    ks_x = Float64[]
    sgn_x = Float64[]
    for i in 1:data.n
        sub = filter(row -> row.i == i, rinla_x_df)
        sort!(sub, :x)
        ks, sgn = _ks_density(
            latte_x[i], Vector{Float64}(sub.x), Vector{Float64}(sub.density),
        )
        push!(ks_x, ks)
        push!(sgn_x, sgn)
    end

    worst_x = argmax(ks_x)
    println()
    println("New Haven temperature — Latte vs R-INLA (simplified.laplace), n=$(data.n)")
    println("="^70)
    @printf "Intercept α:  KS = %.4f   signed = %+.4f\n" ks_α sgn_α
    println()
    @printf "x_t (RW2):    max %.4f (t*=%d, year %d)   median %.4f   count > 0.05: %d / %d\n" ks_x[worst_x] worst_x data.year[worst_x] sort(ks_x)[div(length(ks_x), 2) + 1] count(>(0.05), ks_x) data.n
    println()
    @printf "Timing: Latte cold = %.2f s,  Latte warm (median of 5) = %.3f s\n" t_latte_cold t_latte_warm
    @printf "        R-INLA = %.2f s   (INLA version %s)\n" Float64(rinla_meta.elapsed_seconds) String(rinla_meta.inla_version)
    @printf "        Latte/R-INLA warm speedup: %.1fx\n" Float64(rinla_meta.elapsed_seconds) / t_latte_warm
    println()

    out = (
        scenario = "nhtemp",
        n = data.n,
        ks_α = ks_α, sgn_α = sgn_α,
        ks_x = ks_x, sgn_x = sgn_x, worst_x = worst_x,
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
