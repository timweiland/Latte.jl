# Tokyo rainfall: Latte INLA vs R-INLA, both with `simplified.laplace`.
#
# 366-day RW2 + binomial logit. Per-day x_t marginals are KS-compared
# against R-INLA's output. The scenario module isn't loaded — we
# inline the model here to avoid a world-age cascade through DPPL.
#
# `RW2SumOnly` drops the linear (slope) null-space constraint that
# Latte's stock `RW2Model` imposes. R-INLA's default `rw2` only
# constrains sum-to-zero, so without this wrapper we'd be benchmarking
# different priors. Dropping the slope constraint cuts the per-day
# KS distance by ~60% on Tokyo (from 0.18 to 0.07 at the worst day).

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using CSV
using DataFrames
using Distributions
using DynamicPPL: @model
using GaussianMarkovRandomFields
using GaussianMarkovRandomFields: RWModel, LatentModel
using JSON3
using Latte
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

const TOKYO_CSV = joinpath(@__DIR__, "tokyo_data.csv")
const WORKDIR = joinpath(@__DIR__, "_workdir")

@model function tokyo_model(y, n_trials, M)
    τ ~ PCPrior.Precision(1.0, α = 0.01)
    x ~ M(τ = τ)
    for t in eachindex(y)
        y[t] ~ Binomial(n_trials[t], 1 / (1 + exp(-x[t])); check_args = false)
    end
end

function load_tokyo()
    df = CSV.read(TOKYO_CSV, DataFrame)
    return (
        n = nrow(df),
        y = Vector{Int}(df.y),
        n_trials = Vector{Int}(df.n),
    )
end

# Slice the augmented latent vector down to the user-named `:x` block.
function _user_x_marginals(result, sym::Symbol = :x)
    groups = Latte.latent_groups(result)
    idx = groups[sym]
    return [result.latent_marginals[i] for i in idx]
end

# KS between an engine marginal and a (grid, density) reference.
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
    data = load_tokyo()
    @info "Tokyo dataset" n = data.n sum_y = sum(data.y) sum_n = sum(data.n_trials)

    # ── Latte ────────────────────────────────────────────────────────
    @info "running Latte INLA (simplified.laplace)"
    M = RW2SumOnly(data.n)
    dppl = tokyo_model(data.y, data.n_trials, M)
    lgm = latte_from_dppl(dppl; random = (:x,))
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
    latte_x = _user_x_marginals(result)

    # ── R-INLA (cached unless --refresh-rinla) ───────────────────────
    rinla_marker = joinpath(WORKDIR, "rinla_x_marginals.csv")
    if !isfile(rinla_marker) || "--refresh-rinla" in args
        @info "running R-INLA"
        cp(TOKYO_CSV, joinpath(WORKDIR, "tokyo_data.csv"); force = true)
        open(joinpath(WORKDIR, "params.json"), "w") do io
            JSON3.write(
                io, Dict(
                    "pc_U" => 1.0, "pc_alpha" => 0.01,
                    "strategy" => "simplified.laplace",
                )
            )
        end
        rscript = joinpath(@__DIR__, "tokyo_compare.R")
        t_rinla = @elapsed run(`Rscript $rscript $(WORKDIR) $(WORKDIR)`)
        @info "R-INLA done" elapsed = round(t_rinla, digits = 2)
    else
        @info "loading R-INLA cache" workdir = WORKDIR
    end

    rinla_meta = JSON3.read(read(joinpath(WORKDIR, "rinla_meta.json"), String))
    rinla_marg_df = CSV.read(joinpath(WORKDIR, "rinla_x_marginals.csv"), DataFrame)

    # ── KS distances per t ───────────────────────────────────────────
    ks_per_t = Float64[]
    sign_per_t = Float64[]
    for t in 1:data.n
        sub = filter(row -> row.i == t, rinla_marg_df)
        sort!(sub, :x)
        ks, sgn = _ks_density(
            latte_x[t], Vector{Float64}(sub.x), Vector{Float64}(sub.density),
        )
        push!(ks_per_t, ks)
        push!(sign_per_t, sgn)
    end

    worst_t = argmax(ks_per_t)
    println()
    println("Tokyo rainfall — Latte vs R-INLA (simplified.laplace), n=$(data.n)")
    println("="^70)
    @printf "%-30s max %.4f (t*=%d)   median %.4f   count > 0.05: %d / %d\n" "x_t KS" ks_per_t[worst_t] worst_t sort(ks_per_t)[div(length(ks_per_t), 2) + 1] count(>(0.05), ks_per_t) data.n
    @printf "%-30s signed at argmax: %+.4f\n" "" sign_per_t[worst_t]
    println()
    @printf "Timing: Latte cold = %.2f s,  Latte warm (median of 5) = %.3f s\n" t_latte_cold t_latte_warm
    @printf "        R-INLA = %.2f s   (INLA version %s)\n" Float64(rinla_meta.elapsed_seconds) String(rinla_meta.inla_version)
    @printf "        Latte/R-INLA warm speedup: %.1fx\n" Float64(rinla_meta.elapsed_seconds) / t_latte_warm
    println()

    out = (
        scenario = "tokyo_rainfall",
        n = data.n,
        ks_x = ks_per_t, sgn_x = sign_per_t, worst_x = worst_t,
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
