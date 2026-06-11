# Crowder seeds: Latte INLA vs R-INLA, both with `simplified.laplace`.
#
# 21 plates, Binomial GLMM with 2×2 logit + iid plate random effect.
# Per-fixed-effect and per-plate b_i marginals are KS-compared against
# R-INLA's output. Inlines the model to avoid a world-age cascade
# through DPPL when including the scenario file.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using CSV
using DataFrames
using Distributions
using DynamicPPL          # full module: the @latte macro's expansion references it
using GaussianMarkovRandomFields
using GaussianMarkovRandomFields: IIDModel
using JSON3
using Latte
using LinearAlgebra
using Printf
using Statistics

const SEEDS_CSV = joinpath(@__DIR__, "seeds_data.csv")
const WORKDIR = joinpath(@__DIR__, "_workdir")

@latte function seeds_model(y, n_trials, x1, x2, n_plate)
    τ ~ PCPrior.Precision(1.0, α = 0.01)
    fixed ~ MvNormal(zeros(4), 100.0 * I(4))
    b ~ IIDModel(n_plate)(τ = τ)
    for i in eachindex(y)
        η_i = fixed[1] + fixed[2] * x1[i] + fixed[3] * x2[i] +
            fixed[4] * x1[i] * x2[i] + b[i]
        y[i] ~ Binomial(n_trials[i], 1 / (1 + exp(-η_i)); check_args = false)
    end
end

function load_seeds()
    df = CSV.read(SEEDS_CSV, DataFrame)
    return (
        n = nrow(df),
        y = Vector{Int}(df.r),
        n_trials = Vector{Int}(df.n),
        x1 = Vector{Float64}(df.x1),
        x2 = Vector{Float64}(df.x2),
    )
end

# Slice augmented latent vector to a user-named block.
function _user_marginals(result, sym::Symbol)
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

# R-INLA names for fixed effects under our parameterisation.
# Latte's `fixed` block stores (α, β1, β2, β12) in that order.
const RINLA_FIXED_NAMES = ["(Intercept)", "x1", "x2", "x1:x2"]

function main(args::Vector{String} = ARGS)
    mkpath(WORKDIR)
    data = load_seeds()
    @info "Seeds dataset" n = data.n sum_y = sum(data.y) sum_n = sum(data.n_trials)

    # ── Latte ────────────────────────────────────────────────────────
    # Default: the @latte macro builds a compact LGM and inla resolves the VBC
    # mean correction. `--augmented` opts into the legacy augmented + SLA mode.
    augmented = "--augmented" in args
    marg = augmented ? SimplifiedLaplace() : nothing   # nothing ⇒ resolve (→ VBC, compact LTM)
    @info "running Latte INLA" mode = (augmented ? "augmented + simplified.laplace (legacy)" : "compact + VBC (default)")
    lgm = seeds_model(data.y, data.n_trials, data.x1, data.x2, data.n; augment = augmented)
    # Cold = first call (includes JIT specialisation on top of precompile);
    # warm = median of 5 subsequent calls in the same process.
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
    latte_b = _user_marginals(result, :b)

    # ── R-INLA (cached unless --refresh-rinla) ───────────────────────
    rinla_marker = joinpath(WORKDIR, "rinla_fixed_marginals.csv")
    if !isfile(rinla_marker) || "--refresh-rinla" in args
        @info "running R-INLA"
        cp(SEEDS_CSV, joinpath(WORKDIR, "seeds_data.csv"); force = true)
        open(joinpath(WORKDIR, "params.json"), "w") do io
            JSON3.write(
                io, Dict(
                    "pc_U" => 1.0, "pc_alpha" => 0.01,
                    "strategy" => "simplified.laplace",
                )
            )
        end
        rscript = joinpath(@__DIR__, "seeds_compare.R")
        t_rinla = @elapsed run(`Rscript $rscript $(WORKDIR) $(WORKDIR)`)
        @info "R-INLA done" elapsed = round(t_rinla, digits = 2)
    else
        @info "loading R-INLA cache" workdir = WORKDIR
    end

    rinla_meta = JSON3.read(read(joinpath(WORKDIR, "rinla_meta.json"), String))
    rinla_fixed_df = CSV.read(joinpath(WORKDIR, "rinla_fixed_marginals.csv"), DataFrame)
    rinla_b_df = CSV.read(joinpath(WORKDIR, "rinla_b_marginals.csv"), DataFrame)

    # ── KS distances per fixed effect ────────────────────────────────
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

    # ── KS distances per plate b_i ───────────────────────────────────
    ks_b = Float64[]
    sgn_b = Float64[]
    for i in 1:data.n
        sub = filter(row -> row.i == i, rinla_b_df)
        sort!(sub, :x)
        ks, sgn = _ks_density(
            latte_b[i], Vector{Float64}(sub.x), Vector{Float64}(sub.density),
        )
        push!(ks_b, ks)
        push!(sgn_b, sgn)
    end

    worst_b = argmax(ks_b)
    println()
    println("Crowder seeds — Latte vs R-INLA (simplified.laplace), n=$(data.n)")
    println("="^70)
    println("Fixed effects:")
    for (k, nm) in enumerate(("α", "β1", "β2", "β12"))
        @printf "  %-4s (R-INLA: %-12s)  KS = %.4f   signed = %+.4f\n" nm RINLA_FIXED_NAMES[k] ks_fixed[k] sgn_fixed[k]
    end
    println()
    @printf "Plate b_i KS:  max %.4f (i*=%d)   median %.4f   count > 0.05: %d / %d\n" ks_b[worst_b] worst_b sort(ks_b)[div(length(ks_b), 2) + 1] count(>(0.05), ks_b) data.n
    println()
    @printf "Timing: Latte cold = %.2f s,  Latte warm (median of 5) = %.3f s\n" t_latte_cold t_latte_warm
    @printf "        R-INLA = %.2f s   (INLA version %s)\n" Float64(rinla_meta.elapsed_seconds) String(rinla_meta.inla_version)
    @printf "        Latte/R-INLA warm speedup: %.1fx\n" Float64(rinla_meta.elapsed_seconds) / t_latte_warm
    println()

    out = (
        scenario = "seeds",
        n = data.n,
        ks_fixed = ks_fixed, sgn_fixed = sgn_fixed,
        ks_b = ks_b, sgn_b = sgn_b, worst_b = worst_b,
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
