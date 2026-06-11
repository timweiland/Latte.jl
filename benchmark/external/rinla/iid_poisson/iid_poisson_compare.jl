# Toy IID Poisson: Latte INLA vs R-INLA, both with `simplified.laplace`.
#
# Synthetic n=50 dataset, IID Gaussian random effects, Poisson observations.
# A simpler-than-Tokyo benchmark: no smoothing prior, no per-day correlation.
# Per-x marginals are KS-compared against R-INLA's output.

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
using Printf
using StableRNGs
using Statistics

const WORKDIR = joinpath(@__DIR__, "_workdir")
const N = 50
const SEED = UInt64(0x0badcafe)

@latte function iid_poisson_model(y, n)
    τ ~ PCPrior.Precision(1.0, α = 0.01)
    x ~ IIDModel(n)(τ = τ)
    for i in eachindex(y)
        y[i] ~ Poisson(exp(x[i]); check_args = false)
    end
end

function generate_data(n::Int, seed::UInt64)
    rng = StableRNG(seed)
    true_x = randn(rng, n) .* 0.4 .+ 0.7
    y = rand.(rng, Poisson.(exp.(true_x)))
    return (; n = n, y = y, true_x = true_x)
end

function _user_x_marginals(result, sym::Symbol = :x)
    groups = Latte.latent_groups(result)
    return [result.latent_marginals[i] for i in groups[sym]]
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
    data = generate_data(N, SEED)
    @info "IID Poisson dataset" n = data.n sum_y = sum(data.y) mean_y = mean(data.y)

    # Default: the @latte macro builds a compact LGM and inla resolves the VBC
    # mean correction. `--augmented` opts into the legacy augmented + SLA mode.
    augmented = "--augmented" in args
    marg = augmented ? SimplifiedLaplace() : nothing   # nothing ⇒ resolve (→ VBC, compact LTM)
    @info "running Latte INLA" mode = (augmented ? "augmented + simplified.laplace (legacy)" : "compact + VBC (default)")
    lgm = iid_poisson_model(data.y, data.n; augment = augmented)
    t_latte = @elapsed result = inla(
        lgm, data.y;
        latent_marginalization_method = marg,
        progress = false,
    )
    @info "Latte done" elapsed = round(t_latte, digits = 2) augmented resolved = string(typeof(augmented ? SimplifiedLaplace() : Latte.default_marginalization(lgm)).name.name)
    latte_x = _user_x_marginals(result)

    rinla_marker = joinpath(WORKDIR, "rinla_x_marginals.csv")
    if !isfile(rinla_marker) || "--refresh-rinla" in args
        @info "running R-INLA"
        df = DataFrame(idx = 1:data.n, y = data.y)
        CSV.write(joinpath(WORKDIR, "data.csv"), df)
        open(joinpath(WORKDIR, "params.json"), "w") do io
            JSON3.write(
                io, Dict(
                    "pc_U" => 1.0, "pc_alpha" => 0.01,
                    "strategy" => "simplified.laplace",
                )
            )
        end
        rscript = joinpath(@__DIR__, "iid_poisson_compare.R")
        t_rinla = @elapsed run(`Rscript $rscript $(WORKDIR) $(WORKDIR)`)
        @info "R-INLA done" elapsed = round(t_rinla, digits = 2)
    else
        @info "loading R-INLA cache" workdir = WORKDIR
    end

    rinla_meta = JSON3.read(read(joinpath(WORKDIR, "rinla_meta.json"), String))
    rinla_marg_df = CSV.read(joinpath(WORKDIR, "rinla_x_marginals.csv"), DataFrame)

    ks_per_i = Float64[]
    sign_per_i = Float64[]
    for i in 1:data.n
        sub = filter(row -> row.i == i, rinla_marg_df)
        sort!(sub, :x)
        ks, sgn = _ks_density(
            latte_x[i], Vector{Float64}(sub.x), Vector{Float64}(sub.density),
        )
        push!(ks_per_i, ks)
        push!(sign_per_i, sgn)
    end

    worst_i = argmax(ks_per_i)
    println()
    println("Toy IID Poisson — Latte vs R-INLA (simplified.laplace), n=$(data.n)")
    println("="^70)
    @printf "%-30s max %.4f (i*=%d)   median %.4f   count > 0.05: %d / %d\n" "x_i KS" ks_per_i[worst_i] worst_i sort(ks_per_i)[div(length(ks_per_i), 2) + 1] count(>(0.05), ks_per_i) data.n
    @printf "%-30s signed at argmax: %+.4f\n" "" sign_per_i[worst_i]
    @printf "%-30s INLA version: %s,  elapsed: %.2f s (Latte) vs %.2f s (R-INLA)\n" "" String(rinla_meta.inla_version) t_latte Float64(rinla_meta.elapsed_seconds)
    println()

    return (ks_per_i = ks_per_i, sign_per_i = sign_per_i, worst_i = worst_i)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
