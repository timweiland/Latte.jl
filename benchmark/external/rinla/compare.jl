# Empirical comparison: fit toy_iid_poisson with R-INLA, compute KS
# distance vs the deterministic quadrature oracle, and report.
#
# Usage:
#   julia --project=benchmark benchmark/external/rinla/compare.jl
#         [--strategy simplified.laplace|gaussian|laplace]
#
# Workflow:
#   1. Reload the toy scenario, regenerate `y` deterministically from the
#      benchmark seed.
#   2. Write `y` + prior parameters to a workdir for the R script.
#   3. Invoke `Rscript compare.R` to fit and dump marginals.
#   4. Load the dumped marginals, build CDFs, compare to the oracle.
#   5. Print per-parameter KS for `τ` and aggregate KS for the latent
#      block, both in the same units the rest of the harness uses.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CSV
using DataFrames
using JSON3
using Printf
using StableRNGs

include(joinpath(@__DIR__, "..", "..", "utils", "reporting.jl"))
include(joinpath(@__DIR__, "..", "..", "utils", "reference_store.jl"))

const SCENARIO_FILE = joinpath(@__DIR__, "..", "..", "scenarios", "toy_iid_poisson.jl")
const ORACLE_FILE = joinpath(@__DIR__, "..", "..", "oracles", "toy_iid_poisson.jl")
const WORKDIR = joinpath(@__DIR__, "_workdir")

function _parse_strategy(args)
    for (i, a) in enumerate(args)
        a == "--strategy" && return args[i + 1]
    end
    return "simplified.laplace"
end

function _scenario_module()
    mod = Module(:RinlaCompareScenario)
    Core.eval(mod, :(using Latte))
    Core.eval(mod, :(import Main: Scenario))
    Base.include(mod, SCENARIO_FILE)
    return mod
end

function _oracle_module()
    mod = Module(:RinlaCompareOracle)
    Core.eval(mod, :(using Latte))
    Base.include(mod, ORACLE_FILE)
    return mod
end

# Build a per-parameter CDF on the R-INLA density grid via cumulative
# trapezoidal integration. R-INLA's marginal grids are non-uniform.
function _trap_cdf(x::Vector{Float64}, density::Vector{Float64})
    n = length(x)
    cdf = Vector{Float64}(undef, n)
    cdf[1] = 0.0
    for k in 2:n
        cdf[k] = cdf[k - 1] + 0.5 * (density[k - 1] + density[k]) * (x[k] - x[k - 1])
    end
    Z = cdf[end]
    Z > 0 || return cdf
    return cdf ./ Z
end

# KS distance between R-INLA's CDF (rinla_grid, rinla_cdf) and the
# oracle's (oracle_grid, oracle_cdf) on the union of grid points. Also
# returns the signed gap at the argmax for direction reporting.
function _ks_distance(
        rinla_grid::Vector{Float64}, rinla_cdf::Vector{Float64},
        oracle_grid::Vector{Float64}, oracle_cdf::Vector{Float64},
    )
    eval_pts = sort!(unique(vcat(rinla_grid, oracle_grid)))
    best_abs = 0.0
    best_signed = 0.0
    for x in eval_pts
        f_rinla = _interp_cdf(rinla_grid, rinla_cdf, x)
        f_oracle = _interp_cdf(oracle_grid, oracle_cdf, x)
        gap = f_rinla - f_oracle
        if abs(gap) > best_abs
            best_abs = abs(gap)
            best_signed = gap
        end
    end
    return best_abs, best_signed
end

function _interp_cdf(grid::Vector{Float64}, cdf::Vector{Float64}, x::Real)
    n = length(grid)
    x <= grid[1] && return 0.0
    x >= grid[end] && return 1.0
    k = searchsortedfirst(grid, x)
    t = (x - grid[k - 1]) / (grid[k] - grid[k - 1])
    return cdf[k - 1] + t * (cdf[k] - cdf[k - 1])
end

function main(args::Vector{String} = ARGS)
    strategy = _parse_strategy(args)
    mkpath(WORKDIR)

    scen_mod = _scenario_module()
    scen = Base.invokelatest(scen_mod.scenario)
    n = scen.full_n
    seed = UInt64(0x0badcafe)
    data = Base.invokelatest(scen_mod.generate_data, n; seed = seed)
    data_id = string(hash(data.y), base = 16)

    @info "comparing R-INLA against quadrature oracle" strategy n data_id

    # Step 1: write inputs for the R script.
    CSV.write(joinpath(WORKDIR, "data.csv"), DataFrame(y = data.y))
    open(joinpath(WORKDIR, "params.json"), "w") do io
        JSON3.write(
            io, Dict(
                "pc_U" => 1.0, "pc_alpha" => 0.01,
                "strategy" => strategy,
            )
        )
    end

    # Step 2: run R-INLA.
    rscript = joinpath(@__DIR__, "compare.R")
    cmd = `Rscript $rscript $(WORKDIR) $(WORKDIR)`
    @info "running R-INLA" cmd
    t_rinla = @elapsed run(cmd)
    @info "R-INLA done" elapsed_seconds = round(t_rinla, digits = 2)

    meta = JSON3.read(read(joinpath(WORKDIR, "rinla_meta.json"), String))
    @info "R-INLA meta" inla_version = meta.inla_version inla_elapsed = meta.elapsed_seconds

    # Step 3: load oracle (build fresh if not cached on disk).
    oracle_mod = _oracle_module()
    oracle = Base.invokelatest(oracle_mod.oracle_summary, data)

    # Step 4: load R-INLA marginals into per-parameter (grid, cdf) pairs.
    x_marg_df = CSV.read(joinpath(WORKDIR, "rinla_x_marginals.csv"), DataFrame)
    tau_marg_df = CSV.read(joinpath(WORKDIR, "rinla_tau_marginal.csv"), DataFrame)

    # τ
    tau_grid = Vector{Float64}(tau_marg_df.x)
    tau_cdf = _trap_cdf(tau_grid, Vector{Float64}(tau_marg_df.density))
    tau_oracle_grid = oracle.cdf_grids[1]
    tau_oracle_cdf = oracle.cdf_values[1]
    tau_ks, tau_signed = _ks_distance(tau_grid, tau_cdf, tau_oracle_grid, tau_oracle_cdf)

    # x[i]
    x_kss = Float64[]
    x_signs = Float64[]
    for i in 1:n
        sub = filter(row -> row.i == i, x_marg_df)
        sort!(sub, :x)
        rg = Vector{Float64}(sub.x)
        rcdf = _trap_cdf(rg, Vector{Float64}(sub.density))
        og = oracle.cdf_grids[i + 1]
        ocdf = oracle.cdf_values[i + 1]
        ks, sgn = _ks_distance(rg, rcdf, og, ocdf)
        push!(x_kss, ks)
        push!(x_signs, sgn)
    end

    worst_idx = argmax(x_kss)
    latent_ks_max = x_kss[worst_idx]
    latent_ks_signed = x_signs[worst_idx]

    println()
    @printf "Strategy:       %s\n" strategy
    @printf "n_obs:          %d\n" n
    @printf "INLA version:   %s\n" String(meta.inla_version)
    @printf "INLA elapsed:   %.2f s\n" Float64(meta.elapsed_seconds)
    println()
    @printf "τ KS (vs oracle):           %.4f  (signed %+0.4f)\n" tau_ks tau_signed
    @printf "x[i] KS — max over i:       %.4f  (i*=%d, signed %+0.4f)\n" latent_ks_max worst_idx latent_ks_signed
    @printf "x[i] KS — median:           %.4f\n" sort(x_kss)[div(length(x_kss), 2) + 1]
    @printf "x[i] KS — count > 0.05:     %d / %d\n" count(>(0.05), x_kss) n
    println()

    # For comparison with Latte's own numbers: dump in same shape.
    return (
        strategy = strategy,
        n = n,
        tau_ks = tau_ks, tau_signed = tau_signed,
        latent_ks_max = latent_ks_max,
        latent_ks_signed_at_argmax = latent_ks_signed,
        latent_ks_per_i = x_kss,
        worst_idx = worst_idx,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
