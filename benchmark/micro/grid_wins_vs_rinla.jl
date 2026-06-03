# Cross-scenario grid-coarsening win, lined up against R-INLA.
#
# For each R-INLA-comparison scenario we time `inla` on the REAL dataset
# under two exploration grids:
#   fine          — GridExplorationStrategy(0.75, 6.0)   (pre-change grid default)
#   coarse        — GridExplorationStrategy()            (current grid default, 1.0/2.5)
# both via `latte_from_dppl` and the current default latent marginalization
# (`SimplifiedLaplace`), so the timing reflects what a user gets out of the
# box. R-INLA numbers are the stored `elapsed_seconds` from each scenario's
# `rinla_meta.json` (INLA 25.6.7, int.strategy=grid, simplified.laplace).
#
# Run:  julia --project=benchmark benchmark/micro/grid_wins_vs_rinla.jl [--json out.json]

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)

using Latte
using Printf

const FINE = GridExplorationStrategy(integration_step_z = 0.75, max_log_drop = 6.0)
const COARSE = GridExplorationStrategy()   # current default (1.0, 2.5)

# (scenario file, generate_data n, #hp, stored R-INLA elapsed_seconds)
const SCENARIOS = [
    ("toy_iid_poisson", 50, 1, 1.2989),
    ("seeds", 21, 1, 2.8626),
    ("scotland", 56, 1, 1.4852),
    ("tokyo_rainfall", 366, 1, 1.5091),
    ("nhtemp", 60, 2, 1.5824),
    ("epil", 236, 2, 1.879),
]

# Load a scenario file into a fresh module with Latte (for PCPrior etc.) and
# the deps the files import, so `build_model`/`generate_data` are usable
# without colliding across scenarios.
function load_scenario(path)
    m = Module(gensym(:Scn))
    Core.eval(m, :(using Latte))
    Core.eval(m, :(using DynamicPPL: @model))
    Core.eval(m, :(using Distributions, LinearAlgebra, SparseArrays, Statistics))
    Core.eval(m, :(using GaussianMarkovRandomFields))
    for pkg in (:CSV, :DataFrames, :StableRNGs)
        try
            Core.eval(m, :(using $pkg))
        catch
        end
    end
    Base.include(m, path)
    return m
end

best(f) = minimum(@elapsed(f()) for _ in 1:3)

runinla(lgm, y, strat) = inla(
    lgm, y; progress = false,
    exploration_strategy = strat,   # default latent method (SimplifiedLaplace)
)

rows = NamedTuple[]
for (id, n, nhp, rinla_s) in SCENARIOS
    path = joinpath(@__DIR__, "..", "scenarios", "$(id).jl")
    local m, data, lgm
    try
        m = load_scenario(path)
        data = Base.invokelatest(m.generate_data, n)
        lgm = latte_from_dppl(Base.invokelatest(m.build_model, data); random = m.RANDOM_SYMS)
    catch e
        @warn "skip $id (load/build failed)" exception = (e, catch_backtrace())
        continue
    end
    y = data.y
    # warmups
    runinla(lgm, y, FINE)
    runinla(lgm, y, COARSE)

    res_fine = inla(lgm, y; progress = false, exploration_strategy = FINE)
    res_coarse = inla(lgm, y; progress = false, exploration_strategy = COARSE)
    npts_fine = length(res_fine.exploration.grid_points)
    npts_coarse = length(res_coarse.exploration.grid_points)

    t_fine = best(() -> runinla(lgm, y, FINE))
    t_coarse = best(() -> runinla(lgm, y, COARSE))
    nlat = length(lgm.latent_prior)
    push!(rows, (; id, n = length(y), nlat, nhp, npts_fine, npts_coarse, t_fine, t_coarse, rinla_s))
    @printf("  %-16s done  (fine %.2fs / coarse %.2fs / R-INLA %.2fs)\n", id, t_fine, t_coarse, rinla_s)
end

println("\n─── Grid-coarsening win vs R-INLA (warm inla, SimplifiedLaplace, best of 3) ────────")
@printf(
    "%-16s %4s %5s %3s  %-13s  %-13s  %-9s  %s\n",
    "scenario", "n", "nlat", "hp", "fine grid", "coarse(def)", "R-INLA", "coarse/R-INLA",
)
for r in rows
    @printf(
        "%-16s %4d %5d %3d  %5.2fs (%2dpt)  %5.2fs (%2dpt)  %6.2fs   %.2fx\n",
        r.id, r.n, r.nlat, r.nhp,
        r.t_fine, r.npts_fine, r.t_coarse, r.npts_coarse, r.rinla_s,
        r.t_coarse / r.rinla_s,
    )
end

function _json_arg(a)
    for (i, x) in enumerate(a)
        x == "--json" && i < length(a) && return a[i + 1]
    end
    return nothing
end
jp = _json_arg(ARGS)
if jp !== nothing
    open(jp, "w") do io
        print(io, "[")
        for (i, r) in enumerate(rows)
            i > 1 && print(io, ",")
            print(
                io, "{\"id\":\"", r.id, "\",\"n\":", r.n, ",\"nlat\":", r.nlat, ",\"nhp\":", r.nhp,
                ",\"fine_s\":", r.t_fine, ",\"coarse_s\":", r.t_coarse,
                ",\"npts_fine\":", r.npts_fine, ",\"npts_coarse\":", r.npts_coarse,
                ",\"rinla_s\":", r.rinla_s, "}"
            )
        end
        print(io, "]\n")
    end
    println("wrote $jp")
end
