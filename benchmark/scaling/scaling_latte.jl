# Latte INLA scaling: spatial Poisson + Matérn-SPDE on the SHARED meshes from
# gen_meshes.R. Fit time vs latent dimension n. Also writes y.csv per level so
# R-INLA and Turing fit the identical data on the identical mesh.
#
#   julia --project=benchmark benchmark/scaling/scaling_latte.jl [n_levels]

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Latte
using GaussianMarkovRandomFields
using Distributions
using DynamicPPL
using LinearAlgebra, SparseArrays, Random, Statistics, Printf
using JSON3, CSV, DataFrames
using Ferrite, FerriteGmsh, Gmsh, LibGEOS

const WORKDIR = joinpath(@__DIR__, "_workdir")

# Rebuild R-INLA's mesh (nodes/triangles) as a Ferrite FEM discretization.
function load_discretization(d)
    nodes = CSV.read(joinpath(d, "nodes.csv"), DataFrame)
    tris = CSV.read(joinpath(d, "triangles.csv"), DataFrame)
    fnodes = [Ferrite.Node((nodes.x[i], nodes.y[i])) for i in 1:nrow(nodes)]
    elems = [Ferrite.Triangle((tris.v1[i], tris.v2[i], tris.v3[i])) for i in 1:nrow(tris)]
    grid = Ferrite.Grid(elems, fnodes)
    ip = Ferrite.Lagrange{Ferrite.RefTriangle, 1}()
    qr = Ferrite.QuadratureRule{Ferrite.RefTriangle}(2)
    return FEMDiscretization(grid, ip, qr)
end

@latte function scaling_poisson(y, base_matern, A_obs)
    τ_matern ~ PCPrior.Precision(1.0; α = 0.5)
    range_matern ~ PCPrior.Range(0.3; p = 0.5)
    β ~ MvNormal(zeros(1), 100.0 * I(1))
    field ~ base_matern(τ = τ_matern, range = range_matern)
    η = β[1] .+ A_obs * field
    for i in eachindex(y)
        y[i] ~ Poisson(exp(η[i]); check_args = false)
    end
end

# Draw a field from the prior, push through the obs projection, Poisson counts.
function simulate(base_matern, A_obs; range0 = 0.3, τ0 = 1.0, β0 = 0.3, seed = 1)
    Random.seed!(seed)
    f = rand(base_matern(τ = τ0, range = range0))
    η = β0 .+ A_obs * f
    return [rand(Poisson(exp(clamp(η[i], -20.0, 11.0)))) for i in eachindex(η)]
end

function main(args)
    nlev = isempty(args) ? 99 : parse(Int, args[1])
    levels = JSON3.read(read(joinpath(WORKDIR, "levels.json"), String))
    rows = Dict{String, Any}[]
    for lev in levels[1:min(nlev, length(levels))]
        d = joinpath(WORKDIR, "mesh_$(lev.level)")
        disc = load_discretization(d)
        base_matern = MaternModel(disc; smoothness = 0)
        obs = CSV.read(joinpath(d, "obs_coords.csv"), DataFrame)
        coords = Matrix(hcat(obs.s1, obs.s2))
        A_obs = evaluation_matrix(base_matern, coords)
        n = length(base_matern)
        m = size(coords, 1)
        y = simulate(base_matern, A_obs)
        CSV.write(joinpath(d, "y.csv"), DataFrame(y = y))   # shared data for other engines
        lgm = scaling_poisson(y, base_matern, A_obs)
        # Lean accumulators: marginal likelihood only — matches what R-INLA
        # computes (posterior + mlik, no DIC/WAIC/CPO) and avoids the per-grid
        # per-obs predictor-variance pass that otherwise dominates at scale.
        accum = (MarginalLogLikelihoodStrategy(),)
        t_cold = @elapsed inla(lgm, y; progress = false, accumulators = accum)
        nw = n > 8000 ? 2 : 3
        ws = [@elapsed inla(lgm, y; progress = false, accumulators = accum) for _ in 1:nw]
        t_warm = median(ws)
        push!(rows, Dict("level" => lev.level, "n" => n, "m_obs" => m, "t_cold" => t_cold, "t_warm" => t_warm, "maxy" => maximum(y)))
        # Persist after EACH level so a slow top level can't lose the rest.
        open(joinpath(WORKDIR, "latte_scaling.json"), "w") do io
            JSON3.pretty(io, rows)
        end
        @printf("[done] level=%d n=%d m=%d cold=%.2fs warm=%.3fs maxy=%d\n", lev.level, n, m, t_cold, t_warm, maximum(y))
        flush(stdout)
    end
    return println("wrote latte_scaling.json (", length(rows), " levels)")
end

main(ARGS)
