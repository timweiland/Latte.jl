# Diagnose the super-n^1.5 scaling: did @latte recognize a proper structured LGM
# (fast path, no AD fallback), and where does the per-fit time actually go?
#   julia --project=benchmark benchmark/scaling/inspect_scaling.jl [level=3]

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Latte, GaussianMarkovRandomFields, Distributions, DynamicPPL
using LinearAlgebra, SparseArrays, CSV, DataFrames, Profile, Printf, Statistics
using Ferrite, FerriteGmsh, Gmsh, LibGEOS

const WORKDIR = joinpath(@__DIR__, "_workdir")
lev = isempty(ARGS) ? 3 : parse(Int, ARGS[1])
d = joinpath(WORKDIR, "mesh_$lev")

function load_disc(dir)
    nodes = CSV.read(joinpath(dir, "nodes.csv"), DataFrame)
    tris = CSV.read(joinpath(dir, "triangles.csv"), DataFrame)
    fn = [Ferrite.Node((nodes.x[i], nodes.y[i])) for i in 1:nrow(nodes)]
    el = [Ferrite.Triangle((tris.v1[i], tris.v2[i], tris.v3[i])) for i in 1:nrow(tris)]
    return FEMDiscretization(Ferrite.Grid(el, fn), Ferrite.Lagrange{Ferrite.RefTriangle, 1}(), Ferrite.QuadratureRule{Ferrite.RefTriangle}(2))
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

disc = load_disc(d)
base_matern = MaternModel(disc; smoothness = 0)
obs = CSV.read(joinpath(d, "obs_coords.csv"), DataFrame)
coords = Matrix(hcat(obs.s1, obs.s2))
A_obs = evaluation_matrix(base_matern, coords)
y = CSV.read(joinpath(d, "y.csv"), DataFrame).y
lgm = scaling_poisson(y, base_matern, A_obs)

println("=== RECOGNITION  (level $lev, n=", length(base_matern), ", m=", length(y), ") ===")
println("observation_model :: ", typeof(lgm.observation_model))
println("latent_prior      :: ", typeof(lgm.latent_prior))
println("augmentation_info  : ", lgm.augmentation_info === nothing ? "none (compact)" : string(typeof(lgm.augmentation_info)))
println("latent_layout      : ", lgm.latent_layout)
println("default_marg       : ", typeof(Latte.default_marginalization(lgm)).name.name)
println("nnz(A_obs)/row     : ", round(nnz(A_obs) / size(A_obs, 1), digits = 2))
flush(stdout)

# bisect: hyperparameter mode-finding vs the rest (exploration + marginals).
# LEAN accumulators (mlik only) — profile the path the benchmark actually uses,
# so the dense/super-linear op we're hunting isn't masked by DIC/WAIC/CPO.
const ACCUM = (MarginalLogLikelihoodStrategy(),)
inla(lgm, y; progress = false, accumulators = ACCUM)  # warmup
t_full = @elapsed inla(lgm, y; progress = false, accumulators = ACCUM)
t_mode = try
    @elapsed Latte.find_hyperparameter_mode(lgm, y)
catch e
    @warn "mode timing failed" e
    NaN
end
@printf("\nt_full = %.2f s   t_mode = %.2f s   explore+marg = %.2f s\n", t_full, t_mode, t_full - t_mode)
flush(stdout)

Profile.clear()
Profile.init(n = 10^8, delay = 0.001)
Profile.@profile inla(lgm, y; progress = false, accumulators = ACCUM)
open("/tmp/scaling_prof.txt", "w") do io
    Profile.print(IOContext(io, :displaysize => (240, 320)); format = :flat, sortedby = :count, mincount = 25)
end
println("\nflat profile -> /tmp/scaling_prof.txt")
