# Full Latte scaling curve: fit time vs n across all mesh levels, Sequential and
# Threaded, lean accumulators (mlik only — matches R-INLA's default output),
# BLAS pinned to 1. Warm (min of `reps`). Persists per level so a slow top level
# can't lose the rest.
#   julia --project=benchmark -t 10 benchmark/scaling/latte_curve.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Latte, GaussianMarkovRandomFields, Distributions, DynamicPPL, LinearAlgebra
using CSV, DataFrames, Printf, JSON3, Ferrite, FerriteGmsh, Gmsh, LibGEOS

BLAS.set_num_threads(1)
const WORKDIR = joinpath(@__DIR__, "_workdir")

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

const ACCUM = (MarginalLogLikelihoodStrategy(),)
bestof(f, reps) = (f(); minimum(@elapsed(f()) for _ in 1:reps))

results = Dict{String, Any}[]
levels = readdir(WORKDIR) |> dirs -> sort([parse(Int, replace(d, "mesh_" => "")) for d in dirs if startswith(d, "mesh_")])
@printf("Julia threads=%d  BLAS=%d\n", Threads.nthreads(), BLAS.get_num_threads())

for lev in levels
    d = joinpath(WORKDIR, "mesh_$lev")
    bm = MaternModel(load_disc(d); smoothness = 0)
    n = length(bm)
    obs = CSV.read(joinpath(d, "obs_coords.csv"), DataFrame)
    A_obs = evaluation_matrix(bm, Matrix(hcat(obs.s1, obs.s2)))
    y = CSV.read(joinpath(d, "y.csv"), DataFrame).y
    lgm = scaling_poisson(y, bm, A_obs)
    reps = n > 15000 ? 2 : 3
    t_seq = bestof(() -> inla(lgm, y; progress = false, accumulators = ACCUM, executor = SequentialExecutor()), reps)
    GC.gc()
    t_thr = bestof(() -> inla(lgm, y; progress = false, accumulators = ACCUM, executor = ThreadedExecutor()), reps)
    push!(results, Dict("level" => lev, "n" => n, "m" => length(y), "serial_s" => t_seq, "threaded_s" => t_thr))
    @printf("L%d  n=%6d  serial=%7.2fs  threaded=%7.2fs  self-speedup=%.2fx\n", lev, n, t_seq, t_thr, t_seq / t_thr)
    flush(stdout)
    open(joinpath(WORKDIR, "latte_curve.json"), "w") do io
        JSON3.write(io, Dict("threads" => Threads.nthreads(), "levels" => results))
    end
end
println("done -> _workdir/latte_curve.json")
