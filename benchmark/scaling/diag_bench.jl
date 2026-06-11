# Micro-benchmark for the GMRFs `_row_diag_AΣAt` hotspot: the current full
# `A*Σ` product vs computing diag(AΣAᵀ) directly from selinv's obs-local blocks.
# Uses the real scaling meshes so Σ carries the true selinv fill pattern.
#   julia --project=benchmark benchmark/scaling/diag_bench.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GaussianMarkovRandomFields, LinearAlgebra, SparseArrays, CSV, DataFrames, Printf, Statistics
using Ferrite, FerriteGmsh, Gmsh, LibGEOS
import GaussianMarkovRandomFields: _posterior_cov_sparse

const WORKDIR = joinpath(@__DIR__, "_workdir")

function load_disc(dir)
    nodes = CSV.read(joinpath(dir, "nodes.csv"), DataFrame)
    tris = CSV.read(joinpath(dir, "triangles.csv"), DataFrame)
    fn = [Ferrite.Node((nodes.x[i], nodes.y[i])) for i in 1:nrow(nodes)]
    el = [Ferrite.Triangle((tris.v1[i], tris.v2[i], tris.v3[i])) for i in 1:nrow(tris)]
    return FEMDiscretization(Ferrite.Grid(el, fn), Ferrite.Lagrange{Ferrite.RefTriangle, 1}(), Ferrite.QuadratureRule{Ferrite.RefTriangle}(2))
end

bel(f) = (f(); minimum(@elapsed(f()) for _ in 1:5))

# Current GMRFs implementation: materialize the full m×n product, then diagonal.
current_given(A, Σ) = Vector{Float64}(vec(sum((A * Σ) .* A, dims = 2)))

# Direct: diag(AΣAᵀ)_i = Σ_{j,k∈supp(aᵢ)} aᵢ[j] aᵢ[k] Σ[j,k] — only the obs-local block.
function direct_given(A::SparseMatrixCSC, Σ::SparseMatrixCSC)
    At = sparse(transpose(A))           # columns of At = rows of A
    rv = rowvals(At); nz = nonzeros(At)
    m = size(A, 1)
    out = zeros(m)
    @inbounds for i in 1:m
        rng = nzrange(At, i)
        s = 0.0
        for p in rng
            ap = nz[p]; jp = rv[p]
            for q in rng
                s += ap * nz[q] * Σ[jp, rv[q]]
            end
        end
        out[i] = s
    end
    return out
end

println("level   n       m       current      direct      speedup   identical   nnz(Σ)/row")
for lev in (2, 3, 4)
    d = joinpath(WORKDIR, "mesh_$lev")
    disc = load_disc(d)
    bm = MaternModel(disc; smoothness = 0)
    obs = CSV.read(joinpath(d, "obs_coords.csv"), DataFrame)
    coords = Matrix(hcat(obs.s1, obs.s2))
    A = evaluation_matrix(bm, coords)
    prior = bm(τ = 1.0, range = 0.3)
    std(prior)                          # populate the factorization / selinv cache
    Σ = SparseMatrixCSC(_posterior_cov_sparse(prior))
    n = length(bm); m = size(A, 1)
    vc = current_given(A, Σ); vd = direct_given(A, Σ)
    ok = isapprox(vc, vd; rtol = 1.0e-6)
    tc = bel(() -> current_given(A, Σ)); td = bel(() -> direct_given(A, Σ))
    @printf(
        "%-7d %-7d %-7d %8.1f ms %8.1f ms %7.0fx   %-9s %.1f\n",
        lev, n, m, tc * 1.0e3, td * 1.0e3, tc / td, string(ok), nnz(Σ) / n
    )
end
