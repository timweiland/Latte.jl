# GMRFs `refactorize!` re-converts the Julia Symmetric{SparseMatrixCSC} to a
# CHOLMOD.Sparse and re-runs `check_sparse` on EVERY factorization (the profile
# showed cholmod_l_check_sparse ~11% of a warm hyperparameter_logpdf eval).
# The sparsity pattern is invariant across Newton steps and θ-grid points, so a
# persistent CHOLMOD.Sparse (values updated in place) + cholesky!(F, S; check=false)
# would skip both the re-conversion and the structural check.
#
# This isolates that overhead on the real scaling meshes:
#   A) cholesky!(F, Symmetric(Q))    — current GMRFs path (convert + check + factorize)
#   B) cholesky!(F, S; check=false)  — S built once, reused (factorize only)
# (A) - (B) is the removable per-factorization overhead. Numerics must match.
#
#   julia --project=benchmark benchmark/scaling/chol_refactor_bench.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GaussianMarkovRandomFields, LinearAlgebra, SparseArrays, CSV, DataFrames, Printf
using Ferrite, FerriteGmsh, Gmsh, LibGEOS
using SparseArrays.CHOLMOD: Sparse
import GaussianMarkovRandomFields: precision_matrix

const WORKDIR = joinpath(@__DIR__, "_workdir")

function load_disc(dir)
    nodes = CSV.read(joinpath(dir, "nodes.csv"), DataFrame)
    tris = CSV.read(joinpath(dir, "triangles.csv"), DataFrame)
    fn = [Ferrite.Node((nodes.x[i], nodes.y[i])) for i in 1:nrow(nodes)]
    el = [Ferrite.Triangle((tris.v1[i], tris.v2[i], tris.v3[i])) for i in 1:nrow(tris)]
    return FEMDiscretization(Ferrite.Grid(el, fn), Ferrite.Lagrange{Ferrite.RefTriangle, 1}(), Ferrite.QuadratureRule{Ferrite.RefTriangle}(2))
end

bel(f, n = 20) = (f(); minimum(@elapsed(f()) for _ in 1:n))

println("level   n        A: chol!(Symmetric)   B: chol!(Sparse,nocheck)   overhead   logdet match")
for lev in (2, 3, 4)
    d = joinpath(WORKDIR, "mesh_$lev")
    disc = load_disc(d)
    bm = MaternModel(disc; smoothness = 0)
    n = length(bm)
    # Representative posterior-precision-like matrix: prior precision + a positive
    # diagonal (stands in for the likelihood Hessian A'WA contribution).
    Qp = sparse(precision_matrix(bm; τ = 1.0, range = 0.3))
    Qp = Qp + Qp'                                  # ensure structurally symmetric
    Q = Symmetric(Qp + 2.0 * I)

    F = cholesky(Q)
    S = Sparse(Q)

    fA() = cholesky!(F, Q)
    fB() = cholesky!(F, S; check = false)

    # numeric agreement: both must yield the same factor (same logdet)
    fA(); ldA = logdet(F)
    fB(); ldB = logdet(F)

    tA = bel(fA); tB = bel(fB)
    @printf(
        "%-7d %-8d %14.2f ms %18.2f ms %13.2fx   |Δlogdet|=%.2e\n",
        lev, n, tA * 1.0e3, tB * 1.0e3, tA / tB, abs(ldA - ldB)
    )
end
