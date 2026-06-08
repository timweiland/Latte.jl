# Experiment (not committed): quantify the real bottleneck found by profiling —
# GMRFs caches the selected inverse via `selinv_cache::SparseMatrixCSC = selinv(F).Z`,
# which converts the SupernodalMatrix through the GENERIC element-wise getindex
# constructor. SelectedInversion ships a vectorized `sparse(::SupernodalMatrix)`.
# Time both on a real SPDE selinv result.

include(joinpath(@__DIR__, "spdetoy_compare.jl"))   # definitions only (main guarded)

using SparseArrays, LinearAlgebra
import SelectedInversion

const GMRF2 = GaussianMarkovRandomFields

disc, n_nodes = load_discretization(WORKDIR)
base = MaternModel(disc; smoothness = 0)
Q = GMRF2.precision_matrix(base; τ = 2.0, range = 0.4)          # SPDE field precision (1680)
Qpost = Symmetric(sparse(Q) + 0.5I)                            # posterior-like SPD
F = cholesky(Qpost)
Z = SelectedInversion.selinv(F; depermute = true).Z            # what GMRFs grabs
println("Z type      : ", typeof(Z))
println("nnz(sparse) : ", nnz(sparse(Z)))

# what GMRFs does today (typed field assignment ⇒ generic convert ⇒ getindex):
gmrfs_convert() = SparseMatrixCSC{Float64, Int}(Z)
# the smart vectorized path SelectedInversion ships:
smart_sparse() = sparse(Z)
# marginals only need the diagonal:
diag_only() = SelectedInversion.selinv_diag(F; depermute = true)

# verify identical result
md = maximum(abs, gmrfs_convert() - smart_sparse())
println("max|generic - sparse| = ", md)

gmrfs_convert(); smart_sparse(); diag_only()                   # warmup
N = 20
t_gen = @elapsed for _ in 1:N
    gmrfs_convert()
end
t_smart = @elapsed for _ in 1:N
    smart_sparse()
end
t_diag = @elapsed for _ in 1:N
    diag_only()
end
@info "selinv extraction (per call, $N reps)" generic_getindex_ms = round(1000t_gen / N, digits = 2) smart_sparse_ms = round(1000t_smart / N, digits = 2) diag_only_ms = round(1000t_diag / N, digits = 2) sparse_speedup = round(t_gen / t_smart, digits = 1) diag_speedup = round(t_gen / t_diag, digits = 1)
