using GaussianMarkovRandomFields
using SelectedInversion
using SparseArrays
using LinearAlgebra: Symmetric
import Distributions

export selected_covariance, selinv_mat

"""
    selected_covariance(q) -> AbstractMatrix

Selected inverse of the posterior precision — the entries of `Σ = Q⁻¹` on the
Cholesky factor's sparsity pattern, with off-pattern entries reading as `0`.
For a constrained `q` the constraint Woodbury correction is applied on that
pattern. The concrete return type depends on the solver backend (e.g. a
`Symmetric` sparse matrix, a plain `SparseMatrixCSC`, or a supernodal matrix);
consume it through `diag` and scalar `[i, j]` indexing.

This is one of the three covariance primitives of Latte's posterior-query
interface (alongside [`conditional_column`](@ref) and [`lincomb_variance`](@ref)).
The engine consumes the posterior `q` through these plus the
`Distributions.AbstractMvNormal` methods (`mean`/`var`/`std`/`logpdf`/
`logdetcov`/`rand`); a non-GMRF backend supplies a `q <: AbstractMvNormal`
and methods for these three generics. The default methods here back the
sparse-GMRF case via `GaussianMarkovRandomFields.selinv`.

Off-pattern entries are `0` by construction — this is a faithful port of the
historical `selinv_mat` behavior, not a new approximation.
"""
function selected_covariance(x::GMRF)
    return GaussianMarkovRandomFields.selinv(GaussianMarkovRandomFields.linsolve_cache(x))
end

function selected_covariance(x::GaussianMarkovRandomFields.WorkspaceGMRF)
    GaussianMarkovRandomFields.ensure_loaded!(x)
    base_selinv = GaussianMarkovRandomFields.selinv(x.workspace)
    if x.constraints !== nothing
        _update_sparsely!(base_selinv, x.constraints.A_tilde_T, x.constraints.L_c)
    end
    return base_selinv
end

# Transitional alias: `selinv_mat` was the GMRF-specific name before the
# posterior-query interface was introduced. Kept so diagonal-only callers
# (TMB / HMC-Laplace per-θ summaries) keep working unchanged.
selinv_mat(x) = selected_covariance(x)

# Woodbury correction for constrained selected inverse:
#   Σ_constrained = Σ - Σ A' (A Σ A')⁻¹ A Σ
# Only updates existing nonzero positions (the selected inverse sparsity pattern).

function _update_sparsely!(Σ::SparseMatrixCSC, ΣA_T, AΣA_T_cho)
    for j in 1:size(Σ, 2)
        rhs_vec = AΣA_T_cho \ ΣA_T[j, :]
        rng = nzrange(Σ, j)
        rows = Σ.rowval[rng]
        Σ.nzval[rng] .-= ΣA_T[rows, :] * rhs_vec
    end
    return
end

# GMRFs.jl's `selinv` returns `Symmetric(sparse(...))`. Forward to `parent`
# — only one triangle is stored, but the `Symmetric` wrapper makes reads
# symmetric so updating the stored triangle is sufficient.
_update_sparsely!(Σ::Symmetric, ΣA_T, AΣA_T_cho) =
    _update_sparsely!(parent(Σ), ΣA_T, AΣA_T_cho)

function _update_sparsely!(Σ::SupernodalMatrix, ΣA_T, AΣA_T_cho)
    # Map permuted supernode indices → depermuted (original) indices for ΣA_T lookup
    perm_to_deperm = invperm(Σ.invperm)

    for s in 1:Σ.n_super
        chunk = get_chunk(Σ, s)
        perm_rows, perm_cols = get_row_col_idcs(Σ, s)

        deperm_rows = perm_to_deperm[collect(perm_rows)]
        deperm_cols = perm_to_deperm[collect(perm_cols)]

        # RHS = (AΣA')⁻¹ * ΣA'[cols, :]'  — shape: m × |cols|
        RHS = AΣA_T_cho \ ΣA_T[deperm_cols, :]'

        # correction = ΣA'[rows, :] * RHS  — shape: |rows| × |cols|
        correction = ΣA_T[deperm_rows, :] * RHS

        if Σ.transposed_chunks
            chunk .-= correction'
        else
            chunk .-= correction
        end
    end
    return
end

function selected_covariance(x::ConstrainedGMRF)
    base_selinv = selected_covariance(x.base_gmrf)
    _update_sparsely!(base_selinv, x.A_tilde_T, x.L_c)
    return base_selinv
end

# Dense fallback for any non-GMRF posterior that can materialise its covariance
# (e.g. a plain `Distributions.MvNormal`, or a low-rank/dense backend `q`). GMRF
# posteriors use the selinv methods above (more specific). A precision-free
# backend that cannot afford a dense Σ overrides this with its own method.
selected_covariance(q::Distributions.AbstractMvNormal) = Distributions.cov(q)
