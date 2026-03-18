using GaussianMarkovRandomFields
using SelectedInversion
using SparseArrays

export selinv_mat

function selinv_mat(x::GMRF)
    return GaussianMarkovRandomFields.selinv(GaussianMarkovRandomFields.linsolve_cache(x))
end

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

function selinv_mat(x::ConstrainedGMRF)
    base_selinv = selinv_mat(x.base_gmrf)
    _update_sparsely!(base_selinv, x.A_tilde_T, x.L_c)
    return base_selinv
end
