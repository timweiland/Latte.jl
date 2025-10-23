using GaussianMarkovRandomFields
using SelectedInversion
using SparseArrays

export selinv_mat

function selinv_mat(x::GMRF)
    return GaussianMarkovRandomFields.selinv(x.linsolve_cache)
end

function _update_sparsely!(Σ::SparseMatrixCSC, ΣA_T, AΣA_T_cho)
    Is, Js, Vs = findnz(Σ)
    for j in 1:size(Σ, 2)
        rhs_vec = AΣA_T_cho \ ΣA_T[j, :]
        rng = nzrange(Σ, j)
        rows = Σ.rowval[rng]
        Σ.nzval[rng] .-= ΣA_T[rows, :] * rhs_vec
    end
    return
end

function selinv_mat(x::ConstrainedGMRF)
    base_selinv = selinv_mat(x.base_gmrf)
    _update_sparsely!(base_selinv, x.A_tilde_T, x.L_c)
    return base_selinv
end
