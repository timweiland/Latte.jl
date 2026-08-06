# Sparsity-pattern detection and augmentation.
#
# The GMRFWorkspace requires the prior Q's sparsity pattern to be a superset
# of the posterior Hessian pattern (= prior Q + likelihood Hessian). When the
# likelihood introduces off-diagonal Hessian entries that the prior doesn't
# anticipate (e.g. a block-diagonal latent prior + a likelihood that couples
# across blocks), we union the two patterns by adding structural zeros to Q.

using SparseArrays
using SparseConnectivityTracer: TracerLocalSparsityDetector, hessian_sparsity
using DynamicPPL: getloglikelihood
using LogDensityProblems

"""
    detect_likelihood_pattern(dppl_model, hp_names::Tuple, n_latent::Int)

Sparsity pattern of the likelihood Hessian `∂² log p(y|x) / ∂x∂xᵀ` at a probe
point, as a `SparseMatrixCSC` with structural nonzeros. Computed once at setup
so the latent-prior Q can be augmented with this pattern, making its sparsity
a superset of the posterior Hessian's.
"""
function detect_likelihood_pattern(
        dppl_model, hp_names::Tuple, n_latent::Int;
        hp_probe::NamedTuple = _hp_probe_nt(dppl_model, hp_names),
    )
    cond = DynamicPPL.fix(dppl_model, hp_probe)
    ldf = DynamicPPL.LogDensityFunction(cond, getloglikelihood)
    loglik(x) = LogDensityProblems.logdensity(ldf, x)
    return hessian_sparsity(loglik, zeros(n_latent), TracerLocalSparsityDetector())
end

"""
    augment_pattern(Q::SparseMatrixCSC, pattern)

Return a `SparseMatrixCSC` with the same numeric values as `Q` but with
`pattern`'s nonzero positions present as structural zeros. Lets the workspace
accept any Hessian whose pattern is a subset of `Q ∪ pattern` — purely a
sparsity-structure change, no numeric effect.
"""
function augment_pattern(Q::SparseMatrixCSC, pattern)
    n = size(Q, 1)
    pattern_sp = SparseMatrixCSC(pattern)
    rs_Q, cs_Q, vs_Q = findnz(Q)
    rs_P, cs_P, _ = findnz(pattern_sp)
    # (I, J, V) constructor sums duplicates, so Q-and-pattern positions keep
    # Q's value (Q + 0); pattern-only positions land as structural zeros.
    all_rs = vcat(rs_Q, rs_P)
    all_cs = vcat(cs_Q, cs_P)
    all_vs = vcat(vs_Q, zeros(length(rs_P)))
    return sparse(all_rs, all_cs, all_vs, n, n)
end
