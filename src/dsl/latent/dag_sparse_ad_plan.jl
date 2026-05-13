# Phase 2 of the joint-precision cache: sparse-AD fallback path.
#
# When the body has non-linear cross-edges (or otherwise can't be split
# into atomic-Gaussian conditionals with linear maps), the prior `Q` is
# recovered via sparse-AD on `logp(x)` at `x = 0`:
#
#   Q = -H = -∇²logp(0),   μ = Q \ g   where g = ∇logp(0).
#
# The legacy path called `hessian(logp, prep, backend, x0)` per
# objective evaluation, which returns a fresh `SparseMatrixCSC{Float64}`
# each time, then ran `SparseMatrixCSC(...)` and `augment_pattern(...)`
# on top. We cache:
#
#   * `prep` (already cached by the legacy `Ref`);
#   * `H_buf::SparseMatrixCSC{Float64}` matching the DI prep pattern;
#   * `g_buf::Vector{Float64}`;
#   * `Q_union::SparseMatrixCSC{Float64}` with pattern `H_pattern ∪ lik_pattern`;
#   * `h_to_q_map[k] = index_in_Q_union_nzval(H_buf row/col at k)`.
#
# Per call, we run `value_gradient_and_hessian!(logp, g_buf, H_buf, prep,
# backend, x0)`, then `Q_union.nzval[h_to_q_map[k]] = -H_buf.nzval[k]`,
# then solve μ. No fresh sparse-matrix construction.
#
# Under outer AD over hp (Dual-valued `logp`), the cached Float64 buffers
# can't be reused; we fall back to allocating per-call buffers of the
# right eltype. The prep itself still serialises the pattern correctly
# in that case — `DI.hessian!` decompresses into an eltype-matched buffer.

using ADTypes: AutoSparse, AutoForwardDiff, AutoReverseDiff
using DifferentiationInterface
using DifferentiationInterface:
    prepare_hessian, prepare_gradient, value_gradient_and_hessian!
using SparseConnectivityTracer: TracerLocalSparsityDetector
using SparseMatrixColorings: GreedyColoringAlgorithm, sparsity_pattern
using DynamicPPL: getlogprior
using LogDensityProblems

"""
    CachedSparseADLatentModel

Latent prior for a DPPL model whose latent DAG can't be split into atomic
Gaussian conditionals with linear maps. Computes `(μ, Q)` via cached
sparse-AD over the log-prior at `x = 0`. Used by
`_build_joint_sparse_ad_latent` instead of the bare `FunctionLatentModel`
wrap.
"""
mutable struct CachedSparseADLatentModel{F, B, C <: Union{Nothing, Tuple{AbstractMatrix, AbstractVector}}} <: LatentModel
    make_logp::F            # (hp_nt) -> callable logp(x)
    hp_names::Tuple{Vararg{Symbol}}
    backend::B              # AutoSparse(SecondOrder(...))
    # `Any`-typed slots — the DI prep type is awkward to spell, and the
    # buffers are lazily allocated. Performance impact is negligible
    # (these are accessed once per call, fields used inside loops are
    # already locally bound).
    prep::Any
    n_latent::Int
    lik_pattern::Union{Nothing, SparseMatrixCSC}
    constraint::C
    H_buf::Any
    g_buf::Any
    Q_union::Any
    h_to_q_map::Any
end

function CachedSparseADLatentModel(
        make_logp, hp_names, backend, n_latent, lik_pattern;
        constraint = nothing,
    )
    return CachedSparseADLatentModel(
        make_logp, Tuple(hp_names), backend, nothing, n_latent,
        lik_pattern, constraint,
        nothing, nothing, nothing, nothing,
    )
end

Base.length(m::CachedSparseADLatentModel) = m.n_latent
hyperparameters(::CachedSparseADLatentModel) = NamedTuple()
constraints(m::CachedSparseADLatentModel; kwargs...) = m.constraint
model_name(::CachedSparseADLatentModel) = :cached_sparse_ad_latent

_hp_nt_from_kwargs(m::CachedSparseADLatentModel, kwargs) =
    NamedTuple{m.hp_names}(Tuple(kwargs[k] for k in m.hp_names))

# Resolve the linear position of `Q.nzval` that corresponds to (i, j).
# Returns 0 if (i, j) is not in the stored pattern.
function _csc_linear_index_phase2(Q::SparseMatrixCSC, i::Int, j::Int)
    col_range = Q.colptr[j]:(Q.colptr[j + 1] - 1)
    isempty(col_range) && return 0
    rowvals = view(Q.rowval, col_range)
    pos = searchsortedfirst(rowvals, i)
    pos > length(rowvals) && return 0
    rowvals[pos] == i || return 0
    return col_range[pos]
end

# Build a Q_union pattern from `H_pattern ∪ lik_pattern` and a map from
# H_pattern nzval indices to Q_union nzval indices.
function _build_h_to_q_map(H_pattern::SparseMatrixCSC, lik_pattern)
    n = size(H_pattern, 1)
    Q_struct = if lik_pattern === nothing
        SparseMatrixCSC{Float64, Int}(
            n, n, Vector{Int}(H_pattern.colptr), Vector{Int}(H_pattern.rowval),
            zeros(Float64, length(H_pattern.rowval)),
        )
    else
        lik_bool = SparseMatrixCSC{Float64, Int}(
            lik_pattern.m, lik_pattern.n, lik_pattern.colptr, lik_pattern.rowval,
            ones(length(lik_pattern.nzval)),
        )
        H_bool = SparseMatrixCSC{Float64, Int}(
            n, n, Vector{Int}(H_pattern.colptr), Vector{Int}(H_pattern.rowval),
            ones(length(H_pattern.rowval)),
        )
        # Build a canonical CSC structure with the union pattern; values
        # are placeholders.
        Q_pattern = sparse(H_bool .+ lik_bool .!= 0.0) .* 1.0
        Q_pattern
    end
    h_map = Vector{Int}(undef, length(H_pattern.rowval))
    for j in 1:size(H_pattern, 2)
        for k in H_pattern.colptr[j]:(H_pattern.colptr[j + 1] - 1)
            i_glob = H_pattern.rowval[k]
            lin = _csc_linear_index_phase2(Q_struct, i_glob, j)
            lin == 0 && error("Q_union pattern missing entry ($i_glob, $j)")
            h_map[k] = lin
        end
    end
    return Q_struct, h_map
end

# Initialise prep + buffers on first call (Float64 path).
function _ensure_buffers!(m::CachedSparseADLatentModel, logp)
    if m.prep === nothing
        x0 = zeros(Float64, m.n_latent)
        m.prep = prepare_hessian(logp, m.backend, x0)
    end
    if m.H_buf === nothing
        H_pat = sparsity_pattern(m.prep)
        # `sparsity_pattern` returns a Bool sparse matrix. Materialise as
        # Float64 with the same structure for in-place fill.
        H_pat_csc = SparseMatrixCSC{Float64, Int}(
            size(H_pat, 1), size(H_pat, 2),
            Vector{Int}(H_pat.colptr), Vector{Int}(H_pat.rowval),
            zeros(Float64, length(H_pat.rowval)),
        )
        m.H_buf = H_pat_csc
        m.g_buf = zeros(Float64, m.n_latent)
        Q_union, h_to_q_map = _build_h_to_q_map(H_pat_csc, m.lik_pattern)
        m.Q_union = Q_union
        m.h_to_q_map = h_to_q_map
    end
    return nothing
end

# Float64 hot path: fill cached buffers in place.
function _assemble_float64(m::CachedSparseADLatentModel, hp_nt::NamedTuple)
    logp = m.make_logp(hp_nt)
    _ensure_buffers!(m, logp)
    H_buf = m.H_buf
    g_buf = m.g_buf
    Q_union = m.Q_union
    h_to_q_map = m.h_to_q_map

    x0 = zeros(Float64, m.n_latent)
    fill!(H_buf.nzval, 0.0)
    fill!(g_buf, 0.0)
    value_gradient_and_hessian!(logp, g_buf, H_buf, m.prep, m.backend, x0)

    fill!(Q_union.nzval, 0.0)
    @inbounds for k in eachindex(H_buf.nzval)
        Q_union.nzval[h_to_q_map[k]] = -H_buf.nzval[k]
    end
    μ = Symmetric(Q_union) \ g_buf
    return μ, Q_union
end

# Dual / non-Float64 path: allocate per-call buffers with the right
# eltype. Re-uses the cached prep + pattern; doesn't pollute the cached
# Float64 buffers.
function _assemble_typed(m::CachedSparseADLatentModel, hp_nt::NamedTuple, ::Type{T}) where {T}
    logp = m.make_logp(hp_nt)
    # DI's preparation is type-strict: the cached prep is bound to the
    # Float64-typed `logp` closure + Float64 `x`. Under outer AD over hp,
    # `make_logp(Dual_hp_nt)` returns a different closure type → the
    # cached prep would fail `PreparationMismatch`. Build a fresh prep
    # against the typed inputs. (We could memoise by `typeof(logp)` for
    # repeated Dual calls, but outer-AD inner loops typically run a
    # handful of Hessians per θ — re-prep cost is acceptable.)
    x0 = zeros(T, m.n_latent)
    typed_prep = prepare_hessian(logp, m.backend, x0)
    H_pat = sparsity_pattern(typed_prep)
    H_buf = SparseMatrixCSC{T, Int}(
        size(H_pat, 1), size(H_pat, 2),
        Vector{Int}(H_pat.colptr), Vector{Int}(H_pat.rowval),
        zeros(T, length(H_pat.rowval)),
    )
    g_buf = zeros(T, m.n_latent)

    # Reuse the cached h_to_q_map if available — it's structural and
    # eltype-independent. Otherwise build one on the fly.
    if m.h_to_q_map === nothing
        Q_struct_f, h_map = _build_h_to_q_map(H_buf, m.lik_pattern)
        Q_union = SparseMatrixCSC{T, Int}(
            size(Q_struct_f, 1), size(Q_struct_f, 2),
            Vector{Int}(Q_struct_f.colptr), Vector{Int}(Q_struct_f.rowval),
            zeros(T, length(Q_struct_f.rowval)),
        )
    else
        h_map = m.h_to_q_map
        Q_cached = m.Q_union
        Q_union = SparseMatrixCSC{T, Int}(
            size(Q_cached, 1), size(Q_cached, 2),
            Vector{Int}(Q_cached.colptr), Vector{Int}(Q_cached.rowval),
            zeros(T, length(Q_cached.rowval)),
        )
    end

    value_gradient_and_hessian!(logp, g_buf, H_buf, typed_prep, m.backend, x0)

    @inbounds for k in eachindex(H_buf.nzval)
        Q_union.nzval[h_map[k]] = -H_buf.nzval[k]
    end
    μ = Symmetric(Q_union) \ g_buf
    return μ, Q_union
end

# Dispatch to Float64 hot path or per-call alloc path based on hp eltype.
function _assemble_sparse_ad(m::CachedSparseADLatentModel, hp_nt::NamedTuple)
    T = promote_type(map(typeof, values(hp_nt))..., Float64)
    if T === Float64
        return _assemble_float64(m, hp_nt)
    else
        return _assemble_typed(m, hp_nt, T)
    end
end

function Distributions.mean(m::CachedSparseADLatentModel; kwargs...)
    μ, _Q = _assemble_sparse_ad(m, _hp_nt_from_kwargs(m, kwargs))
    return μ
end

function precision_matrix(m::CachedSparseADLatentModel; kwargs...)
    _μ, Q = _assemble_sparse_ad(m, _hp_nt_from_kwargs(m, kwargs))
    return Q
end

function (m::CachedSparseADLatentModel)(; kwargs...)
    μ, Q = _assemble_sparse_ad(m, _hp_nt_from_kwargs(m, kwargs))
    gmrf = GMRF(μ, Q)
    return m.constraint === nothing ? gmrf :
        ConstrainedGMRF(gmrf, m.constraint[1], m.constraint[2])
end

function (m::CachedSparseADLatentModel)(ws::GMRFWorkspace; kwargs...)
    μ, Q = _assemble_sparse_ad(m, _hp_nt_from_kwargs(m, kwargs))
    Q_sparse = Q isa SparseMatrixCSC ? Q : sparse(Q)
    if eltype(Q_sparse) === eltype(ws.Q)
        update_precision_values!(ws, Q_sparse.nzval)
        return m.constraint === nothing ?
            WorkspaceGMRF(μ, Q_sparse, ws) :
            WorkspaceGMRF(μ, Q_sparse, ws, m.constraint[1], m.constraint[2])
    else
        gmrf = GMRF(μ, Q_sparse)
        return m.constraint === nothing ? gmrf :
            ConstrainedGMRF(gmrf, m.constraint[1], m.constraint[2])
    end
end
