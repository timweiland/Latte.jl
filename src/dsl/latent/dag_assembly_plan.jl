# Build-time-cached assembly plan for the joint precision matrix.
#
# Replaces the per-call CSC `setindex!` storm in `assemble_joint` with
# a structural plan (computed once) + a runtime value-fill loop that
# writes into a pre-allocated `nzval` buffer using precomputed linear
# indices. The structural plan is hp-independent: it depends only on
# the DAG topology, the per-edge linear-map sparsity (cached upstream),
# and the per-variable conditional precision sparsity (probed at LGM
# build time and asserted invariant on the first runtime call).
#
# Caller pattern:
#
#   plan = build_dag_assembly_plan(random_syms, dims, edges,
#                                  linear_maps, cond_Qs_probe;
#                                  lik_pattern = lik_pattern)
#   nzval = zeros(T, plan.nnz)
#   assemble_values!(nzval, plan, cond_Qs, linear_maps)
#   μ = assemble_mean(plan, intercepts, linear_maps)
#   Q = SparseMatrixCSC(plan.n_total, plan.n_total,
#                       plan.colptr, plan.rowval, nzval)
#
# The plan's final `(colptr, rowval)` is already a superset of
# `lik_pattern`, so the runtime `augment_pattern` step is unnecessary.

using SparseArrays
using LinearAlgebra

"""
    DAGAssemblyPlan

Hp-independent structural plan for assembling the joint precision matrix
`Q_out` of a multi-random DPPL model. Built once at LGM construction
time. Per-call assembly fills values into `nzval` buffers using the
cached linear-index maps; no CSC `setindex!`, no fresh allocation
beyond the value buffer itself.

Fields:
- `n_total`: total latent dimension `Σₛ dims[s]`.
- `offsets::Vector{Tuple{Symbol, UnitRange{Int}}}`: latent-vector offsets
  per random sym, in the order they appear in `random_syms` (stable).
- `topo_order::Vector{Symbol}`: parents-before-children ordering used by
  `assemble_mean`.
- `colptr::Vector{Int}`, `rowval::Vector{Int}`, `nnz::Int`: canonical CSC
  structure of the joint `Q_out` (∪ `lik_pattern`).
- `diag_block_nzidx::Dict{Symbol, Vector{Int}}`: `diag_block_nzidx[s][k]`
  is the index in `nzval` to which `cond_Qs[s].nzval[k]` should be added.
- `edge_pp_nzidx`, `edge_pc_nzidx`, `edge_cp_nzidx`:
  for each parent→child edge `(child, parent)`, the index lists for the
  three contributions:
   - `pp`: `A' Q_child A` into the parent block (+= A'QA)
   - `pc`: `A' Q_child`   into the parent→child off-diag (-= A'Q)
   - `cp`: `Q_child A`    into the child→parent off-diag (-= QA)
- `diag_block_patterns::Dict{Symbol, Tuple{Vector{Int}, Vector{Int}}}`:
  cached `(colptr, rowval)` of each `cond_Qs[s]` at probe time. Used by
  the optional pattern-invariance check.
"""
struct DAGAssemblyPlan
    n_total::Int
    offsets::Vector{Tuple{Symbol, UnitRange{Int}}}
    topo_order::Vector{Symbol}
    colptr::Vector{Int}
    rowval::Vector{Int}
    nnz::Int
    diag_block_nzidx::Dict{Symbol, Vector{Int}}
    edge_pp_nzidx::Dict{Tuple{Symbol, Symbol}, Vector{Int}}
    edge_pc_nzidx::Dict{Tuple{Symbol, Symbol}, Vector{Int}}
    edge_cp_nzidx::Dict{Tuple{Symbol, Symbol}, Vector{Int}}
    diag_block_patterns::Dict{Symbol, Tuple{Vector{Int}, Vector{Int}}}
end

# Resolve the linear position of `Q.nzval` that corresponds to (i, j).
# Returns 0 if (i, j) is not in the stored pattern (caller's bug; we
# assert non-zero return at runtime).
function _csc_linear_index(Q::SparseMatrixCSC, i::Int, j::Int)
    col_range = Q.colptr[j]:(Q.colptr[j + 1] - 1)
    isempty(col_range) && return 0
    rowvals = view(Q.rowval, col_range)
    pos = searchsortedfirst(rowvals, i)
    pos > length(rowvals) && return 0
    rowvals[pos] == i || return 0
    return col_range[pos]
end

# Walk a `SparseMatrixCSC` and emit `(i_local, j_local)` pairs for each
# structurally-stored entry. Stored elements are in CSC canonical order:
# columns 1..n, row indices ascending within each column.
function _csc_pairs(M::SparseMatrixCSC)
    pairs = Tuple{Int, Int}[]
    sizehint!(pairs, nnz(M))
    for j in 1:size(M, 2)
        for k in M.colptr[j]:(M.colptr[j + 1] - 1)
            push!(pairs, (M.rowval[k], j))
        end
    end
    return pairs
end

"""
    build_dag_assembly_plan(random_syms, dims, edges, linear_maps, cond_Qs_probe;
                            lik_pattern = nothing) -> DAGAssemblyPlan

Build the structural plan from the per-variable conditional precisions
(probed once at LGM construction), the cached per-edge linear maps, and
an optional likelihood-Hessian sparsity pattern to union into `Q_out`.
"""
function build_dag_assembly_plan(
        random_syms, dims, edges, linear_maps,
        cond_Qs_probe::Dict;
        lik_pattern::Union{Nothing, SparseMatrixCSC} = nothing,
    )
    offsets_dict = Dict{Symbol, UnitRange{Int}}()
    offsets_vec = Tuple{Symbol, UnitRange{Int}}[]
    off = 0
    for s in random_syms
        rng = (off + 1):(off + dims[s])
        offsets_dict[s] = rng
        push!(offsets_vec, (s, rng))
        off += dims[s]
    end
    n_total = off

    # ─── Pass 1: collect all (i, j) contribution sites + the M_xx ─────────
    # matrices we'll need (for indexing later).
    triplet_I = Int[]
    triplet_J = Int[]
    diag_pairs = Dict{Symbol, Vector{Tuple{Int, Int}}}()
    edge_pp_pairs = Dict{Tuple{Symbol, Symbol}, Vector{Tuple{Int, Int}}}()
    edge_pc_pairs = Dict{Tuple{Symbol, Symbol}, Vector{Tuple{Int, Int}}}()
    edge_cp_pairs = Dict{Tuple{Symbol, Symbol}, Vector{Tuple{Int, Int}}}()

    # Diagonal blocks.
    for s in random_syms
        Qs = cond_Qs_probe[s]
        off_s = first(offsets_dict[s]) - 1
        local_pairs = _csc_pairs(Qs)
        diag_pairs[s] = local_pairs
        for (i_loc, j_loc) in local_pairs
            push!(triplet_I, off_s + i_loc)
            push!(triplet_J, off_s + j_loc)
        end
    end

    # Edge contributions.
    for child in random_syms
        for parent in edges[child]
            A = linear_maps[(child, parent)].A
            Q_child = cond_Qs_probe[child]
            off_p = first(offsets_dict[parent]) - 1
            off_c = first(offsets_dict[child]) - 1

            M_pp = A' * Q_child * A
            M_pc = A' * Q_child
            M_cp = Q_child * A

            pp_pairs = _csc_pairs(M_pp)
            pc_pairs = _csc_pairs(M_pc)
            cp_pairs = _csc_pairs(M_cp)
            edge_pp_pairs[(child, parent)] = pp_pairs
            edge_pc_pairs[(child, parent)] = pc_pairs
            edge_cp_pairs[(child, parent)] = cp_pairs

            for (i_loc, j_loc) in pp_pairs
                push!(triplet_I, off_p + i_loc)
                push!(triplet_J, off_p + j_loc)
            end
            for (i_loc, j_loc) in pc_pairs
                push!(triplet_I, off_p + i_loc)
                push!(triplet_J, off_c + j_loc)
            end
            for (i_loc, j_loc) in cp_pairs
                push!(triplet_I, off_c + i_loc)
                push!(triplet_J, off_p + j_loc)
            end
        end
    end

    # ─── Pass 2: build the canonical Q_out pattern (∪ lik_pattern) ────────
    Q_struct = sparse(triplet_I, triplet_J, ones(length(triplet_I)), n_total, n_total)
    Q_canonical = if lik_pattern === nothing
        Q_struct
    else
        # Boolean union of patterns; values irrelevant.
        lik_bool = SparseMatrixCSC{Float64, Int}(
            lik_pattern.m, lik_pattern.n, lik_pattern.colptr, lik_pattern.rowval,
            ones(length(lik_pattern.nzval)),
        )
        sparse(Q_struct .+ lik_bool .!= 0.0) .* 1.0
    end
    # Ensure all entries are stored (no dropzeros — we want the union pattern
    # preserved even if a sum happens to be zero).
    # `sparse(...)` above already canonicalises with ascending rowval per col.

    colptr = Vector{Int}(Q_canonical.colptr)
    rowval = Vector{Int}(Q_canonical.rowval)
    nnz_total = length(rowval)

    # ─── Pass 3: resolve linear nzval indices for each contribution ───────
    diag_block_nzidx = Dict{Symbol, Vector{Int}}()
    for s in random_syms
        off_s = first(offsets_dict[s]) - 1
        idxs = Int[]
        sizehint!(idxs, length(diag_pairs[s]))
        for (i_loc, j_loc) in diag_pairs[s]
            i_glob = off_s + i_loc
            j_glob = off_s + j_loc
            lin = _csc_linear_index(Q_canonical, i_glob, j_glob)
            lin == 0 && error(
                "build_dag_assembly_plan: missing pattern entry for diag-block " *
                    "($s, $i_glob, $j_glob)"
            )
            push!(idxs, lin)
        end
        diag_block_nzidx[s] = idxs
    end

    edge_pp_nzidx = Dict{Tuple{Symbol, Symbol}, Vector{Int}}()
    edge_pc_nzidx = Dict{Tuple{Symbol, Symbol}, Vector{Int}}()
    edge_cp_nzidx = Dict{Tuple{Symbol, Symbol}, Vector{Int}}()
    for child in random_syms
        for parent in edges[child]
            off_p = first(offsets_dict[parent]) - 1
            off_c = first(offsets_dict[child]) - 1

            pp_pairs = edge_pp_pairs[(child, parent)]
            pp_idxs = Int[]
            sizehint!(pp_idxs, length(pp_pairs))
            for (i_loc, j_loc) in pp_pairs
                lin = _csc_linear_index(Q_canonical, off_p + i_loc, off_p + j_loc)
                lin == 0 && error("missing pattern entry for edge ($child, $parent) pp")
                push!(pp_idxs, lin)
            end
            edge_pp_nzidx[(child, parent)] = pp_idxs

            pc_pairs = edge_pc_pairs[(child, parent)]
            pc_idxs = Int[]
            sizehint!(pc_idxs, length(pc_pairs))
            for (i_loc, j_loc) in pc_pairs
                lin = _csc_linear_index(Q_canonical, off_p + i_loc, off_c + j_loc)
                lin == 0 && error("missing pattern entry for edge ($child, $parent) pc")
                push!(pc_idxs, lin)
            end
            edge_pc_nzidx[(child, parent)] = pc_idxs

            cp_pairs = edge_cp_pairs[(child, parent)]
            cp_idxs = Int[]
            sizehint!(cp_idxs, length(cp_pairs))
            for (i_loc, j_loc) in cp_pairs
                lin = _csc_linear_index(Q_canonical, off_c + i_loc, off_p + j_loc)
                lin == 0 && error("missing pattern entry for edge ($child, $parent) cp")
                push!(cp_idxs, lin)
            end
            edge_cp_nzidx[(child, parent)] = cp_idxs
        end
    end

    diag_block_patterns = Dict{Symbol, Tuple{Vector{Int}, Vector{Int}}}()
    for s in random_syms
        Qs = cond_Qs_probe[s]
        diag_block_patterns[s] = (Vector{Int}(Qs.colptr), Vector{Int}(Qs.rowval))
    end

    return DAGAssemblyPlan(
        n_total, offsets_vec, topo_order(random_syms, edges),
        colptr, rowval, nnz_total,
        diag_block_nzidx,
        edge_pp_nzidx, edge_pc_nzidx, edge_cp_nzidx,
        diag_block_patterns,
    )
end

"""
    assemble_values!(nzval, plan, cond_Qs, linear_maps;
                     check_pattern_invariance = false)

Fill `nzval` (length `plan.nnz`) with the joint precision values from the
per-variable conditional precisions and the cached linear maps. `nzval`
must be zero-initialised by the caller.

If `check_pattern_invariance = true`, verify that the runtime `cond_Qs`
have the same `colptr`/`rowval` as the patterns probed at plan-build
time. Asserts on mismatch — useful as a one-shot sanity check on the
first call, expensive to leave on in the hot loop.
"""
function assemble_values!(
        nzval::AbstractVector, plan::DAGAssemblyPlan,
        cond_Qs::AbstractDict, linear_maps::AbstractDict;
        check_pattern_invariance::Bool = false,
    )
    length(nzval) == plan.nnz ||
        throw(ArgumentError("nzval length $(length(nzval)) ≠ plan.nnz $(plan.nnz)"))

    # Pattern-invariance check.
    if check_pattern_invariance
        for (s, (probe_colptr, probe_rowval)) in plan.diag_block_patterns
            Qs = cond_Qs[s]
            (Vector{Int}(Qs.colptr) == probe_colptr && Vector{Int}(Qs.rowval) == probe_rowval) ||
                throw(
                ErrorException(
                    "DAGAssemblyPlan: cond_Qs[:$s] sparsity pattern changed at runtime " *
                        "(plan was built against a different structure). Pattern-keyed " *
                        "assembly relies on invariance across hp values."
                )
            )
        end
    end

    # Diagonal blocks.
    for (s, _rng) in plan.offsets
        Qs = cond_Qs[s]
        idxs = plan.diag_block_nzidx[s]
        nz_local = Qs.nzval
        @inbounds for k in eachindex(nz_local)
            nzval[idxs[k]] += nz_local[k]
        end
    end

    # Edge contributions.
    for ((child, parent), pp_idxs) in plan.edge_pp_nzidx
        A = linear_maps[(child, parent)].A
        Q_child = cond_Qs[child]

        # Sparse matrix products: still allocate fresh sparse matrices,
        # but the dominant savings come from avoiding CSC setindex! on
        # the joint Q. Optimising the products themselves is a possible
        # follow-up (Phase 2 of the perf doc).
        M_pp = A' * Q_child * A
        M_pc = A' * Q_child
        M_cp = Q_child * A

        @inbounds for k in eachindex(M_pp.nzval)
            nzval[pp_idxs[k]] += M_pp.nzval[k]
        end
        pc_idxs = plan.edge_pc_nzidx[(child, parent)]
        @inbounds for k in eachindex(M_pc.nzval)
            nzval[pc_idxs[k]] -= M_pc.nzval[k]
        end
        cp_idxs = plan.edge_cp_nzidx[(child, parent)]
        @inbounds for k in eachindex(M_cp.nzval)
            nzval[cp_idxs[k]] -= M_cp.nzval[k]
        end
    end
    return nzval
end

"""
    assemble_mean(plan, intercepts, linear_maps) -> Vector{T}

Forward-substitute the per-variable intercepts through the DAG to
produce the joint mean. `T` is the promoted intercept eltype.
"""
function assemble_mean(plan::DAGAssemblyPlan, intercepts::AbstractDict, linear_maps::AbstractDict)
    T = promote_type(map(eltype, values(intercepts))...)
    μ = zeros(T, plan.n_total)
    offsets_dict = Dict(plan.offsets)

    # Find the edges per child (we only have the global topo_order; we
    # don't carry `edges` explicitly through the plan, but the cached
    # `linear_maps` keys give us `(child, parent)` pairs).
    child_to_parents = Dict{Symbol, Vector{Symbol}}()
    for (child, _rng) in plan.offsets
        child_to_parents[child] = Symbol[]
    end
    for (key, _lm) in linear_maps
        child, parent = key
        push!(child_to_parents[child], parent)
    end

    for s in plan.topo_order
        rng = offsets_dict[s]
        μ[rng] .= intercepts[s]
        for parent in child_to_parents[s]
            A = linear_maps[(s, parent)].A
            μ[rng] .+= A * μ[offsets_dict[parent]]
        end
    end
    return μ
end

# ─── CachedDAGLatentModel ─────────────────────────────────────────────────────
# Sibling of `FunctionLatentModel` for the DAG-extracted latent prior path.
# Owns a `DAGAssemblyPlan` so per-call `(μ, Q)` assembly fills a
# pre-allocated `nzval` buffer instead of rebuilding the joint via CSC
# `setindex!`. Used by `_build_dag_latent`.

import Distributions
using GaussianMarkovRandomFields:
    GMRFWorkspace, WorkspaceGMRF, GMRF, ConstrainedGMRF,
    update_precision_values!, hyperparameters, precision_matrix, constraints, model_name

"""
    CachedDAGLatentModel

Latent prior for a multi-random DPPL model with all-atomic-Gaussian
random variables connected by linear maps. Holds a `DAGAssemblyPlan`
plus the runtime callable closures (`compute_cond_Qs`,
`compute_intercepts`) that produce the per-variable conditional
precisions / intercepts at a given hp value.

Per-call cost is `O(nnz)` for value-fill into a `Vector{T}` buffer plus
the sparse matrix products `A' Q_child A` etc. across edges. The CSC
`setindex!` storm of the old `assemble_joint` path is eliminated.
"""
struct CachedDAGLatentModel{F, C <: Union{Nothing, Tuple{AbstractMatrix, AbstractVector}}} <: LatentModel
    plan::DAGAssemblyPlan
    linear_maps::Dict{Tuple{Symbol, Symbol}, NamedTuple}
    # Single closure returning `(cond_Qs, intercepts)` at the given hp
    # values. One closure (not two) so per-call DPPL probing isn't
    # duplicated.
    compute_cond_state::F
    hp_names::Tuple{Vararg{Symbol}}
    n_latent::Int
    constraint::C
end

function CachedDAGLatentModel(
        plan, linear_maps, compute_cond_state, hp_names;
        constraint = nothing,
    )
    return CachedDAGLatentModel(
        plan, linear_maps, compute_cond_state,
        Tuple(hp_names), plan.n_total, constraint,
    )
end

Base.length(m::CachedDAGLatentModel) = m.n_latent
hyperparameters(::CachedDAGLatentModel) = NamedTuple()
constraints(m::CachedDAGLatentModel; kwargs...) = m.constraint
model_name(::CachedDAGLatentModel) = :cached_dag_latent

# Build (μ, Q) at the given hp values. Eltype follows the promoted
# eltype of the conditional precisions / intercepts — supports Dual
# under outer AD over hp.
function _assemble_at(m::CachedDAGLatentModel, hp_nt::NamedTuple)
    cond_Qs, intercepts = m.compute_cond_state(hp_nt)
    T = promote_type(
        map(v -> eltype(v), values(cond_Qs))...,
        map(v -> eltype(v), values(intercepts))...,
    )
    nzval = zeros(T, m.plan.nnz)
    assemble_values!(nzval, m.plan, cond_Qs, m.linear_maps)
    Q = SparseMatrixCSC(m.plan.n_total, m.plan.n_total, m.plan.colptr, m.plan.rowval, nzval)
    μ = assemble_mean(m.plan, intercepts, m.linear_maps)
    return μ, Q
end

# Extract the hp NamedTuple in canonical order from kwargs.
_hp_nt_from_kwargs(m::CachedDAGLatentModel, kwargs) =
    NamedTuple{m.hp_names}(Tuple(kwargs[k] for k in m.hp_names))

function Distributions.mean(m::CachedDAGLatentModel; kwargs...)
    μ, _Q = _assemble_at(m, _hp_nt_from_kwargs(m, kwargs))
    return μ
end

function precision_matrix(m::CachedDAGLatentModel; kwargs...)
    _μ, Q = _assemble_at(m, _hp_nt_from_kwargs(m, kwargs))
    return Q
end

# Cold path: produce a free-standing GMRF (or ConstrainedGMRF). Allocates
# a SparseMatrixCSC backed by the cached structure + a fresh nzval.
function (m::CachedDAGLatentModel)(; kwargs...)
    μ, Q = _assemble_at(m, _hp_nt_from_kwargs(m, kwargs))
    gmrf = GMRF(μ, Q)
    return m.constraint === nothing ? gmrf :
        ConstrainedGMRF(gmrf, m.constraint[1], m.constraint[2])
end

# Warm path: caller supplies a workspace with a pre-factorised symbolic
# Cholesky. When the workspace eltype matches the assembled eltype we
# can fill `ws.Q.nzval` directly via `update_precision_values!`. Type-
# mismatch (e.g. Float64 workspace + Dual under outer AD) falls back to
# building a fresh SparseMatrixCSC{T} — the workspace's symbolic factor
# is useless for that AD pass anyway.
function (m::CachedDAGLatentModel)(ws::GMRFWorkspace; kwargs...)
    μ, Q = _assemble_at(m, _hp_nt_from_kwargs(m, kwargs))
    Q_sparse = Q isa SparseMatrixCSC ? Q : sparse(Q)
    if eltype(Q_sparse) === eltype(ws.Q)
        update_precision_values!(ws, Q_sparse.nzval)
        return m.constraint === nothing ?
            WorkspaceGMRF(μ, Q_sparse, ws) :
            WorkspaceGMRF(μ, Q_sparse, ws, m.constraint[1], m.constraint[2])
    else
        # Eltype mismatch — fall back to cold-path semantics.
        gmrf = GMRF(μ, Q_sparse)
        return m.constraint === nothing ? gmrf :
            ConstrainedGMRF(gmrf, m.constraint[1], m.constraint[2])
    end
end
