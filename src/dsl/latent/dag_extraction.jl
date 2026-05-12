# DAG-based extraction of the joint (μ, Q) for a multi-random DPPL model.
#
# Strategy:
# 1. Analyse structure: classify each random variable, detect parent-child
#    edges by toggling each parent and seeing which child priors change.
# 2. For each parent→child edge, extract the linear map via sparse-AD
#    Jacobian of child's mean w.r.t. the parent.
# 3. Assemble the joint precision and mean by stacking conditional Gaussians
#    and applying the linear cross-couplings.
#
# Only valid when all random variables are atomic Gaussians with linear
# cross-couplings. The fallback path is joint sparse AD (latent_prior.jl).

using SparseArrays, LinearAlgebra
using ADTypes: AutoSparse, AutoForwardDiff
using DifferentiationInterface
using SparseConnectivityTracer: TracerLocalSparsityDetector
using SparseMatrixColorings: GreedyColoringAlgorithm
using Distributions: Normal, MvNormal, invcov
using GaussianMarkovRandomFields: precision_matrix

"""
    analyze_structure(dppl_model, random_syms::Tuple, hp_values::NamedTuple)

Returns `(dims, classification, edges)` describing the DAG structure of
`random_syms` in `dppl_model`:
- `dims`: dimension of each random variable
- `classification`: `:atomic_gaussian`, `:non_gaussian`, or `:loop_built`
- `edges[child]`: list of `parent` symbols on whom `child`'s prior depends
"""
function analyze_structure(dppl_model, random_syms::Tuple, hp_values::NamedTuple)
    dims = Dict(s => variable_length(dppl_model, s, hp_values) for s in random_syms)
    classification = Dict(s => classify_sym(dppl_model, s, hp_values) for s in random_syms)
    # Detect scalar (UnivariateDistribution) latents so we can seed probe
    # values as scalars, not 1-vectors. DPPL's body for `α ~ Normal(0,1)`
    # expects a scalar α; passing a 1-vector via `DynamicPPL.fix` makes the
    # body see a vector and crashes downstream operations like `α + ...`.
    is_scalar = Dict(s => _is_scalar_latent(dppl_model, s, hp_values) for s in random_syms)

    edges = Dict(s => Symbol[] for s in random_syms)
    for child in random_syms
        classification[child] === :atomic_gaussian || continue
        for parent in random_syms
            child === parent && continue
            others = Tuple(s for s in random_syms if s !== child)
            fix_0 = NamedTuple{others}(Tuple(_zero_seed(is_scalar[s], dims[s]) for s in others))
            fix_1 = merge(
                fix_0,
                NamedTuple{(parent,)}((_ones_seed(is_scalar[parent], dims[parent]),)),
            )
            p0 = extract_priors(DynamicPPL.fix(dppl_model, merge(hp_values, fix_0)))
            p1 = extract_priors(DynamicPPL.fix(dppl_model, merge(hp_values, fix_1)))
            if priors_differ(find_dist(p0, child), find_dist(p1, child))
                push!(edges[child], parent)
            end
        end
    end
    return (; dims, classification, edges, is_scalar)
end

# Helpers: scalar vs vector seeds for probe NamedTuples. DPPL needs scalars
# for univariate variables; seeding as a 1-vector breaks the model body.
_zero_seed(scalar::Bool, dim::Int) = scalar ? 0.0 : zeros(dim)
_ones_seed(scalar::Bool, dim::Int) = scalar ? 1.0 : ones(dim)

function _is_scalar_latent(dppl_model, sym::Symbol, hp_values::NamedTuple)
    cond = DynamicPPL.fix(dppl_model, hp_values)
    priors = extract_priors(cond)
    matches = [d for (vn, d) in pairs(priors) if getsym(vn) === sym]
    return length(matches) == 1 && matches[1] isa UnivariateDistribution
end

"""
    extract_linear_map(dppl_model, child, parent, random_syms, dims, hp_values)

Extract the linear map `A` in `E[child | parent] = A * parent + b`, plus the
intercept `b`. Uses sparse-AD Jacobian of child's mean function. Returns
`(A, b, linear)` where `linear` is whether the map is truly linear (Jacobian
constant in `parent`).
"""
function extract_linear_map(
        dppl_model, child::Symbol, parent::Symbol,
        random_syms, dims, hp_values::NamedTuple;
        is_scalar::Union{Nothing, Dict{Symbol, Bool}} = nothing,
    )
    p_dim = dims[parent]
    is_scalar_dict = is_scalar === nothing ?
        Dict(s => _is_scalar_latent(dppl_model, s, hp_values) for s in random_syms) :
        is_scalar
    child_zero = _zero_seed(is_scalar_dict[child], dims[child])
    others_rest = Tuple(s for s in random_syms if s !== parent && s !== child)

    function μ_of_parent(parent_val)
        # `prepare_jacobian` / `jacobian` always pass a vector. If the
        # parent variable is itself a scalar in the DPPL body, unwrap the
        # 1-element vector before placing it in the init NamedTuple.
        parent_init = is_scalar_dict[parent] ? parent_val[1] : parent_val
        init_nt = merge(
            NamedTuple{others_rest}(
                Tuple(_zero_seed(is_scalar_dict[s], dims[s]) for s in others_rest)
            ),
            NamedTuple{(parent,)}((parent_init,)),
            NamedTuple{(child,)}((child_zero,)),
        )
        cond = DynamicPPL.fix(dppl_model, hp_values)
        priors = extract_priors_no_sample(cond, init_nt)
        return Vector(mean(find_dist(priors, child)))
    end

    backend = AutoSparse(
        AutoForwardDiff();
        sparsity_detector = TracerLocalSparsityDetector(),
        coloring_algorithm = GreedyColoringAlgorithm(),
    )
    prep = prepare_jacobian(μ_of_parent, backend, zeros(p_dim))
    A = jacobian(μ_of_parent, prep, backend, zeros(p_dim))
    A_check = jacobian(μ_of_parent, prep, backend, ones(p_dim))
    b = μ_of_parent(zeros(p_dim))
    linear = isapprox(A, A_check; atol = 1.0e-10, rtol = 1.0e-10)
    A_sp = SparseMatrixCSC(A)
    dropzeros!(A_sp)
    return (A = A_sp, b = b, linear = linear)
end

"""
    conditional_precision(dist)

Precision matrix of an atomic Gaussian conditional distribution, as a sparse
matrix. Handles `Normal`, `MvNormal`, and `AbstractGMRF`.
"""
function conditional_precision(dist)
    if dist isa Normal
        return sparse([1 / dist.σ^2;;])
    elseif dist isa MvNormal
        return sparse(Matrix(invcov(dist)))
    else
        return sparse(Matrix(precision_matrix(dist)))
    end
end

"""
    atomic_conditional_and_intercept(dppl_model, sym, random_syms, dims, hp_values)

For atomic-Gaussian `sym`, return `(Q_cond, μ_cond)` at `hp_values` with all
other random variables fixed at zero. The conditional distribution is
`sym | others=0, θ=hp_values`.
"""
function atomic_conditional_and_intercept(
        dppl_model, sym::Symbol, random_syms, dims, hp_values::NamedTuple;
        is_scalar::Union{Nothing, Dict{Symbol, Bool}} = nothing,
    )
    is_scalar_dict = is_scalar === nothing ?
        Dict(s => _is_scalar_latent(dppl_model, s, hp_values) for s in random_syms) :
        is_scalar
    others = Tuple(s for s in random_syms if s !== sym)
    others_zero = NamedTuple{others}(
        Tuple(_zero_seed(is_scalar_dict[s], dims[s]) for s in others)
    )
    cond = DynamicPPL.fix(dppl_model, merge(hp_values, others_zero))
    # No-sample path: `extract_priors`'s default init runs the model via
    # sampling, which breaks when hp_values are Dual and the prior is a
    # ConstrainedGMRF (no Dual-typed `_rand!` downstream). Providing an
    # explicit init value for `sym` sidesteps that.
    sym_init = NamedTuple{(sym,)}((_zero_seed(is_scalar_dict[sym], dims[sym]),))
    priors = extract_priors_no_sample(cond, sym_init)
    d = find_dist(priors, sym)
    # `mean(d)` returns a scalar for `Normal(...)` and a vector for `MvNormal`.
    # The downstream `assemble_joint` expects vectors, so wrap scalars.
    μ = mean(d)
    μ_vec = μ isa AbstractVector ? Vector(μ) : [μ]
    return conditional_precision(d), μ_vec
end

"""
    topo_order(random_syms, edges)

Topological sort of `random_syms` consistent with the DAG `edges` (parents
before children). Used to propagate means during joint assembly.
"""
function topo_order(random_syms, edges)
    in_deg = Dict(s => length(edges[s]) for s in random_syms)
    queue = Symbol[s for s in random_syms if in_deg[s] == 0]
    order = Symbol[]
    while !isempty(queue)
        s = popfirst!(queue)
        push!(order, s)
        for child in random_syms
            if s in edges[child]
                in_deg[child] -= 1
                in_deg[child] == 0 && push!(queue, child)
            end
        end
    end
    return order
end

"""
    assemble_joint(random_syms, dims, edges, linear_maps, intercepts, cond_Qs)

Assemble the joint `(μ, Q)` of the multi-random latent field from per-variable
conditional precisions, per-edge linear maps, and per-variable intercepts.
Applies the Schur complement formula block by block.
"""
function assemble_joint(random_syms, dims, edges, linear_maps, intercepts, cond_Qs)
    offsets = Dict{Symbol, UnitRange{Int}}()
    off = 0
    for s in random_syms
        offsets[s] = (off + 1):(off + dims[s])
        off += dims[s]
    end
    n_total = off

    # Element type follows the conditional-precision pieces — allows `Dual`
    # to flow through when the assembly runs inside outer AD.
    T = promote_type(
        (eltype(cond_Qs[s]) for s in random_syms)...,
        (eltype(intercepts[s]) for s in random_syms)...,
    )

    Q = spzeros(T, n_total, n_total)
    for s in random_syms
        Q[offsets[s], offsets[s]] .+= cond_Qs[s]
    end
    for child in random_syms
        for parent in edges[child]
            A = linear_maps[(child, parent)].A
            Q_child = cond_Qs[child]
            idx_p = offsets[parent]
            idx_c = offsets[child]
            Q[idx_p, idx_p] .+= A' * Q_child * A
            Q[idx_p, idx_c] .-= A' * Q_child
            Q[idx_c, idx_p] .-= Q_child * A
        end
    end

    μ = zeros(T, n_total)
    for s in topo_order(random_syms, edges)
        μ[offsets[s]] .= intercepts[s]
        for parent in edges[s]
            A = linear_maps[(s, parent)].A
            μ[offsets[s]] .+= A * μ[offsets[parent]]
        end
    end
    return (μ = μ, Q = Q)
end
