# Build the latent prior from a DPPL model.
#
# Chooses between two extraction paths:
# - **DAG path**: valid when all random variables are atomic Gaussians with
#   linear cross-couplings. Builds (μ, Q) by block assembly. Fast, exact.
# - **Joint sparse-AD fallback**: for any model with loop-built components or
#   nonlinear dependencies. Runs sparse-AD on the prior log-density to
#   recover Q (Hessian) and μ (from Qμ = g at x=0). Slower setup, still fast
#   evaluation.
#
# Probe once at a sentinel θ to decide; the choice is θ-independent for
# Gaussian priors.

using SparseArrays, LinearAlgebra

# Union of two optional sparsity patterns. Either argument may be
# `nothing` (no pattern) or a `SparseMatrixCSC` whose nonzeros encode
# the pattern.
_union_patterns(::Nothing, ::Nothing) = nothing
_union_patterns(a::SparseMatrixCSC, ::Nothing) = a
_union_patterns(::Nothing, b::SparseMatrixCSC) = b
function _union_patterns(a::SparseMatrixCSC, b::SparseMatrixCSC)
    # Cast to Bool to get a structural union regardless of numeric values.
    return SparseMatrixCSC{Bool, Int}((a .!= 0) .| (b .!= 0))
end
using ADTypes: AutoSparse, AutoForwardDiff, AutoReverseDiff
using DifferentiationInterface
using DifferentiationInterface: SecondOrder
using SparseConnectivityTracer: TracerLocalSparsityDetector
using SparseMatrixColorings: GreedyColoringAlgorithm
using DynamicPPL: getlogprior
using LogDensityProblems
using GaussianMarkovRandomFields: ConstrainedGMRF
import ReverseDiff
import ForwardDiff

# Pull `(A, e)` off a per-sym prior distribution, else `nothing`. Only
# `ConstrainedGMRF` priors — produced by `IIDModel(n, :sumtozero)(τ=τ)` and
# friends — carry a constraint. Constraints are assumed hyperparameter-
# independent (sum-to-zero stays sum-to-zero across τ); we probe once at
# build time.
_dist_constraint(d::ConstrainedGMRF) = (d.constraint_matrix, d.constraint_vector)
_dist_constraint(_) = nothing

# Join per-sym constraints into a single `(A_joint, e_joint)` over the
# concatenated base-latent vector `[random_syms[1]; random_syms[2]; …]`.
# Syms without constraints contribute nothing; syms with `(A_s, e_s)` get
# their block horizontally embedded at the right offset, zero-padded on
# the other syms' coordinates. Returns `nothing` if no sym has a constraint.
function _extract_joint_constraint(dppl_model, random_syms::Tuple, dims, hp_values::NamedTuple)
    cond = DynamicPPL.fix(dppl_model, hp_values)
    priors = extract_priors(cond)
    per_sym = Dict{Symbol, Tuple{AbstractMatrix, AbstractVector}}()
    for s in random_syms
        d = find_dist(priors, s)
        isa(d, AbstractVector) && continue   # loop-built, no single distribution
        c = _dist_constraint(d)
        c === nothing || (per_sym[s] = c)
    end
    isempty(per_sym) && return nothing

    n_total = sum(dims[s] for s in random_syms)
    offsets = Dict{Symbol, Int}()
    off = 0
    for s in random_syms
        offsets[s] = off
        off += dims[s]
    end

    A_blocks = AbstractMatrix{Float64}[]
    e_blocks = AbstractVector{Float64}[]
    for s in random_syms
        haskey(per_sym, s) || continue
        A_s, e_s = per_sym[s]
        A_row = zeros(Float64, size(A_s, 1), n_total)
        A_row[:, (offsets[s] + 1):(offsets[s] + dims[s])] .= A_s
        push!(A_blocks, A_row)
        push!(e_blocks, Vector{Float64}(e_s))
    end
    A_joint = reduce(vcat, A_blocks)
    e_joint = reduce(vcat, e_blocks)
    return (A_joint, e_joint)
end

"""
    build_latent_model(dppl_model, random_syms::Tuple, hp_names::Tuple;
                       skip_pattern_augment = false,
                       extra_pattern = nothing)

Return `(FunctionLatentModel, path::Symbol)` where `path` is `:dag` or
`:sparse_ad`. The model evaluates `(μ, Q)` for any hyperparameter value.

Q's sparsity pattern needs to be a superset of the posterior Hessian's
so the `GMRFWorkspace`'s symbolic factorization can accept any runtime
numeric update. Two sources of pattern augmentation:

- `skip_pattern_augment = false` (default): detect the DPPL likelihood's
  Hessian pattern via sparse AD and union it into `Q`. Needed for the
  AD-based obs-model path; skipped when the caller will hand us a
  hand-coded obs model (fast path).
- `extra_pattern` (optional): explicit extra sparsity pattern to union
  into `Q`. Used by the fast path with `augment_latent = false`, where
  the adapter knows the design matrix `A` and can pass `A'A`'s pattern
  directly without re-running DPPL hessian detection.
"""
function build_latent_model(
        dppl_model, random_syms::Tuple, hp_names::Tuple;
        skip_pattern_augment::Bool = false,
        extra_pattern::Union{Nothing, SparseMatrixCSC} = nothing,
    )
    probe_hp = NamedTuple{hp_names}(Tuple(1.0 for _ in hp_names))

    info = analyze_structure(dppl_model, random_syms, probe_hp)
    all_atomic = all(info.classification[s] === :atomic_gaussian for s in random_syms)
    n_latent = sum(info.dims[s] for s in random_syms)

    detected_pattern = skip_pattern_augment ? nothing :
        detect_likelihood_pattern(dppl_model, hp_names, n_latent)
    lik_pattern = _union_patterns(detected_pattern, extra_pattern)

    joint_constraint = _extract_joint_constraint(dppl_model, random_syms, info.dims, probe_hp)

    if all_atomic
        linear_ok = true
        for child in random_syms, parent in info.edges[child]
            lm_probe = extract_linear_map(
                dppl_model, child, parent,
                random_syms, info.dims, probe_hp
            )
            lm_probe.linear || (linear_ok = false; break)
        end
        if linear_ok
            return _build_dag_latent(
                    dppl_model, random_syms, info, hp_names, lik_pattern, joint_constraint
                ), :dag
        end
    end

    return _build_joint_sparse_ad_latent(
            dppl_model, random_syms, n_latent, hp_names, lik_pattern, joint_constraint
        ), :sparse_ad
end

function _build_dag_latent(dppl_model, random_syms, info, hp_names, lik_pattern, joint_constraint)
    # Linear maps are *structural*: they encode the per-edge `child = A·parent + b`
    # relation in the model graph and do not depend on hyperparameter values.
    # Compute them once at adapter-build time with concrete Float64 inputs,
    # cache, and reuse on every assembly call. This is both a perf win
    # (no per-call sparse-AD probe) and unblocks outer AD over the latent
    # function — `extract_linear_map`'s SCT-based sparsity tracing collides
    # with `ForwardDiff.Dual` when called inside an outer AD pass.
    probe_hp = NamedTuple{hp_names}(Tuple(1.0 for _ in hp_names))
    linear_maps_cached = Dict{Tuple{Symbol, Symbol}, NamedTuple}()
    for child in random_syms, parent in info.edges[child]
        linear_maps_cached[(child, parent)] = extract_linear_map(
            dppl_model, child, parent, random_syms, info.dims, probe_hp;
            is_scalar = info.is_scalar,
        )
    end

    # Probe `cond_Qs` once at probe_hp to capture the structural pattern
    # of each per-variable conditional precision. The same probe is used
    # to seed the DAG assembly plan; the per-call closures below recompute
    # values at the actual hp values.
    cond_Qs_probe = Dict{Symbol, SparseMatrixCSC{Float64, Int}}()
    for s in random_syms
        Q_s, _int_s = atomic_conditional_and_intercept(
            dppl_model, s, random_syms, info.dims, probe_hp;
            is_scalar = info.is_scalar,
        )
        cond_Qs_probe[s] = SparseMatrixCSC{Float64, Int}(Q_s)
    end

    plan = build_dag_assembly_plan(
        random_syms, info.dims, info.edges, linear_maps_cached, cond_Qs_probe;
        lik_pattern = lik_pattern,
    )

    compute_cond_state = function (hp_nt::NamedTuple)
        # `Any`-valued dicts so Dual-parametrized Q and μ flow through under
        # outer AD (ForwardDiff through `hyperparameter_logpdf`). Pinning to
        # Float64 silently strips Duals and breaks `ADStrategy` on DPPL-built
        # LGMs.
        cond_Qs = Dict{Symbol, Any}()
        intercepts = Dict{Symbol, Any}()
        for s in random_syms
            Q_s, int_s = atomic_conditional_and_intercept(
                dppl_model, s, random_syms, info.dims, hp_nt;
                is_scalar = info.is_scalar,
            )
            cond_Qs[s] = Q_s
            intercepts[s] = int_s
        end
        return cond_Qs, intercepts
    end

    return CachedDAGLatentModel(
        plan, linear_maps_cached, compute_cond_state, hp_names;
        constraint = joint_constraint,
    )
end

function _build_joint_sparse_ad_latent(dppl_model, random_syms, n_latent, hp_names, lik_pattern, joint_constraint)
    sparse_backend = AutoSparse(
        SecondOrder(AutoForwardDiff(), AutoReverseDiff());
        sparsity_detector = TracerLocalSparsityDetector(),
        coloring_algorithm = GreedyColoringAlgorithm(),
    )

    # `make_logp` builds the conditioned log-prior function at given hp
    # values. Sparse-AD prep + buffers are cached on the model itself;
    # the per-call work is just the in-place fill via DI's `hessian!`.
    make_logp = function (hp_nt::NamedTuple)
        cond = DynamicPPL.fix(dppl_model, hp_nt)
        ldf = DynamicPPL.LogDensityFunction(cond, getlogprior)
        logp(x) = LogDensityProblems.logdensity(ldf, x)
        return logp
    end

    return CachedSparseADLatentModel(
        make_logp, hp_names, sparse_backend, n_latent, lik_pattern;
        constraint = joint_constraint,
    )
end
