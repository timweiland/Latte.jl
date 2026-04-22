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
import ReverseDiff
import ForwardDiff

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
                    dppl_model, random_syms, info, hp_names, lik_pattern
                ), :dag
        end
    end

    return _build_joint_sparse_ad_latent(
            dppl_model, random_syms, n_latent, hp_names, lik_pattern
        ), :sparse_ad
end

function _build_dag_latent(dppl_model, random_syms, info, hp_names, lik_pattern)
    function latent_fn(; kwargs...)
        hp_values = NamedTuple{hp_names}(Tuple(kwargs[k] for k in hp_names))
        # `Any`-valued dicts so Dual-parametrized Q and μ flow through under
        # outer AD (ForwardDiff through `hyperparameter_logpdf`). Pinning to
        # Float64 here silently stripped Duals and broke `ADStrategy` on
        # DPPL-built LGMs.
        cond_Qs = Dict{Symbol, Any}()
        intercepts = Dict{Symbol, Any}()
        for s in random_syms
            Q_s, int_s = atomic_conditional_and_intercept(
                dppl_model, s, random_syms, info.dims, hp_values,
            )
            cond_Qs[s] = Q_s
            intercepts[s] = int_s
        end
        linear_maps = Dict{Tuple{Symbol, Symbol}, NamedTuple}()
        for child in random_syms, parent in info.edges[child]
            linear_maps[(child, parent)] = extract_linear_map(
                dppl_model, child, parent, random_syms, info.dims, hp_values,
            )
        end
        joint = assemble_joint(
            random_syms, info.dims, info.edges,
            linear_maps, intercepts, cond_Qs
        )
        Q_out = lik_pattern === nothing ? joint.Q : augment_pattern(joint.Q, lik_pattern)
        return (joint.μ, Q_out)
    end
    return FunctionLatentModel(latent_fn, sum(info.dims[s] for s in random_syms))
end

function _build_joint_sparse_ad_latent(dppl_model, random_syms, n_latent, hp_names, lik_pattern)
    sparse_backend = AutoSparse(
        SecondOrder(AutoForwardDiff(), AutoReverseDiff());
        sparsity_detector = TracerLocalSparsityDetector(),
        coloring_algorithm = GreedyColoringAlgorithm(),
    )
    grad_backend = AutoForwardDiff()
    hess_prep_ref = Ref{Any}(nothing)
    grad_prep_ref = Ref{Any}(nothing)

    function latent_fn(; kwargs...)
        hp_values = NamedTuple{hp_names}(Tuple(kwargs[k] for k in hp_names))
        cond = DynamicPPL.fix(dppl_model, hp_values)
        ldf = DynamicPPL.LogDensityFunction(cond, getlogprior)
        logp(x) = LogDensityProblems.logdensity(ldf, x)

        x0 = zeros(n_latent)
        if hess_prep_ref[] === nothing
            hess_prep_ref[] = prepare_hessian(logp, sparse_backend, x0)
            grad_prep_ref[] = prepare_gradient(logp, grad_backend, x0)
        end
        H = hessian(logp, hess_prep_ref[], sparse_backend, x0)
        g = gradient(logp, grad_prep_ref[], grad_backend, x0)
        Q = -H
        μ = Symmetric(Q) \ g
        Q_out = lik_pattern === nothing ?
            SparseMatrixCSC(Q) : augment_pattern(SparseMatrixCSC(Q), lik_pattern)
        return (μ, Q_out)
    end
    return FunctionLatentModel(latent_fn, n_latent)
end
