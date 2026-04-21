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
                       skip_pattern_augment = false)

Return `(FunctionLatentModel, path::Symbol)` where `path` is `:dag` or
`:sparse_ad`. The model evaluates `(μ, Q)` for any hyperparameter value.

With `skip_pattern_augment = true`, skip unioning the likelihood Hessian
pattern into `Q`. The pattern union is needed for the AD-based observation
model path (whose workspace requires the prior Q's sparsity to be a
superset of the posterior Hessian's), but unnecessary — and numerically
suspect — when the observation model is an `ExponentialFamily` wrapped in
`LinearlyTransformedObservationModel`, because the LGM's auto-augmentation
handles the x ↔ η coupling via a separate design-matrix block.
"""
function build_latent_model(
        dppl_model, random_syms::Tuple, hp_names::Tuple;
        skip_pattern_augment::Bool = false,
    )
    probe_hp = NamedTuple{hp_names}(Tuple(1.0 for _ in hp_names))

    info = analyze_structure(dppl_model, random_syms, probe_hp)
    all_atomic = all(info.classification[s] === :atomic_gaussian for s in random_syms)
    n_latent = sum(info.dims[s] for s in random_syms)

    # Likelihood Hessian pattern — unioned into Q for the AD obs-model path
    # so the workspace's symbolic factorization accepts any posterior Hessian
    # at runtime. Skipped for the fast path (see kwarg docstring).
    lik_pattern = skip_pattern_augment ? nothing :
        detect_likelihood_pattern(dppl_model, hp_names, n_latent)

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
        cond_Qs = Dict{Symbol, SparseMatrixCSC{Float64, Int}}()
        intercepts = Dict{Symbol, Vector{Float64}}()
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
