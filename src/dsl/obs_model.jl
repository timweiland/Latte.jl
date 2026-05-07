# Extract an observation model from a DPPL model by wrapping its likelihood
# as an `AutoDiffObservationModel`. Uses DPPL's accumulator split to isolate
# the likelihood from the prior.

using ADTypes: AutoSparse, AutoForwardDiff, KnownHessianSparsityDetector
using SparseConnectivityTracer: TracerLocalSparsityDetector
using SparseMatrixColorings: GreedyColoringAlgorithm
using GaussianMarkovRandomFields: AutoDiffObservationModel

"""
    extract_obs_model(dppl_model, n_latent, random_syms, dims; hp_names)

Return an `AutoDiffObservationModel` wrapping the DPPL model's likelihood
(everything except the prior `~` statements for hyperparameters and random
variables). Supplies both the scalar loglik and a pointwise variant for
WAIC / CPO accumulators.
"""
function extract_obs_model(
        dppl_model, n_latent::Int, random_syms, dims;
        hp_names::Tuple,
        hessian_pattern::Union{Nothing, SparseMatrixCSC} = nothing,
    )
    function loglik(x; kwargs...)
        hp_nt = NamedTuple{hp_names}(Tuple(kwargs[k] for k in hp_names))
        cond = DynamicPPL.fix(dppl_model, hp_nt)
        ldf = DynamicPPL.LogDensityFunction(cond, getloglikelihood)
        return LogDensityProblems.logdensity(ldf, x)
    end

    # Pointwise likelihood: split x into named components, use DPPL's
    # `pointwise_loglikelihoods` to grab per-site log-likelihoods.
    offsets = Dict{Symbol, UnitRange{Int}}()
    off = 0
    for s in random_syms
        offsets[s] = (off + 1):(off + dims[s])
        off += dims[s]
    end

    function pointwise_loglik(x; kwargs...)
        hp_nt = NamedTuple{hp_names}(Tuple(kwargs[k] for k in hp_names))
        rand_nt = NamedTuple{Tuple(random_syms)}(
            Tuple(Vector(x[offsets[s]]) for s in random_syms)
        )
        cond = DynamicPPL.fix(dppl_model, hp_nt)
        # Build a VarInfo seeded with `rand_nt`; then ask DPPL for per-site
        # likelihoods. (DPPL's `pointwise_loglikelihoods` requires an
        # AbstractVarInfo, not an init strategy directly.)
        vi = DynamicPPL.VarInfo(cond, InitFromParams(rand_nt, nothing))
        pointwise = DynamicPPL.pointwise_loglikelihoods(cond, vi)
        # `collect(values(::Dict))` gives `Vector{Any}` here because DPPL's
        # pointwise dict isn't statically typed on its value type. Tighten
        # the eltype by promoting from the actual contents — without this,
        # downstream broadcast operations (e.g. inside the FD ext's
        # `_forwarddiff_workspace_ga_obs_dual` when hyperparameters are
        # Dual) fall back to `Vector{Any}` and trip `zero(::Type{Any})`.
        vec = collect(values(pointwise))
        isempty(vec) && return Float64[]
        T = mapreduce(typeof, promote_type, vec)
        return convert(Vector{T}, vec)
    end

    # When the caller has pre-supplied a Hessian sparsity pattern (e.g. for
    # black-box likelihoods where tracer-based detection can't flow, like
    # ODE solvers), bake it into the sparse-AD backend via
    # `KnownHessianSparsityDetector`. Otherwise default to the local
    # tracer, which works for standard DPPL models.
    sparsity_detector = hessian_pattern === nothing ?
        TracerLocalSparsityDetector() :
        KnownHessianSparsityDetector(hessian_pattern)
    hess_backend = AutoSparse(
        AutoForwardDiff();
        sparsity_detector = sparsity_detector,
        coloring_algorithm = GreedyColoringAlgorithm(),
    )
    return AutoDiffObservationModel(
        loglik; n_latent = n_latent, hyperparams = hp_names,
        hessian_backend = hess_backend,
        pointwise_loglik_func = pointwise_loglik,
    )
end
