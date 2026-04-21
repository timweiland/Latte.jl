# Extract an observation model from a DPPL model by wrapping its likelihood
# as an `AutoDiffObservationModel`. Uses DPPL's accumulator split to isolate
# the likelihood from the prior.

using ADTypes: AutoSparse, AutoForwardDiff
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
        hp_names::Tuple
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
        return collect(values(pointwise))
    end

    hess_backend = AutoSparse(
        AutoForwardDiff();
        sparsity_detector = TracerLocalSparsityDetector(),
        coloring_algorithm = GreedyColoringAlgorithm(),
    )
    return AutoDiffObservationModel(
        loglik; n_latent = n_latent, hyperparams = hp_names,
        hessian_backend = hess_backend,
        pointwise_loglik_func = pointwise_loglik,
    )
end
