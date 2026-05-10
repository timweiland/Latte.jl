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
    # Build the flat-vector layout (`vnt`) ONCE at adapter time using a
    # primal probe hp_nt. `LogDensityFunction(model, getlogdensity)` (3-arg
    # form) would otherwise rebuild it per call via `_default_vnt`, which
    # runs `init!!(model, ..., InitFromPrior(), ...)` — i.e. samples from
    # priors. That sampling triggers `_rand!` on `ConstrainedGMRF` (used by
    # `RWModel`, `BesagModel`, `IIDModel(...; constraint=:sumtozero)`,
    # etc.) which has no method for `Dual` eltype, blowing up under outer
    # AD. Caching the layout side-steps the issue cleanly.
    probe_hp = NamedTuple{hp_names}(Tuple(1.0 for _ in hp_names))
    cond_probe = DynamicPPL.fix(dppl_model, probe_hp)
    vnt_cached = DynamicPPL._default_vnt(cond_probe, DynamicPPL.UnlinkAll())

    # Detect scalar (univariate) latents so `pointwise_loglik` seeds them as
    # scalars, not 1-vectors — DPPL's body for `phase ~ Normal(0, π)` needs
    # scalar `phase`, mirroring the seeding done in
    # `try_exponential_family_fast_path` and DAG extraction.
    is_scalar = Dict(s => _is_scalar_latent(dppl_model, s, probe_hp) for s in random_syms)

    function loglik(x; kwargs...)
        hp_nt = NamedTuple{hp_names}(Tuple(kwargs[k] for k in hp_names))
        cond = DynamicPPL.fix(dppl_model, hp_nt)
        ldf = DynamicPPL.LogDensityFunction(
            cond, getloglikelihood, vnt_cached, DynamicPPL.ldf_accs(getloglikelihood),
        )
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
            Tuple(
                is_scalar[s] ? x[first(offsets[s])] : Vector(x[offsets[s]])
                    for s in random_syms
            )
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
    # Pin `grad_backend = AutoForwardDiff()`. The unified IFT path
    # (GMRFs #100/#101) handles hp-gradient flow correctly, but its
    # internal `loggrad(x_dual, lik)` call uses the AD likelihood's
    # stored `grad_backend`. Mooncake (the default) currently has a
    # broken `LinearSolveMooncakeExt.solve!_adjoint` for the
    # `SymTridiagonal{Float64} + LDLt` shapes the latent prior produces
    # downstream. ForwardDiff sidesteps that.
    #
    # `diagonal_hessian_safe = false`: this is the AD fallback path
    # (fast-path detection rejected the model), so the linear predictor
    # is generally non-trivial and the Hessian is NOT diagonal. The flag
    # makes `loghessian` go through the full `DI.hessian` route instead
    # of the per-element 1D shortcut. We keep `pointwise_loglik_func`
    # attached so WAIC / CPO accumulators have per-observation values to
    # consume — the upstream change in GMRFs #102/#107 decouples that
    # from the diagonal-Hessian assumption.
    return AutoDiffObservationModel(
        loglik; n_latent = n_latent, hyperparams = hp_names,
        grad_backend = AutoForwardDiff(),
        hessian_backend = hess_backend,
        pointwise_loglik_func = pointwise_loglik,
        diagonal_hessian_safe = false,
    )
end
