# DPPL adapter — composite observation model construction.
#
# Splits a DPPL `@model`'s observed `~` blocks into named groups, building
# one `AutoDiffObservationModel` per group, wrapping them in a
# `CompositeObservationModel` with explicit identity kwarg routes. Lets the
# user expose distinct hyperparameters per channel (e.g. `σ_phys` vs
# `σ_data` for a PDE-inverse problem) without folding everything into one
# AD likelihood.
#
# v1 limitations (deliberate):
# - Each component declares the FULL hp tuple. The DPPL model body still
#   evaluates RHS distributions for every `~` block regardless of which
#   group is being summed, so a Dual-tagged hyperparameter can flow through
#   non-group code paths. Restricting per-component hyperparams won't avoid
#   that — an explicit submodel API would.
# - Per-group `hessian_pattern` overrides are not exposed yet; the global
#   pattern (used by `build_latent_model` for the prior Q) is still the
#   union of every component's posterior Hessian sparsity.

using Distributions: loglikelihood
using DynamicPPL: @model
import DynamicPPL
using DynamicPPL.AbstractPPL: getsym
using ADTypes: AutoSparse, AutoForwardDiff, KnownHessianSparsityDetector
using SparseConnectivityTracer: TracerLocalSparsityDetector
using SparseMatrixColorings: GreedyColoringAlgorithm
using GaussianMarkovRandomFields:
    AutoDiffObservationModel, CompositeObservationModel,
    CompositeObservations, ObservationModel,
    hyperparameters, latent_dimension
import ForwardDiff

# ─── Custom DPPL accumulator: per-group log-likelihood ────────────────────
# Mirrors `_ObsDistributionAccumulator` — runs the full model body but only
# accumulates `Distributions.loglikelihood(right, left)` for observation
# sites whose `getsym(vn)` is in the group. Skipped sites still execute
# their RHS expressions; the savings are limited to the per-site likelihood
# evaluation.
mutable struct _GroupLogLikelihoodAccumulator{T <: Real, N} <: DynamicPPL.AbstractAccumulator
    logp::T
    group_syms::NTuple{N, Symbol}
end

function _GroupLogLikelihoodAccumulator(group_syms::NTuple{N, Symbol}) where {N}
    return _GroupLogLikelihoodAccumulator{Float64, N}(0.0, group_syms)
end

DynamicPPL.accumulator_name(::Type{<:_GroupLogLikelihoodAccumulator}) = :GroupLogLikelihood

Base.copy(a::_GroupLogLikelihoodAccumulator) =
    _GroupLogLikelihoodAccumulator(a.logp, a.group_syms)

function DynamicPPL.reset(a::_GroupLogLikelihoodAccumulator{T}) where {T}
    return _GroupLogLikelihoodAccumulator{T, length(a.group_syms)}(zero(T), a.group_syms)
end

function DynamicPPL.split(a::_GroupLogLikelihoodAccumulator{T}) where {T}
    return _GroupLogLikelihoodAccumulator{T, length(a.group_syms)}(zero(T), a.group_syms)
end

function DynamicPPL.combine(
        a::_GroupLogLikelihoodAccumulator, b::_GroupLogLikelihoodAccumulator,
    )
    return _GroupLogLikelihoodAccumulator(a.logp + b.logp, a.group_syms)
end

function DynamicPPL.accumulate_observe!!(
        a::_GroupLogLikelihoodAccumulator, right, left, vn, template,
    )
    if getsym(vn) in a.group_syms
        contrib = loglikelihood(right, left)
        return _GroupLogLikelihoodAccumulator(a.logp + contrib, a.group_syms)
    end
    return a
end

DynamicPPL.accumulate_assume!!(a::_GroupLogLikelihoodAccumulator, val, tval, lj, vn, d, t) = a

# Per-site variant: stores per-observation contributions for sites whose
# `getsym(vn)` is in `group_syms`. Used to produce a `pointwise_loglik_func`
# for each composite component without going through DPPL's
# `pointwise_loglikelihoods` (which returns a `VarNamedTuple` in current
# DPPL — not a plain Dict — and doesn't iterate as pairs).
mutable struct _GroupPointwiseAccumulator{T <: Real, N} <: DynamicPPL.AbstractAccumulator
    contribs::Vector{T}
    group_syms::NTuple{N, Symbol}
end

function _GroupPointwiseAccumulator(::Type{T}, group_syms::NTuple{N, Symbol}) where {T, N}
    return _GroupPointwiseAccumulator{T, N}(T[], group_syms)
end

DynamicPPL.accumulator_name(::Type{<:_GroupPointwiseAccumulator}) = :GroupPointwise

Base.copy(a::_GroupPointwiseAccumulator) =
    _GroupPointwiseAccumulator(copy(a.contribs), a.group_syms)

function DynamicPPL.reset(a::_GroupPointwiseAccumulator{T}) where {T}
    return _GroupPointwiseAccumulator{T, length(a.group_syms)}(T[], a.group_syms)
end

function DynamicPPL.split(a::_GroupPointwiseAccumulator{T}) where {T}
    return _GroupPointwiseAccumulator{T, length(a.group_syms)}(T[], a.group_syms)
end

function DynamicPPL.combine(
        a::_GroupPointwiseAccumulator, b::_GroupPointwiseAccumulator,
    )
    return _GroupPointwiseAccumulator(vcat(a.contribs, b.contribs), a.group_syms)
end

function DynamicPPL.accumulate_observe!!(
        a::_GroupPointwiseAccumulator, right, left, vn, template,
    )
    if getsym(vn) in a.group_syms
        push!(a.contribs, loglikelihood(right, left))
    end
    return a
end

DynamicPPL.accumulate_assume!!(a::_GroupPointwiseAccumulator, val, tv, lj, vn, d, t) = a

# ─── Spec normalisation + validation ──────────────────────────────────────
"""
    _normalize_obs_groups(spec) -> Vector{Pair{Symbol, NTuple{N, Symbol}}}

Accept either a `NamedTuple` (e.g. `(physics = (:y_phys,), data = (:y_sensor,))`)
or a `Vector{Pair{Symbol, <:Tuple}}`. Returns the canonical pair-vector form.
"""
function _normalize_obs_groups(spec::NamedTuple)
    return [
        Symbol(k) => Tuple(Symbol(s) for s in v) for (k, v) in pairs(spec)
    ]
end

function _normalize_obs_groups(spec::AbstractVector)
    pairs_out = Pair{Symbol, NTuple{N, Symbol} where {N}}[]
    for entry in spec
        if !(entry isa Pair)
            throw(
                ArgumentError(
                    "obs_groups entries must be `Symbol => Tuple{Symbol...}` pairs; got $(typeof(entry))"
                )
            )
        end
        name, syms = entry
        push!(pairs_out, Symbol(name) => Tuple(Symbol(s) for s in syms))
    end
    return pairs_out
end

_normalize_obs_groups(::Nothing) = nothing

"""
    _probe_obs_syms(dppl_model, hp_names, random_syms, dims)

Return the set of distinct `getsym(vn)` symbols at which `dppl_model` emits
an observation. Implemented via `_ObsDistributionAccumulator` — the same
probe used by the fast-path detector, but we collect varname syms instead
of distributions.
"""
function _probe_obs_syms(dppl_model, hp_names::Tuple, random_syms::Tuple, dims::Dict)
    probe_hp = _hp_probe_nt(dppl_model, hp_names)
    # Seed univariate latents as scalars; DPPL's body for `α ~ Normal(0,1)`
    # crashes on `exp(::Vector{Float64})` if we hand it a 1-vector.
    is_scalar = Dict(s => _is_scalar_latent(dppl_model, s, probe_hp) for s in random_syms)
    probe_x = NamedTuple{random_syms}(
        Tuple(_zero_seed(is_scalar[s], dims[s]) for s in random_syms)
    )
    cond = DynamicPPL.fix(dppl_model, probe_hp)
    vi = DynamicPPL.OnlyAccsVarInfo((_ObsVnAccumulator(),))
    vi = last(
        DynamicPPL.init!!(
            cond, vi, DynamicPPL.InitFromParams(probe_x, nothing), DynamicPPL.UnlinkAll(),
        )
    )
    return Set(DynamicPPL.getacc(vi, Val(:ObsVnAccumulator)).syms)
end

mutable struct _ObsVnAccumulator <: DynamicPPL.AbstractAccumulator
    syms::Vector{Symbol}
end
_ObsVnAccumulator() = _ObsVnAccumulator(Symbol[])
DynamicPPL.accumulator_name(::Type{_ObsVnAccumulator}) = :ObsVnAccumulator
Base.copy(a::_ObsVnAccumulator) = _ObsVnAccumulator(copy(a.syms))
DynamicPPL.reset(a::_ObsVnAccumulator) = (empty!(a.syms); a)
DynamicPPL.split(::_ObsVnAccumulator) = _ObsVnAccumulator()
function DynamicPPL.combine(a::_ObsVnAccumulator, b::_ObsVnAccumulator)
    return _ObsVnAccumulator(vcat(a.syms, b.syms))
end
function DynamicPPL.accumulate_observe!!(a::_ObsVnAccumulator, right, left, vn, template)
    push!(a.syms, getsym(vn))
    return a
end
DynamicPPL.accumulate_assume!!(a::_ObsVnAccumulator, val, tv, lj, vn, d, t) = a

"""
    _validate_obs_groups(groups, dppl, hp_names, random_syms, dims)

Verify that the user-declared groups (in canonical pair-vector form):
- cover every observation `~` symbol exactly once,
- only mention symbols that are actually observed (not hyperparameters or
  random variables).
"""
function _validate_obs_groups(
        groups::AbstractVector, dppl_model, hp_names::Tuple, random_syms::Tuple, dims::Dict,
    )
    obs_syms = _probe_obs_syms(dppl_model, hp_names, random_syms, dims)
    declared_syms = Symbol[]
    seen_names = Symbol[]
    for (name, syms) in groups
        if name in seen_names
            throw(
                ArgumentError(
                    "obs_groups has duplicate group name :$(name); group names must be unique"
                )
            )
        end
        push!(seen_names, name)
        if isempty(syms)
            throw(
                ArgumentError(
                    "obs_groups[$(name)] is empty; every group must declare at least one observed symbol"
                )
            )
        end
        for s in syms
            if s in hp_names
                throw(
                    ArgumentError(
                        "obs_groups[$(name)] mentions :$(s), which is a hyperparameter, not an observation"
                    )
                )
            end
            if s in random_syms
                throw(
                    ArgumentError(
                        "obs_groups[$(name)] mentions :$(s), which is a latent random variable, not an observation"
                    )
                )
            end
            if !(s in obs_syms)
                throw(
                    ArgumentError(
                        "obs_groups[$(name)] mentions :$(s), which the DPPL model never observes"
                    )
                )
            end
            if s in declared_syms
                throw(
                    ArgumentError(
                        "obs symbol :$(s) appears in more than one group; each obs sym belongs to exactly one group"
                    )
                )
            end
            push!(declared_syms, s)
        end
    end
    missing_syms = setdiff(obs_syms, Set(declared_syms))
    if !isempty(missing_syms)
        throw(
            ArgumentError(
                "obs_groups must cover every observed symbol; missing: $(sort(collect(missing_syms)))"
            )
        )
    end
    return nothing
end

# ─── Per-group log-likelihood + composite assembly ────────────────────────
"""
    _make_group_loglik(dppl_model, group_syms, hp_names, random_syms, dims)

Return a closure `(x; kwargs...) -> Real` that runs `dppl_model` with
hyperparameters fixed from `kwargs` and latent values seeded from the
flat `x`, accumulating the log-likelihood only at observation sites whose
`getsym(vn)` is in `group_syms`.
"""
function _make_group_loglik(
        dppl_model, group_syms::NTuple{N, Symbol},
        hp_names::Tuple, random_syms::Tuple, dims::Dict,
        is_scalar::Dict{Symbol, Bool},
    ) where {N}
    offsets = _component_offsets(random_syms, dims)
    function loglik(x; kwargs...)
        hp_nt = NamedTuple{hp_names}(Tuple(kwargs[k] for k in hp_names))
        rand_nt = NamedTuple{random_syms}(
            Tuple(
                is_scalar[s] ? x[first(offsets[s])] : Vector(x[offsets[s]])
                    for s in random_syms
            )
        )
        cond = DynamicPPL.fix(dppl_model, hp_nt)
        # The accumulator scalar type must accommodate AD partials from
        # *either* `x` (inner Newton differentiating w.r.t. latents) or
        # the hp kwargs (outer hp-gradient pass injecting Duals). Start
        # at the promoted type so `accumulate_observe!!` doesn't have to
        # silently widen between calls.
        T = promote_type(eltype(x), map(typeof, values(hp_nt))...)
        acc0 = _GroupLogLikelihoodAccumulator{T, N}(zero(T), group_syms)
        vi = DynamicPPL.OnlyAccsVarInfo((acc0,))
        vi = last(
            DynamicPPL.init!!(
                cond, vi, DynamicPPL.InitFromParams(rand_nt, nothing), DynamicPPL.UnlinkAll(),
            )
        )
        return DynamicPPL.getacc(vi, Val(:GroupLogLikelihood)).logp
    end
    return loglik
end

# Per-group pointwise log-likelihood: returns a vector with one entry per
# observation site whose `getsym(vn)` is in `group_syms`. Used to attach
# `pointwise_loglik_func` to composite components so WAIC / CPO can
# integrate over the group's sites without silently no-op-ing.
function _make_group_pointwise_loglik(
        dppl_model, group_syms::NTuple{N, Symbol},
        hp_names::Tuple, random_syms::Tuple, dims::Dict,
        is_scalar::Dict{Symbol, Bool},
    ) where {N}
    offsets = _component_offsets(random_syms, dims)
    function pointwise_loglik(x; kwargs...)
        hp_nt = NamedTuple{hp_names}(Tuple(kwargs[k] for k in hp_names))
        rand_nt = NamedTuple{random_syms}(
            Tuple(
                is_scalar[s] ? x[first(offsets[s])] : Vector(x[offsets[s]])
                    for s in random_syms
            )
        )
        cond = DynamicPPL.fix(dppl_model, hp_nt)
        T = promote_type(eltype(x), map(typeof, values(hp_nt))...)
        acc0 = _GroupPointwiseAccumulator{T, N}(T[], group_syms)
        vi = DynamicPPL.OnlyAccsVarInfo((acc0,))
        vi = last(
            DynamicPPL.init!!(
                cond, vi, DynamicPPL.InitFromParams(rand_nt, nothing), DynamicPPL.UnlinkAll(),
            )
        )
        return DynamicPPL.getacc(vi, Val(:GroupPointwise)).contribs
    end
    return pointwise_loglik
end

"""
    _build_obs_groups_composite(dppl, groups, hp_names, n_latent, random_syms, dims, hessian_pattern)

Construct the underlying `CompositeObservationModel` and the matching
`CompositeObservations` payload for the user-declared `groups`. Each
component is an `AutoDiffObservationModel` that uses
`_GroupLogLikelihoodAccumulator` to compute its slice of the log-likelihood;
each component declares the full `hp_names` tuple as its hyperparameters,
and routes are explicit identity NamedTuples (not `nothing`).
"""
function _build_obs_groups_composite(
        dppl_model, groups::AbstractVector,
        hp_names::Tuple, n_latent::Int,
        random_syms::Tuple, dims::Dict,
        hessian_pattern;
        fast_results::AbstractDict = Dict{Symbol, Any}(),
    )
    args = dppl_model.args
    probe_hp = _hp_probe_nt(dppl_model, hp_names)
    is_scalar = Dict(s => _is_scalar_latent(dppl_model, s, probe_hp) for s in random_syms)

    built = map(groups) do (name, syms)
        fast = get(fast_results, name, nothing)
        if fast !== nothing
            # Fast-path component: rename-only route from the family's
            # nuisance kwargs to outer hp names, plus emission-order y so
            # the component's `loglik` aligns with the design matrix rows.
            # Keep the wrapped observation type (PoissonObservations etc.)
            # — `collect` would flatten it back to a raw tuple vector and
            # blow up `_materialize` dispatch.
            return (component = fast.model, route = fast.route, y = fast.y)
        end
        component = _build_one_ad_component(
            dppl_model, syms, hp_names, n_latent, random_syms, dims,
            hessian_pattern, is_scalar,
        )
        return (
            component = component,
            route = _identity_route(hp_names),
            y = _collect_group_y(args, syms),
        )
    end

    components = Tuple(b.component for b in built)
    routes = Tuple(b.route for b in built)
    composite = CompositeObservationModel(components, routes)
    composite_obs = CompositeObservations(Tuple(b.y for b in built))
    return composite, composite_obs
end

function _build_one_ad_component(
        dppl_model, group_syms::Tuple,
        hp_names::Tuple, n_latent::Int,
        random_syms::Tuple, dims::Dict,
        hessian_pattern,
        is_scalar::Dict{Symbol, Bool},
    )
    loglik = _make_group_loglik(
        dppl_model, group_syms, hp_names, random_syms, dims, is_scalar,
    )
    pointwise = _make_group_pointwise_loglik(
        dppl_model, group_syms, hp_names, random_syms, dims, is_scalar,
    )
    sparsity_detector = hessian_pattern === nothing ?
        TracerLocalSparsityDetector() :
        KnownHessianSparsityDetector(hessian_pattern)
    hess_backend = AutoSparse(
        AutoForwardDiff();
        sparsity_detector = sparsity_detector,
        coloring_algorithm = GreedyColoringAlgorithm(),
    )
    # Pin `grad_backend = AutoForwardDiff()`. Mooncake's
    # `LinearSolveMooncakeExt.solve!_adjoint` is broken for the
    # `SymTridiagonal{Float64} + LDLt` shapes the latent prior produces,
    # and IFT's internal `loggrad` uses the AD likelihood's stored
    # `grad_backend`. ForwardDiff sidesteps that. `diagonal_hessian_safe
    # = false` because composite components generally have non-trivial
    # linear predictors. With GMRFs #102/#107, attaching
    # `pointwise_loglik_func` no longer forces the diagonal-Hessian
    # shortcut, so WAIC / CPO can integrate per-site over each group.
    return AutoDiffObservationModel(
        loglik;
        n_latent = n_latent,
        hyperparams = hp_names,
        grad_backend = AutoForwardDiff(),
        hessian_backend = hess_backend,
        pointwise_loglik_func = pointwise,
        diagonal_hessian_safe = false,
    )
end

_identity_route(hp_names::Tuple) = NamedTuple{hp_names}(hp_names)

function _collect_group_y(args::NamedTuple, group_syms::Tuple)
    parts = Any[]
    for s in group_syms
        if !haskey(args, s)
            # The probe confirmed the sym IS observed but its data wasn't
            # bound to a top-level @model argument (e.g. observed via a
            # closure or computed inside the model body). The loglik
            # closure ignores `y` so inference would still work, but the
            # `CompositeObservations` length / display / future pointwise
            # diagnostics would silently lie. Reject explicitly.
            throw(
                ArgumentError(
                    "obs_groups currently requires grouped observed symbols to be top-level @model arguments; :$(s) wasn't passed when the @model was constructed"
                )
            )
        end
        v = getfield(args, s)
        push!(parts, v isa AbstractArray ? collect(v) : [v])
    end
    return reduce(vcat, parts)
end

# ─── Wrapper: ObservationModel that bakes in CompositeObservations ────────
# `CompositeObservationModel.(y::CompositeObservations; kwargs...)` requires
# `length(y.components) == length(model.components)`. The LGM contract calls
# `model.observation_model(y; kwargs...)` with whatever `y` the user passed
# to `inla(model, y)` — typically a flat `Vector`. The wrapper substitutes
# the prebaked `CompositeObservations` so we keep the existing
# `inla(model, y_obs)` ergonomics.
struct _DPPLCompositeObservationModel{C <: CompositeObservationModel, Y <: CompositeObservations, H} <: ObservationModel
    composite::C
    composite_y::Y
    hp_names::H
    n_latent::Int
end

function (w::_DPPLCompositeObservationModel)(_y; kwargs...)
    return w.composite(w.composite_y; kwargs...)
end

hyperparameters(w::_DPPLCompositeObservationModel) = w.hp_names
latent_dimension(w::_DPPLCompositeObservationModel, ::AbstractVector) = w.n_latent

function Base.show(io::IO, w::_DPPLCompositeObservationModel)
    print(
        io, "DPPL-composite ObservationModel(", length(w.composite.components),
        " groups, hp = ", w.hp_names, ")"
    )
    return
end

# Internal accessor for tests/introspection.
_underlying_composite(w::_DPPLCompositeObservationModel) = w.composite
_underlying_composite(_) = nothing

# ─── Lifted composite obs model ───────────────────────────────────────────────
# Prelude-lift variant: per-component the AD-fallback paths re-use a
# top-level `obs_body_fn` (generated by the `@latte` macro) that destructures
# a payload supplied via the GMRFs `y` slot. The wrapper computes the
# hp-dependent prelude state once per `model(_y; θ_nt...)` call and
# substitutes per-component payloads into the inner composite call.
#
# Fast-path components are routed through unchanged — their `composite_y`
# entry is the closed-form likelihood payload (PoissonObservations etc.),
# and `is_lifted[i] === false` keeps that entry through at materialisation.

struct _LiftedCompositeObsModel{C <: CompositeObservationModel, BY, H, P, A, O, S, GS, IL} <: ObservationModel
    composite::C
    base_y::BY                # CompositeObservations: fast components carry their y; lifted components are nothing
    hp_names::H
    n_latent::Int
    prelude_fn::P
    args_nt::A
    offsets::O                # NamedTuple{random_syms}
    is_scalar::S              # NamedTuple{random_syms}
    group_syms_per_component::GS  # NTuple{ncomp, NTuple{N, Symbol}}
    is_lifted::IL             # NTuple{ncomp, Bool}
end

function (w::_LiftedCompositeObsModel)(_y; kwargs...)
    hp_nt = NamedTuple{w.hp_names}(Tuple(kwargs[k] for k in w.hp_names))
    prelude_state = w.prelude_fn(w.args_nt, hp_nt)
    new_components = ntuple(length(w.is_lifted)) do i
        if w.is_lifted[i]
            (
                args = w.args_nt, prelude_state = prelude_state,
                group_syms = w.group_syms_per_component[i],
                offsets = w.offsets, is_scalar = w.is_scalar,
            )
        else
            w.base_y.components[i]
        end
    end
    return w.composite(CompositeObservations(new_components); kwargs...)
end

hyperparameters(w::_LiftedCompositeObsModel) = w.hp_names
latent_dimension(w::_LiftedCompositeObsModel, ::AbstractVector) = w.n_latent

function Base.show(io::IO, w::_LiftedCompositeObsModel)
    n_lifted = count(w.is_lifted)
    n_total = length(w.is_lifted)
    print(
        io, "Lifted-composite ObservationModel(", n_total,
        " groups, ", n_lifted, " lifted, hp = ", w.hp_names, ")"
    )
    return
end

_underlying_composite(w::_LiftedCompositeObsModel) = w.composite

# Build the per-group AD component using the lifted `obs_body_fn` from
# `lift_spec` instead of the DPPL `_make_group_loglik` closure. The
# component's hyperparams declaration is the full outer `hp_names` tuple so
# routes line up with non-lifted siblings.
function _build_one_ad_component_lifted(
        n_latent::Int, hp_names::Tuple, hessian_pattern,
        lift_spec::NamedTuple,
    )
    sparsity_detector = hessian_pattern === nothing ?
        TracerLocalSparsityDetector() :
        KnownHessianSparsityDetector(hessian_pattern)
    hess_backend = AutoSparse(
        AutoForwardDiff();
        sparsity_detector = sparsity_detector,
        coloring_algorithm = GreedyColoringAlgorithm(),
    )
    return AutoDiffObservationModel(
        lift_spec.obs_body_fn;
        n_latent = n_latent,
        hyperparams = hp_names,
        grad_backend = AutoForwardDiff(),
        hessian_backend = hess_backend,
        pointwise_loglik_func = lift_spec.pointwise_fn,
        diagonal_hessian_safe = false,
    )
end

"""
    _build_obs_groups_observation_model(...)

Unified composite obs-model builder. With `lift_spec === nothing` this
reproduces the legacy `_DPPLCompositeObservationModel` wrap around
`_build_obs_groups_composite`. With `lift_spec` present, AD-fallback
components route through `_build_one_ad_component_lifted` and the whole
thing is wrapped in `_LiftedCompositeObsModel` so the hp-prelude runs
once per θ instead of once per AD sweep.
"""
function _build_obs_groups_observation_model(
        dppl_model, groups::AbstractVector,
        hp_names::Tuple, n_latent::Int,
        random_syms::Tuple, dims::Dict,
        hessian_pattern;
        fast_results::AbstractDict = Dict{Symbol, Any}(),
        lift_spec = nothing,
    )
    if lift_spec === nothing
        composite, composite_obs = _build_obs_groups_composite(
            dppl_model, groups, hp_names, n_latent,
            random_syms, dims, hessian_pattern;
            fast_results = fast_results,
        )
        return _DPPLCompositeObservationModel(
            composite, composite_obs, hp_names, n_latent,
        )
    end

    # ─── Lifted path ──────────────────────────────────────────────────────
    args = dppl_model.args
    probe_hp = _hp_probe_nt(dppl_model, hp_names)
    is_scalar_dict = Dict(s => _is_scalar_latent(dppl_model, s, probe_hp) for s in random_syms)
    offsets_dict = _component_offsets(random_syms, dims)

    built = map(groups) do (name, syms)
        fast = get(fast_results, name, nothing)
        if fast !== nothing
            return (
                component = fast.model, route = fast.route, y = fast.y,
                lifted = false, syms = syms,
            )
        end
        component = _build_one_ad_component_lifted(
            n_latent, hp_names, hessian_pattern, lift_spec,
        )
        return (
            component = component, route = _identity_route(hp_names),
            y = nothing, lifted = true, syms = syms,
        )
    end

    components = Tuple(b.component for b in built)
    routes = Tuple(b.route for b in built)
    composite = CompositeObservationModel(components, routes)

    # Build base_y: fast components contribute their closed-form y;
    # lifted components are placeholders (will be replaced by payload at
    # materialisation).
    base_y_entries = Tuple(b.lifted ? nothing : b.y for b in built)
    base_y = CompositeObservations(base_y_entries)

    group_syms_per_component = Tuple(Tuple(b.syms) for b in built)
    is_lifted = Tuple(b.lifted for b in built)

    args_nt = NamedTuple{Tuple(keys(args))}(
        Tuple(getfield(args, s) for s in keys(args))
    )
    offsets_nt = NamedTuple{random_syms}(
        Tuple(offsets_dict[s] for s in random_syms)
    )
    is_scalar_nt = NamedTuple{random_syms}(
        Tuple(is_scalar_dict[s] for s in random_syms)
    )

    return _LiftedCompositeObsModel(
        composite, base_y, hp_names, n_latent,
        lift_spec.prelude_fn, args_nt, offsets_nt, is_scalar_nt,
        group_syms_per_component, is_lifted,
    )
end
