# `RoutedLatentModel` — preserve a user-supplied concrete `LatentModel`
# (e.g. `RW1Model`, `IIDModel`) recognized on a `~` site by the `@latte`
# macro, while translating the LGM's outer hyperparameter names to the
# inner model's constructor-call kwarg names.
#
# A `~` site like `x ~ RW1Model(n)(; τ = τ)` recognizes the inner model
# `RW1Model(n)` together with a *route* mapping its call kwarg `τ` to the
# outer hyperparameter symbol `τ`. The route is a NamedTuple keyed by the
# inner kwarg names whose values are the outer hp symbols:
#
#     route = (; τ = :τ)        # inner kwarg `τ` ← outer hp `τ`
#
# At inference time the engine calls the latent prior with *all* natural-
# space hp values (e.g. `(; τ = 2.0, σ = 0.5)`). `RoutedLatentModel`
# selects + renames just the inner model's kwargs and forwards them.

import Distributions
import GaussianMarkovRandomFields
import LinearSolve
using SparseArrays: SparseMatrixCSC
using LinearAlgebra: Diagonal
using GaussianMarkovRandomFields:
    LatentModel, CombinedModel, GMRFWorkspace, GMRF, ConstrainedGMRF,
    AbstractLatentWorkspace,
    hyperparameters, precision_matrix, constraints, model_name
import GaussianMarkovRandomFields: make_workspace, make_workspace_pool

struct RoutedLatentModel{M <: LatentModel, R <: NamedTuple} <: LatentModel
    inner::M
    route::R
end

# Outer hp kwargs -> inner call kwargs, selecting + renaming via `route`.
@inline function _route_inner_kwargs(m::RoutedLatentModel, kwargs)
    kw = (; kwargs...)
    inner_vals = map(outer_sym -> kw[outer_sym], values(m.route))
    return NamedTuple{keys(m.route)}(inner_vals)
end

Base.length(m::RoutedLatentModel) = length(m.inner)
hyperparameters(m::RoutedLatentModel) = hyperparameters(m.inner)
model_name(m::RoutedLatentModel) = model_name(m.inner)

precision_matrix(m::RoutedLatentModel; kwargs...) =
    precision_matrix(m.inner; _route_inner_kwargs(m, kwargs)...)

Distributions.mean(m::RoutedLatentModel; kwargs...) =
    Distributions.mean(m.inner; _route_inner_kwargs(m, kwargs)...)

constraints(m::RoutedLatentModel; kwargs...) =
    constraints(m.inner; _route_inner_kwargs(m, kwargs)...)

# Delegate both call paths to the inner model (which owns its own `alg`
# and workspace handling) after renaming kwargs. The workspace-backed call is
# generic over `AbstractLatentWorkspace`, not just `GMRFWorkspace`, so non-GMRF
# backends (e.g. a filter with its own workspace type) route through too.
(m::RoutedLatentModel)(; kwargs...) = m.inner(; _route_inner_kwargs(m, kwargs)...)
(m::RoutedLatentModel)(ws::AbstractLatentWorkspace; kwargs...) =
    m.inner(ws; _route_inner_kwargs(m, kwargs)...)
# Same body, but the explicit `GMRFWorkspace` method disambiguates against
# GMRFs' generic `(::LatentModel)(::GMRFWorkspace)` at the
# `(RoutedLatentModel, GMRFWorkspace)` intersection (neither the model nor the
# workspace type dominates without it).
(m::RoutedLatentModel)(ws::GMRFWorkspace; kwargs...) =
    m.inner(ws; _route_inner_kwargs(m, kwargs)...)

# Forward the workspace-construction protocol to the inner model. Without this,
# `make_workspace`/`make_workspace_pool` fall back to the GMRF default, which
# calls `precision_matrix` — wrong (and throwing) for workspace-only backends
# that have no single sparse precision.
make_workspace(m::RoutedLatentModel; kwargs...) =
    make_workspace(m.inner; _route_inner_kwargs(m, kwargs)...)
make_workspace_pool(m::RoutedLatentModel; size::Int = Threads.nthreads(), kwargs...) =
    make_workspace_pool(m.inner; size = size, _route_inner_kwargs(m, kwargs)...)

# Forward the cheap-prior-logdet structure hook through both wrappers, so the
# materialized prior carries it and `logdetcov` never factorizes the joint
# workspace at Q_prior. Both forwards are exact: routing only renames kwargs,
# and pattern augmentation adds structural zeros (no determinant effect).
@static if isdefined(GaussianMarkovRandomFields, :precision_logdet)
    GaussianMarkovRandomFields.precision_logdet(m::RoutedLatentModel; kwargs...) =
        GaussianMarkovRandomFields.precision_logdet(m.inner; _route_inner_kwargs(m, kwargs)...)
end

# Probing stand-in for the `~` slot of a recognized latent in the DPPL model
# behind `@latte` — serves both `@latte` assembly and the `Latte.dppl_model`
# Turing handoff.
#
# For a latent with a sparse precision (AR1, IID, Besag, Matérn, …) we
# materialise the *real* prior `m(; kw...)` so the DPPL model is a faithful
# generative model and the Turing handoff couples the hyperparameters to the
# latent. Two cases keep the cheap dimension-matched stand-in: workspace-only
# backends (no single sparse precision, e.g. a filter — their real prior is
# supplied separately during recognition), and `@latte` *assembly*, which
# probes the body at a sentinel θ (all 1.0) that is out of domain for bounded
# hyperparameters (an AR(1) ρ = 1.0) — there only the slot's *shape* matters,
# so a domain error from the sentinel falls back rather than aborting assembly.
# A non-`LatentModel` callee (a curried distribution factory for the DAG
# fallback) is materialised for real.
function _recognized_latent_probe(m::LatentModel, kw::NamedTuple)
    _has_sparse_precision(m) || return _probe_standin(m)
    try
        return m(; kw...)
    catch e
        (e isa ArgumentError || e isa DomainError) || rethrow(e)
        return _probe_standin(m)
    end
end
_recognized_latent_probe(m, kw::NamedTuple) = m(; kw...)
_probe_standin(m) = Distributions.MvNormal(zeros(length(m)), Diagonal(ones(length(m))))

# Whether Latte can materialise a sparse precision for this latent — used to
# decide pattern augmentation / auto-augmentation. Workspace-only backends
# (custom `gaussian_approximation`, no single sparse precision, e.g. a filter)
# override this to `false`; their GA handles the likelihood coupling itself.
_has_sparse_precision(::LatentModel) = true
_has_sparse_precision(m::RoutedLatentModel) = _has_sparse_precision(m.inner)

# `_PatternAugmentedLatentModel` — wrap a latent prior so its precision pattern
# is a superset of the likelihood Hessian's. Mirrors what the DAG path bakes
# into its per-θ precision via `augment_pattern`; needed when a recognized
# (structurally sparse) prior is paired with a likelihood that couples latents
# beyond the prior pattern — e.g. a fixed-effect `β` whose `FixedEffectsModel`
# precision is diagonal but whose `dot(A, β)` predictor gives a dense Hessian.
# Adds structural zeros only — no numeric effect. The generic warm-path
# `(::LatentModel)(ws; …)` and workspace builders route through `precision_matrix`
# / `mean`, so overriding those suffices for the whole inla pipeline.
struct _PatternAugmentedLatentModel{M <: LatentModel, P} <: LatentModel
    inner::M
    pattern::P
end

Base.length(m::_PatternAugmentedLatentModel) = length(m.inner)
hyperparameters(m::_PatternAugmentedLatentModel) = hyperparameters(m.inner)
model_name(m::_PatternAugmentedLatentModel) = model_name(m.inner)
Distributions.mean(m::_PatternAugmentedLatentModel; kwargs...) = Distributions.mean(m.inner; kwargs...)
constraints(m::_PatternAugmentedLatentModel; kwargs...) = constraints(m.inner; kwargs...)

precision_matrix(m::_PatternAugmentedLatentModel; kwargs...) =
    augment_pattern(SparseMatrixCSC(precision_matrix(m.inner; kwargs...)), m.pattern)

# The prior log-determinant is unchanged by pattern augmentation (structural
# zeros only), so the structure hook forwards to the inner model. The @static
# guard keeps the method definition compatible with GMRFs versions that
# predate the hook.
@static if isdefined(GaussianMarkovRandomFields, :precision_logdet)
    GaussianMarkovRandomFields.precision_logdet(m::_PatternAugmentedLatentModel; kwargs...) =
        GaussianMarkovRandomFields.precision_logdet(m.inner; kwargs...)
end

function (m::_PatternAugmentedLatentModel)(; kwargs...)
    μ = Distributions.mean(m.inner; kwargs...)
    Q = precision_matrix(m; kwargs...)
    c = constraints(m.inner; kwargs...)
    g = GMRF(μ, Q, LinearSolve.CHOLMODFactorization())
    return c === nothing ? g : ConstrainedGMRF(g, c[1], c[2])
end

"""
    latent_components(model::LatentGaussianModel) -> OrderedDict{Symbol, LatentModel} | nothing

Per-component concrete latent priors recognized from the `@latte` body,
keyed by latent symbol in body order. Lets downstream code dispatch on the
concrete prior type (e.g. `RWModel{1}`, `IIDModel`) rather than a type-erased
cached latent.

Returns `nothing` when the latent prior was not recognized as concrete
`LatentModel`(s) — i.e. the DAG / sparse-AD path (`latte_from_dppl` without
the macro). A single recognized component yields a one-entry mapping; multiple
components are unwrapped from the `CombinedModel`.
"""
function latent_components(model::LatentGaussianModel)
    lp = model.latent_prior
    base = lp isa AugmentedLatentModel ? lp.base_model : lp
    base isa _PatternAugmentedLatentModel && (base = base.inner)
    base isa RoutedLatentModel || return nothing
    syms = collect(keys(model.latent_layout))
    inner = base.inner
    comps = inner isa CombinedModel ? inner.components : LatentModel[inner]
    return OrderedDict{Symbol, LatentModel}(s => c for (s, c) in zip(syms, comps))
end
