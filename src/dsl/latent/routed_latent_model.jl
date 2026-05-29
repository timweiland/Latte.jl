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
using GaussianMarkovRandomFields:
    LatentModel, CombinedModel, GMRFWorkspace,
    hyperparameters, precision_matrix, constraints, model_name

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
# and workspace handling) after renaming kwargs.
(m::RoutedLatentModel)(; kwargs...) = m.inner(; _route_inner_kwargs(m, kwargs)...)
(m::RoutedLatentModel)(ws::GMRFWorkspace; kwargs...) =
    m.inner(ws; _route_inner_kwargs(m, kwargs)...)

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
    base isa RoutedLatentModel || return nothing
    syms = collect(keys(model.latent_layout))
    inner = base.inner
    comps = inner isa CombinedModel ? inner.components : LatentModel[inner]
    return OrderedDict{Symbol, LatentModel}(s => c for (s, c) in zip(syms, comps))
end
