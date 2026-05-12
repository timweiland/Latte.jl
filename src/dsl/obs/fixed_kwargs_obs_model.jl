# `_FixedKwargsObservationModel`: thin wrapper that pre-binds constant
# nuisance kwargs into a base observation model. Used by the per-group
# fast-path detector when an inner kwarg of the family (e.g. `σ` for
# Normal, `phi` for Gamma) is hardcoded in the DPPL body — no outer hp
# drives it — but the component is otherwise a textbook
# `LinearlyTransformedObservationModel(ExponentialFamily(F), A)`.
#
# By baking the constant into the model object instead of forcing the
# whole group through `AutoDiffObservationModel`, we keep the closed-form
# likelihood path (and its Hessian).
#
# Sits *inside* the LTM in the assembly order:
#     LTM(_FixedKwargsObservationModel(base, fixed), A)
# so downstream code that pattern-matches `obs isa
# LinearlyTransformedObservationModel` still sees the LTM shape.

using GaussianMarkovRandomFields:
    ObservationModel, hyperparameters, latent_dimension

struct _FixedKwargsObservationModel{M <: ObservationModel, K <: NamedTuple} <: ObservationModel
    base::M
    fixed_kwargs::K
end

function (m::_FixedKwargsObservationModel)(y; kwargs...)
    nt = values(kwargs)
    overlap = intersect(keys(nt), keys(m.fixed_kwargs))
    if !isempty(overlap)
        throw(
            ArgumentError(
                "kwarg(s) $(Tuple(overlap)) are pre-bound as constants on this observation model and cannot be passed in"
            )
        )
    end
    return m.base(y; merge(nt, m.fixed_kwargs)...)
end

# Subtract fixed names from the inherited hyperparameter set so callers
# that introspect the model's outer kwarg surface see only the
# variable-driven kwargs.
function hyperparameters(m::_FixedKwargsObservationModel)
    fixed = keys(m.fixed_kwargs)
    return Tuple(s for s in hyperparameters(m.base) if !(s in fixed))
end

latent_dimension(m::_FixedKwargsObservationModel, y) = latent_dimension(m.base, y)

function Base.show(io::IO, m::_FixedKwargsObservationModel)
    print(io, "_FixedKwargsObservationModel(", m.base, ", fixed=", m.fixed_kwargs, ")")
    return
end
