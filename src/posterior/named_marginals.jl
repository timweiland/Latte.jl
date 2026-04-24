using OrderedCollections: OrderedDict

export NamedMarginals

"""
    NamedMarginals{V} <: AbstractVector

Thin wrapper over a vector of posterior marginal distributions, tagged with
a `sym → UnitRange{Int}` layout so users can pick out named blocks via
property access (`x.β`) or symbol indexing (`x[:β]`). Iteration and integer
indexing still behave exactly like the underlying vector.
"""
struct NamedMarginals{T, V <: AbstractVector{T}} <: AbstractVector{T}
    parent::V
    groups::OrderedDict{Symbol, UnitRange{Int}}

    NamedMarginals(parent::V, groups::OrderedDict{Symbol, UnitRange{Int}}) where {T, V <: AbstractVector{T}} =
        new{T, V}(parent, groups)
end

Base.parent(x::NamedMarginals) = getfield(x, :parent)
_groups(x::NamedMarginals) = getfield(x, :groups)

Base.size(x::NamedMarginals) = size(parent(x))
Base.length(x::NamedMarginals) = length(parent(x))
Base.IndexStyle(::Type{<:NamedMarginals{T, V}}) where {T, V} = IndexStyle(V)

Base.getindex(x::NamedMarginals, i::Int) = parent(x)[i]
Base.getindex(x::NamedMarginals, I::AbstractVector) = parent(x)[I]
Base.getindex(x::NamedMarginals, r::AbstractUnitRange) = parent(x)[r]

function Base.getindex(x::NamedMarginals, sym::Symbol)
    g = _groups(x)
    haskey(g, sym) || throw(KeyError(sym))
    return parent(x)[g[sym]]
end

function Base.getproperty(x::NamedMarginals, sym::Symbol)
    sym === :parent && return getfield(x, :parent)
    sym === :groups && return getfield(x, :groups)
    g = getfield(x, :groups)
    haskey(g, sym) || throw(KeyError(sym))
    return getfield(x, :parent)[g[sym]]
end
Base.propertynames(x::NamedMarginals) = Tuple(keys(_groups(x)))

Base.iterate(x::NamedMarginals, state...) = iterate(parent(x), state...)
