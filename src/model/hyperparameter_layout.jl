# Flat-layout helpers for hyperparameter specs.
#
# Each free hyperparameter contributes `_hp_dim` coordinates to the flat
# working/natural parameter vector; blocks are laid out in declaration order.
# Vector-valued entries are restricted (at `Hyperparameter` construction) to
# dimension-preserving elementwise transforms, so working and natural space
# share this single layout.

using Distributions
using OrderedCollections: OrderedDict

"""
    _hp_dim(hp::Hyperparameter) -> Int

Number of scalar coordinates a hyperparameter contributes to the flat
parameter vector: 1 for univariate priors, `length(prior)` for multivariate
(vector-valued) priors.
"""
_hp_dim(hp::Hyperparameter) = _prior_dim(hp.prior)

_prior_dim(d::Distribution{Univariate}) = 1
_prior_dim(d::Distribution{Multivariate}) = length(d)
# Best effort for distributions without a clean variate-form dispatch
# (e.g. custom working-space priors): `length`, defaulting to 1.
function _prior_dim(d)
    try
        return length(d)
    catch
        return 1
    end
end

"""
    _hp_isscalar(hp::Hyperparameter) -> Bool

Whether the hyperparameter's value is a scalar (univariate prior). A
vector-valued hyperparameter — even of length 1 — keeps vector semantics:
`getproperty` and `convert(NamedTuple, θ)` expose it as a `Vector`.
"""
_hp_isscalar(hp::Hyperparameter) = hp.prior isa Distribution{Univariate}

"""
    _hp_total_dim(spec::HyperparameterSpec) -> Int

Total number of scalar coordinates in the flat parameter vector.
"""
_hp_total_dim(spec::HyperparameterSpec) =
    mapreduce(_hp_dim, +, values(spec.free); init = 0)

"""
    _hp_ranges(spec::HyperparameterSpec) -> NamedTuple

Name → `UnitRange` mapping into the flat parameter vector, in declaration
order.
"""
function _hp_ranges(spec::HyperparameterSpec)
    names = keys(spec.free)
    off = 0
    ranges = map(names) do name
        d = _hp_dim(spec.free[name])
        r = (off + 1):(off + d)
        off += d
        r
    end
    return NamedTuple{names}(ranges)
end

"""
    _expanded_hp_names(spec::HyperparameterSpec) -> Vector{Symbol}

One name per flat coordinate: the plain name for scalar (dim-1) entries,
`name[i]` for the components of vector-valued entries. Matches the labels
SBC uses for vector targets.
"""
function _expanded_hp_names(spec::HyperparameterSpec)
    out = Symbol[]
    for name in keys(spec.free)
        d = _hp_dim(spec.free[name])
        if d == 1
            push!(out, name)
        else
            for i in 1:d
                push!(out, Symbol(name, "[", i, "]"))
            end
        end
    end
    return out
end

"""
    _coordinate_hp(spec::HyperparameterSpec, d::Int)
        -> (name::Symbol, hp::Hyperparameter, component::Int)

Resolve flat coordinate `d` to the hyperparameter block it belongs to and
its component index within that block. Per-coordinate transforms are
well-defined because vector entries only admit elementwise transforms.
"""
function _coordinate_hp(spec::HyperparameterSpec, d::Int)
    off = 0
    for name in keys(spec.free)
        hp = spec.free[name]
        dim = _hp_dim(hp)
        if d <= off + dim
            return name, hp, d - off
        end
        off += dim
    end
    throw(BoundsError(spec.free, d))
end

"""
    hyperparameter_groups(spec::HyperparameterSpec)
        -> OrderedDict{Symbol, UnitRange{Int}}

Name → flat-coordinate-range mapping for the free hyperparameters. Scalar
entries map to a length-1 range, vector-valued entries to the range of their
components. Shared by the inference results' `hyperparameter_groups`.
"""
function hyperparameter_groups(spec::HyperparameterSpec)
    groups = OrderedDict{Symbol, UnitRange{Int}}()
    off = 0
    for name in keys(spec.free)
        d = _hp_dim(spec.free[name])
        groups[name] = (off + 1):(off + d)
        off += d
    end
    return groups
end

# ─── Block-wise application and flattening ───────────────────────────────────

"""
    _map_hp_blocks(f, spec, θ::AbstractVector) -> Vector

Apply `f(hp, value)` to each free block of the flat vector `θ` (a scalar for
univariate entries, a `Vector` slice for vector-valued ones) and re-flatten
the results in layout order. Result eltype is promoted across blocks, so AD
number types pass through.
"""
function _map_hp_blocks(f, spec::HyperparameterSpec, θ::AbstractVector)
    names = keys(spec.free)
    blocks = Vector{Any}(undef, length(names))
    off = 0
    for (i, name) in enumerate(names)
        hp = spec.free[name]
        d = _hp_dim(hp)
        blocks[i] = _hp_isscalar(hp) ? f(hp, θ[off + 1]) : f(hp, θ[(off + 1):(off + d)])
        off += d
    end
    return _flatten_hp_blocks(blocks)
end

"""
    _flatten_hp_blocks(blocks) -> Vector

Concatenate a mix of scalars and vectors into one flat vector with promoted
eltype.
"""
function _flatten_hp_blocks(blocks::AbstractVector)
    T = mapreduce(_block_eltype, promote_type, blocks)
    n = sum(_block_length, blocks)
    out = Vector{T}(undef, n)
    off = 0
    for b in blocks
        if b isa AbstractVector
            copyto!(out, off + 1, b, 1, length(b))
            off += length(b)
        else
            out[off + 1] = b
            off += 1
        end
    end
    return out
end

_block_eltype(x::AbstractVector) = eltype(x)
_block_eltype(x) = typeof(x)
_block_length(x::AbstractVector) = length(x)
_block_length(::Any) = 1

"""
    _flatten_hp_namedtuple(nt::NamedTuple, spec) -> Vector{Float64}

Flatten a user-supplied per-name NamedTuple of natural-space values into the
flat layout, validating that vector-valued entries have the block's length.
"""
function _flatten_hp_namedtuple(nt::NamedTuple, spec::HyperparameterSpec)
    out = Vector{Float64}(undef, _hp_total_dim(spec))
    off = 0
    for name in keys(spec.free)
        hp = spec.free[name]
        d = _hp_dim(hp)
        v = nt[name]
        if _hp_isscalar(hp)
            out[off + 1] = Float64(v)
        else
            v isa AbstractVector && length(v) == d || throw(
                ArgumentError(
                    "hyperparameter `$name` is vector-valued with $d components; " *
                        "got `$v`"
                )
            )
            out[(off + 1):(off + d)] .= Float64.(v)
        end
        off += d
    end
    return out
end
