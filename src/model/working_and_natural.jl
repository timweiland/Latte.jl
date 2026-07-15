using Bijectors

export WorkingHyperparameters, NaturalHyperparameters
export logdetjac

################################
##          STRUCTS          ###
################################
"""
    WorkingHyperparameters{T, Spec} <: AbstractVector{T}

Hyperparameters in working (unconstrained) space.

Behaves like a vector while preserving semantic meaning. Supports indexing,
iteration, broadcasting, and other vector operations.

# Type Parameters
- `T`: Element type of the parameter vector
- `Spec`: HyperparameterSpec type

# Fields
- `θ::Vector{T}`: Parameter values in working space
- `spec::Spec`: Hyperparameter specification

# Example
```julia
spec = HyperparameterSpec(
    free = (σ = Hyperparameter(Exponential(1.0), transform=elementwise(log), prior_space=:natural),)
)
θ_w = WorkingHyperparameters([0.5], spec)

# Vector-like operations
θ_w[1]           # indexing
θ_w .+ 0.1       # broadcasting
sum(θ_w)         # reduction
```
"""
struct WorkingHyperparameters{T, Spec} <: AbstractVector{T}
    θ::Vector{T}
    spec::Spec

    function WorkingHyperparameters(θ::Vector{T}, spec::HyperparameterSpec) where {T}
        n = _hp_total_dim(spec)
        if length(θ) != n
            error("Vector length ($(length(θ))) does not match the total dimension of the free parameters ($n)")
        end
        return new{T, typeof(spec)}(θ, spec)
    end
end

"""
    NaturalHyperparameters{T, Spec} <: AbstractVector{T}

Hyperparameters in natural (constrained) space.

Behaves like a vector while preserving semantic meaning. Supports indexing,
iteration, broadcasting, and other vector operations.

# Type Parameters
- `T`: Element type of the parameter vector
- `Spec`: HyperparameterSpec type

# Fields
- `θ::Vector{T}`: Parameter values in natural space
- `spec::Spec`: Hyperparameter specification

# Example
```julia
spec = HyperparameterSpec(
    free = (σ = Hyperparameter(Exponential(1.0), transform=elementwise(log), prior_space=:natural),)
)
θ_n = NaturalHyperparameters([2.0], spec)

# Vector-like operations
θ_n[1]           # indexing
θ_n .* 2         # broadcasting
maximum(θ_n)     # reduction
```
"""
struct NaturalHyperparameters{T, Spec} <: AbstractVector{T}
    θ::Vector{T}
    spec::Spec

    function NaturalHyperparameters(θ::Vector{T}, spec::HyperparameterSpec) where {T}
        n = _hp_total_dim(spec)
        if length(θ) != n
            error("Vector length ($(length(θ))) does not match the total dimension of the free parameters ($n)")
        end
        return new{T, typeof(spec)}(θ, spec)
    end
end

################################
##       ARRAY INTERFACE     ###
################################

# AbstractArray interface for WorkingHyperparameters
Base.size(θ::WorkingHyperparameters) = size(θ.θ)
Base.getindex(θ::WorkingHyperparameters, i::Int) = getindex(θ.θ, i)
Base.setindex!(θ::WorkingHyperparameters, v, i::Int) = setindex!(θ.θ, v, i)

# AbstractArray interface for NaturalHyperparameters
Base.size(θ::NaturalHyperparameters) = size(θ.θ)
Base.getindex(θ::NaturalHyperparameters, i::Int) = getindex(θ.θ, i)
Base.setindex!(θ::NaturalHyperparameters, v, i::Int) = setindex!(θ.θ, v, i)

# Broadcasting support - preserve type for same-size operations
function Base.similar(θ::WorkingHyperparameters, ::Type{S}, dims::Dims) where {S}
    if dims == size(θ)
        # Same size, return same type with new data
        return WorkingHyperparameters(similar(θ.θ, S), θ.spec)
    else
        # Different size, fall back to regular Array
        return similar(θ.θ, S, dims)
    end
end

function Base.similar(θ::NaturalHyperparameters, ::Type{S}, dims::Dims) where {S}
    if dims == size(θ)
        # Same size, return same type with new data
        return NaturalHyperparameters(similar(θ.θ, S), θ.spec)
    else
        # Different size, fall back to regular Array
        return similar(θ.θ, S, dims)
    end
end

# Custom broadcast styles to preserve type in broadcasting
struct WorkingHyperparametersStyle <: Broadcast.AbstractArrayStyle{1} end
struct NaturalHyperparametersStyle <: Broadcast.AbstractArrayStyle{1} end

Base.BroadcastStyle(::Type{<:WorkingHyperparameters}) = WorkingHyperparametersStyle()
Base.BroadcastStyle(::Type{<:NaturalHyperparameters}) = NaturalHyperparametersStyle()

# When broadcasting with scalars or other arrays, preserve our type
Base.BroadcastStyle(::WorkingHyperparametersStyle, ::Broadcast.DefaultArrayStyle{0}) = WorkingHyperparametersStyle()
Base.BroadcastStyle(::NaturalHyperparametersStyle, ::Broadcast.DefaultArrayStyle{0}) = NaturalHyperparametersStyle()

# Allocate output for broadcasting - this is the key method
function Base.similar(bc::Broadcast.Broadcasted{WorkingHyperparametersStyle}, ::Type{ElType}) where {ElType}
    # Find the WorkingHyperparameters in the broadcast arguments
    θ = find_hyperparameters(bc)
    return WorkingHyperparameters(similar(Vector{ElType}, axes(bc)), θ.spec)
end

function Base.similar(bc::Broadcast.Broadcasted{NaturalHyperparametersStyle}, ::Type{ElType}) where {ElType}
    # Find the NaturalHyperparameters in the broadcast arguments
    θ = find_hyperparameters(bc)
    return NaturalHyperparameters(similar(Vector{ElType}, axes(bc)), θ.spec)
end

# Helper to extract the hyperparameter object from broadcast arguments
find_hyperparameters(bc::Broadcast.Broadcasted) = find_hyperparameters(bc.args)
find_hyperparameters(args::Tuple) = find_hyperparameters(find_hyperparameters(args[1]), Base.tail(args))
find_hyperparameters(x) = x
find_hyperparameters(::Any, rest) = find_hyperparameters(rest)
find_hyperparameters(θ::Union{WorkingHyperparameters, NaturalHyperparameters}, rest) = θ

################################
##         CONVERSION        ###
################################

"""
    Base.convert(::Type{NaturalHyperparameters}, θ_working::WorkingHyperparameters)

Convert hyperparameters from working (unconstrained) space to natural (constrained) space.

# Example
```julia
spec = HyperparameterSpec(
    free = (σ = Hyperparameter(Exponential(1.0), transform=elementwise(log), prior_space=:natural),)
)
θ_w = WorkingHyperparameters([0.5], spec)  # log(σ) = 0.5
θ_n = convert(NaturalHyperparameters, θ_w)  # σ = exp(0.5) ≈ 1.649
```
"""
function Base.convert(::Type{NaturalHyperparameters}, θ_working::WorkingHyperparameters{T, Spec}) where {T, Spec}
    # Transform each parameter block from working to natural space
    θ_natural = _map_hp_blocks(θ_working.spec, θ_working.θ) do hp, working_value
        # Apply inverse: working → natural
        inverse(hp.transform)(working_value)
    end
    return NaturalHyperparameters(θ_natural, θ_working.spec)
end

"""
    Base.convert(::Type{WorkingHyperparameters}, θ_natural::NaturalHyperparameters)

Convert hyperparameters from natural (constrained) space to working (unconstrained) space.

# Example
```julia
spec = HyperparameterSpec(
    free = (σ = Hyperparameter(Exponential(1.0), transform=elementwise(log), prior_space=:natural),)
)
θ_n = NaturalHyperparameters([2.0], spec)  # σ = 2.0
θ_w = convert(WorkingHyperparameters, θ_n)  # log(σ) = log(2.0) ≈ 0.693
```
"""
function Base.convert(::Type{WorkingHyperparameters}, θ_natural::NaturalHyperparameters{T, Spec}) where {T, Spec}
    # Transform each parameter block from natural to working space
    θ_working = _map_hp_blocks(θ_natural.spec, θ_natural.θ) do hp, natural_value
        # Apply forward transformation: natural → working
        hp.transform(natural_value)
    end
    return WorkingHyperparameters(θ_working, θ_natural.spec)
end

"""
    Base.convert(::Type{NamedTuple}, θ::WorkingHyperparameters)

Convert WorkingHyperparameters to a NamedTuple containing both free parameters (in working space) and fixed parameters.

# Example
```julia
spec = HyperparameterSpec(
    free = (σ = Hyperparameter(Exponential(1.0), transform=elementwise(log), prior_space=:natural),),
    fixed = (μ = 0.0,)
)
θ_w = WorkingHyperparameters([0.5], spec)
nt = convert(NamedTuple, θ_w)  # (σ = 0.5, μ = 0.0)
```
"""
function Base.convert(::Type{NamedTuple}, θ::WorkingHyperparameters)
    # Build NamedTuple of free parameters in working space (scalar per
    # univariate entry, Vector per vector-valued entry)
    free_nt = _hp_blocks_namedtuple(θ.spec, getfield(θ, :θ))

    # Merge with fixed parameters
    return merge(free_nt, θ.spec.fixed)
end

"""
    Base.convert(::Type{NamedTuple}, θ::NaturalHyperparameters)

Convert NaturalHyperparameters to a NamedTuple containing both free parameters (in natural space) and fixed parameters.

# Example
```julia
spec = HyperparameterSpec(
    free = (σ = Hyperparameter(Exponential(1.0), transform=elementwise(log), prior_space=:natural),),
    fixed = (μ = 0.0,)
)
θ_n = NaturalHyperparameters([2.0], spec)
nt = convert(NamedTuple, θ_n)  # (σ = 2.0, μ = 0.0)
```
"""
function Base.convert(::Type{NamedTuple}, θ::NaturalHyperparameters)
    # Build NamedTuple of free parameters in natural space (scalar per
    # univariate entry, Vector per vector-valued entry)
    free_nt = _hp_blocks_namedtuple(θ.spec, getfield(θ, :θ))

    # Merge with fixed parameters
    return merge(free_nt, θ.spec.fixed)
end

# Per-name blocks of a flat parameter vector as a NamedTuple.
function _hp_blocks_namedtuple(spec::HyperparameterSpec, θ::AbstractVector)
    names = keys(spec.free)
    off = 0
    vals = map(names) do name
        hp = spec.free[name]
        d = _hp_dim(hp)
        v = _hp_isscalar(hp) ? θ[off + 1] : θ[(off + 1):(off + d)]
        off += d
        v
    end
    return NamedTuple{names}(vals)
end

################################
##        GETPROPERTY        ###
################################

"""
    Base.getproperty(θ::WorkingHyperparameters, name::Symbol)

Access hyperparameter values by name using dot notation.

Returns free parameter values (in working space) or fixed parameter values.

# Example
```julia
spec = HyperparameterSpec(
    free = (σ = Hyperparameter(Exponential(1.0), transform=elementwise(log), prior_space=:natural),),
    fixed = (μ = 0.0,)
)
θ_w = WorkingHyperparameters([0.5], spec)
θ_w.σ  # 0.5 (working space value)
θ_w.μ  # 0.0 (fixed value)
```
"""
function Base.getproperty(θ::WorkingHyperparameters, name::Symbol)
    # Access actual fields first (θ, spec)
    if name === :θ || name === :spec
        return getfield(θ, name)
    end

    # Look up in free parameters
    spec = getfield(θ, :spec)
    loc = _hp_locate(spec, name)
    if loc !== nothing
        start, d, isscalar = loc
        flat = getfield(θ, :θ)
        return isscalar ? flat[start] : flat[start:(start + d - 1)]
    end

    # Look up in fixed parameters
    if haskey(spec.fixed, name)
        return spec.fixed[name]
    end

    # Not found
    error("type WorkingHyperparameters has no field $name")
end

"""
    Base.getproperty(θ::NaturalHyperparameters, name::Symbol)

Access hyperparameter values by name using dot notation.

Returns free parameter values (in natural space) or fixed parameter values.

# Example
```julia
spec = HyperparameterSpec(
    free = (σ = Hyperparameter(Exponential(1.0), transform=elementwise(log), prior_space=:natural),),
    fixed = (μ = 0.0,)
)
θ_n = NaturalHyperparameters([2.0], spec)
θ_n.σ  # 2.0 (natural space value)
θ_n.μ  # 0.0 (fixed value)
```
"""
function Base.getproperty(θ::NaturalHyperparameters, name::Symbol)
    # Access actual fields first (θ, spec)
    if name === :θ || name === :spec
        return getfield(θ, name)
    end

    # Look up in free parameters
    spec = getfield(θ, :spec)
    loc = _hp_locate(spec, name)
    if loc !== nothing
        start, d, isscalar = loc
        flat = getfield(θ, :θ)
        return isscalar ? flat[start] : flat[start:(start + d - 1)]
    end

    # Look up in fixed parameters
    if haskey(spec.fixed, name)
        return spec.fixed[name]
    end

    # Not found
    error("type NaturalHyperparameters has no field $name")
end

# Locate free parameter `name` in the flat layout without allocating:
# `(start, dim, isscalar)`, or `nothing` when `name` is not a free parameter.
function _hp_locate(spec::HyperparameterSpec, name::Symbol)
    off = 0
    for k in keys(spec.free)
        hp = spec.free[k]
        d = _hp_dim(hp)
        k === name && return (off + 1, d, _hp_isscalar(hp))
        off += d
    end
    return nothing
end

################################
##         LOGDETJAC         ###
################################

"""
    logdetjac(θ::WorkingHyperparameters) -> Float64

Compute the log absolute determinant of the Jacobian for the working → natural transformation.

This is the correction term to add when converting a density from working space to natural space:
```
log p_natural(θ) = log p_working(η) + logdetjac(θ_working)
```
where η are working values, θ = g(η) are the corresponding natural values.

# Example
```julia
spec = HyperparameterSpec(
    free = (σ = Hyperparameter(Exponential(1.0), transform=elementwise(log), prior_space=:natural),)
)
θ_w = WorkingHyperparameters([0.5], spec)

# Have a density in working space, want natural space
log_density_working = some_function(θ_w)
log_density_natural = log_density_working + logdetjac(θ_w)
```
"""
function logdetjac(θ::WorkingHyperparameters)
    return _sum_hp_blocks(θ.spec, θ.θ) do hp, working_value
        # Jacobian of working → natural transformation
        # This is logabsdetjac of inverse(transform)
        Bijectors.logabsdetjac(inverse(hp.transform), working_value)
    end
end

"""
    logdetjac(θ::NaturalHyperparameters) -> Float64

Compute the log absolute determinant of the Jacobian for the natural → working transformation.

This is the correction term to add when converting a density from natural space to working space:
```
log p_working(η) = log p_natural(θ) + logdetjac(θ_natural)
```
where θ are natural values, η = f(θ) are the corresponding working values.

# Example
```julia
spec = HyperparameterSpec(
    free = (σ = Hyperparameter(Exponential(1.0), transform=elementwise(log), prior_space=:natural),)
)
θ_n = NaturalHyperparameters([2.0], spec)

# Prior is in natural space, want working space density
log_prior_natural = logpdf(Exponential(1.0), θ_n.σ)
log_density_working = log_prior_natural + logdetjac(θ_n)
```
"""
function logdetjac(θ::NaturalHyperparameters)
    return _sum_hp_blocks(θ.spec, θ.θ) do hp, natural_value
        # Jacobian of natural → working transformation
        Bijectors.logabsdetjac(hp.transform, natural_value)
    end
end

# Sum `f(hp, block)` over the free blocks of a flat parameter vector. Vector
# blocks are passed as views; elementwise transforms return a summed scalar
# for them. Starts from 0.0 and lets the accumulator promote (e.g. to Dual).
function _sum_hp_blocks(f, spec::HyperparameterSpec, θ::AbstractVector)
    total = 0.0
    off = 0
    for name in keys(spec.free)
        hp = spec.free[name]
        d = _hp_dim(hp)
        v = _hp_isscalar(hp) ? θ[off + 1] : view(θ, (off + 1):(off + d))
        total += f(hp, v)
        off += d
    end
    return total
end

################################
##           SHOW            ###
################################

"""
    Base.show(io::IO, ::MIME"text/plain", θ::WorkingHyperparameters)

Pretty printing for WorkingHyperparameters objects in REPL.
"""
function Base.show(io::IO, ::MIME"text/plain", θ::WorkingHyperparameters{T, Spec}) where {T, Spec}
    n_params = length(θ.θ)
    println(io, "WorkingHyperparameters{$T} with $n_params parameters:")

    for (name, value) in pairs(_hp_blocks_namedtuple(θ.spec, getfield(θ, :θ)))
        println(io, "  $name = $value")
    end

    return nothing
end

"""
    Base.show(io::IO, ::MIME"text/plain", θ::NaturalHyperparameters)

Pretty printing for NaturalHyperparameters objects in REPL.
"""
function Base.show(io::IO, ::MIME"text/plain", θ::NaturalHyperparameters{T, Spec}) where {T, Spec}
    n_params = length(θ.θ)
    println(io, "NaturalHyperparameters{$T} with $n_params parameters:")

    for (name, value) in pairs(_hp_blocks_namedtuple(θ.spec, getfield(θ, :θ)))
        println(io, "  $name = $value")
    end

    return nothing
end
