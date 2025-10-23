using Distributions
using Bijectors
using Printf

export Hyperparameter, HyperparameterSpec
export working_to_natural, to_working, logpdf_prior, to_named_tuple, to_vector

using Bijectors: elementwise
export elementwise

"""
    Hyperparameter{T, S}

A hyperparameter specification with an explicit transformation between working and natural spaces.

# Type Parameters
- `T`: The transformation type (Bijector or identity function)
- `S`: The space in which the prior was originally specified (`:natural` or `:working`)

# Fields
- `prior::Distribution`: Prior distribution in working space (after transformation if needed)
- `transform::T`: Bijector mapping natural → working (constrained → unconstrained)

# Transformation Convention
Follows Bijectors.jl/Turing.jl convention:
- `transform`: natural (constrained) → working (unconstrained)
- `inverse(transform)`: working (unconstrained) → natural (constrained)

For example, for σ ∈ (0,∞):
- Natural space: σ > 0
- Working space: log(σ) ∈ ℝ
- Transform: `elementwise(log)` maps σ → log(σ)

# Spaces
- **Working space (η)**: Unconstrained space used for optimization/exploration
- **Natural space (θ)**: Natural parameter space used in user model functions

# Example
```julia
using Bijectors

# Used within a HyperparameterSpec, not standalone
spec = HyperparameterSpec(
    free = (
        σ = Hyperparameter(Exponential(1.0), elementwise(log), Val(:natural)),
        ρ = Hyperparameter(Beta(2, 2), Bijectors.Logit(0.0, 1.0), Val(:natural))
    ),
    fixed = (μ = 0.0,)
)
```
"""
struct Hyperparameter{T, S}
    prior::Distribution          # Always stored in natural space
    transform::T                 # natural → working transformation

    function Hyperparameter(prior::Distribution, transform::T, ::Val{S}) where {T, S}
        if S ∉ (:natural, :working)
            error("Prior space must be :natural or :working, got :$S")
        end
        return new{T, S}(prior, transform)
    end
end

"""
    Hyperparameter(prior::Distribution; transform=identity, prior_space=:working)

Construct a Hyperparameter with automatic prior space conversion.

# Arguments
- `prior::Distribution`: Prior distribution
- `transform`: Bijector mapping natural → working (constrained → unconstrained), default: `identity`
- `prior_space::Symbol`: Space in which prior is specified (`:natural` or `:working`, default: `:working`)

# Details
The prior is always stored internally in natural space. When `prior_space=:working`, the prior
is automatically transformed back to natural space using `transformed(prior, inverse(transform))`.

# Examples
```julia
using Bijectors

# Used within a HyperparameterSpec:
spec = HyperparameterSpec(
    free = (
        σ = Hyperparameter(Exponential(1.0), transform=elementwise(log), prior_space=:natural),
        ρ = Hyperparameter(Beta(2, 2), transform=Bijectors.Logit(0.0, 1.0), prior_space=:natural),
        μ = Hyperparameter(Normal(0, 10), transform=identity, prior_space=:working)
    ),
    fixed = NamedTuple()
)
```
"""
function Hyperparameter(
        prior::Distribution;
        transform = identity,
        prior_space::Symbol = :working
    )

    if prior_space == :natural
        # Prior already in natural space, store as-is
        return Hyperparameter(prior, transform, Val(:natural))
    elseif prior_space == :working
        # Special case: identity transform means working = natural space
        if transform === identity
            return Hyperparameter(prior, transform, Val(:working))
        end
        # Prior in working space, transform back to natural space
        # working → natural uses inverse(transform)
        natural_prior = transformed(prior, inverse(transform))
        return Hyperparameter(natural_prior, transform, Val(:working))
    else
        error("prior_space must be :natural or :working, got :$prior_space")
    end
end

"""
    prior_space(hp::Hyperparameter{T, S}) where {T, S}

Extract the prior space from a Hyperparameter's type.

Returns `:natural` or `:working`.
"""
prior_space(::Hyperparameter{T, S}) where {T, S} = S

"""
    HyperparameterSpec{Free, Fixed}

Complete specification of hyperparameters with both free and fixed parameters.

# Type Parameters
- `Free`: Concrete NamedTuple type for free parameters
- `Fixed`: Concrete NamedTuple type for fixed parameter values

# Fields
- `free::Free`: Free parameters to be estimated (NamedTuple of Hyperparameter objects)
- `fixed::Fixed`: Fixed parameter values (NamedTuple of scalar values)

# Example
```julia
using Bijectors

spec = HyperparameterSpec(
    free = (
        σ = Hyperparameter(Exponential(1.0), transform=elementwise(log), prior_space=:natural),
        ρ = Hyperparameter(Beta(2, 2), transform=Bijectors.Logit(0.0, 1.0), prior_space=:natural)
    ),
    fixed = (μ = 0.0,)
)
```
"""
struct HyperparameterSpec{Free, Fixed}
    free::Free
    fixed::Fixed

    function HyperparameterSpec(; free::NamedTuple, fixed::NamedTuple = NamedTuple())
        # Validate: must have at least one free parameter
        if isempty(keys(free))
            error("INLA requires at least one free hyperparameter. All-fixed hyperparameter specs are not supported.")
        end

        # Validate: no overlap between free and fixed
        free_names = keys(free)
        fixed_names = keys(fixed)
        overlap = intersect(Set(free_names), Set(fixed_names))
        if !isempty(overlap)
            error("Parameters cannot be both free and fixed: $(collect(overlap))")
        end

        return new{typeof(free), typeof(fixed)}(free, fixed)
    end
end

"""
    working_to_natural(θ_working::NamedTuple, spec::HyperparameterSpec) -> NamedTuple

Transform hyperparameters from working space to natural space.

# Arguments
- `θ_working::NamedTuple`: Free parameters in working space (unconstrained)
- `spec::HyperparameterSpec`: Hyperparameter specification

# Returns
- `NamedTuple`: All parameters (free + fixed) in natural space

# Example
```julia
spec = HyperparameterSpec(
    free = (σ = Hyperparameter(Exponential(1.0), transform=elementwise(log), prior_space=:natural),),
    fixed = (μ = 0.0,)
)

θ_working = (σ = -0.5,)  # log(σ) in working space
θ_natural = working_to_natural(θ_working, spec)  # (σ = exp(-0.5) ≈ 0.606, μ = 0.0)
```
"""
function working_to_natural(θ_working::NamedTuple, spec::HyperparameterSpec)
    # Transform free parameters from working to natural space
    θ_free_natural = map(keys(spec.free), values(spec.free)) do name, hp
        working_value = θ_working[name]
        # Apply inverse: working → natural
        inverse(hp.transform)(working_value)
    end
    θ_free_natural_nt = NamedTuple{keys(spec.free)}(θ_free_natural)

    # Merge with fixed parameters
    return merge(θ_free_natural_nt, spec.fixed)
end

function working_to_natural(θ_working::AbstractVector, spec::HyperparameterSpec)
    return working_to_natural(to_named_tuple(θ_working, spec), spec)
end

"""
    to_working(θ_natural::NamedTuple, spec::HyperparameterSpec) -> NamedTuple

Transform hyperparameters from natural space to working space.

# Arguments
- `θ_natural::NamedTuple`: Parameters in natural space (constrained)
- `spec::HyperparameterSpec`: Hyperparameter specification

# Returns
- `NamedTuple`: Free parameters in working space (unconstrained)

# Example
```julia
spec = HyperparameterSpec(
    free = (σ = Hyperparameter(Exponential(1.0), transform=elementwise(log), prior_space=:natural),),
    fixed = (μ = 0.0,)
)

θ_natural = (σ = 2.0, μ = 0.0)
θ_working = to_working(θ_natural, spec)  # (σ = log(2.0) ≈ 0.693,)
```
"""
function to_working(θ_natural::NamedTuple, spec::HyperparameterSpec)
    # Only transform free parameters (fixed parameters are not in working space)
    working_values = map(keys(spec.free), values(spec.free)) do name, hp
        natural_value = θ_natural[name]
        # Apply forward transformation: natural → working
        hp.transform(natural_value)
    end

    return NamedTuple{keys(spec.free)}(working_values)
end

"""
    logpdf_prior(θ_natural::NamedTuple, spec::HyperparameterSpec) -> Float64

Evaluate the log prior density in natural space (no Jacobian correction).

# Arguments
- `θ_natural::NamedTuple`: Free parameters in natural space
- `spec::HyperparameterSpec`: Hyperparameter specification

# Returns
- `Float64`: Log prior density in natural space

# Details
The prior distributions in `spec.free` are stored in natural space.
This function evaluates the joint log prior density by summing the individual log prior densities.
No Jacobian correction is applied - this evaluates π(θ) directly.

# Example
```julia
spec = HyperparameterSpec(
    free = (σ = Hyperparameter(Exponential(1.0), transform=elementwise(log), prior_space=:natural),)
)

θ_natural = (σ = 2.0,)  # σ in natural space
log_p = logpdf_prior(θ_natural, spec)  # Evaluates log π(σ) without Jacobian
```
"""
function logpdf_prior(θ_natural::NamedTuple, spec::HyperparameterSpec)
    # Use mapreduce for better type stability
    return mapreduce(+, pairs(spec.free); init = 0.0) do (name, hp)
        natural_value = θ_natural[name]
        # The prior is stored in natural space
        logpdf(hp.prior, natural_value)::Float64
    end
end

"""
    to_named_tuple(θ_vec::Vector, spec::HyperparameterSpec) -> NamedTuple

Convert a vector of free parameters to a NamedTuple in working space.

# Arguments
- `θ_vec::Vector`: Free parameter values as a vector
- `spec::HyperparameterSpec`: Hyperparameter specification

# Returns
- `NamedTuple`: Free parameters with names, in working space

# Example
```julia
spec = HyperparameterSpec(
    free = (
        σ = Hyperparameter(Exponential(1.0), transform=elementwise(log), prior_space=:natural),
        ρ = Hyperparameter(Beta(2, 2), transform=Bijectors.Logit(0.0, 1.0), prior_space=:natural)
    )
)

θ_vec = [-0.5, 0.2]
θ_nt = to_named_tuple(θ_vec, spec)  # (σ = -0.5, ρ = 0.2) in working space
```
"""
function to_named_tuple(θ_vec::Vector, spec::HyperparameterSpec{Free, Fixed}) where {Free, Fixed}
    # Extract names from the Free type parameter at compile time
    free_names = fieldnames(Free)

    if length(θ_vec) != length(free_names)
        error("Vector length ($(length(θ_vec))) does not match number of free parameters ($(length(free_names)))")
    end

    θ_free = NamedTuple{free_names}(θ_vec)

    # Merge with fixed parameters
    return merge(θ_free, spec.fixed)
end

"""
    to_vector(θ_nt::NamedTuple, spec::HyperparameterSpec) -> Vector{Float64}

Convert a NamedTuple of free parameters to a vector.

# Arguments
- `θ_nt::NamedTuple`: Free parameters with names
- `spec::HyperparameterSpec`: Hyperparameter specification

# Returns
- `Vector{Float64}`: Free parameter values as a vector

# Example
```julia
spec = HyperparameterSpec(
    free = (
        σ = Hyperparameter(Exponential(1.0), transform=elementwise(log), prior_space=:natural),
        ρ = Hyperparameter(Beta(2, 2), transform=Bijectors.Logit(0.0, 1.0), prior_space=:natural)
    )
)

θ_nt = (σ = -0.5, ρ = 0.2)
θ_vec = to_vector(θ_nt, spec)  # [-0.5, 0.2]
```
"""
function to_vector(θ_nt::NamedTuple, spec::HyperparameterSpec)
    # Extract values in the order specified by keys(spec.free)
    return [θ_nt[name] for name in keys(spec.free)]
end

"""
    Base.show(io::IO, hp::Hyperparameter)

Pretty printing for Hyperparameter objects.
"""
function Base.show(io::IO, hp::Hyperparameter{T, S}) where {T, S}
    space_str = S == :natural ? "natural space" : "working space"
    transform_str = hp.transform === identity ? "identity" : string(typeof(hp.transform))

    return print(io, "Hyperparameter($(hp.prior) via $(transform_str), prior in $(space_str))")
end

"""
    Base.show(io::IO, spec::HyperparameterSpec)

Pretty printing for HyperparameterSpec objects.
"""
function Base.show(io::IO, spec::HyperparameterSpec)
    n_free = length(keys(spec.free))
    n_fixed = length(keys(spec.fixed))
    n_total = n_free + n_fixed

    println(io, "HyperparameterSpec with $n_total parameters:")

    # Show free parameters
    println(io, "  Free parameters ($n_free):")
    for (name, hp) in pairs(spec.free)
        transform_name = hp.transform === identity ? "identity" : string(typeof(hp.transform).name.name)
        space_indicator = prior_space(hp) == :natural ? "ⁿ" : "ʷ"
        println(io, "    $name ~ $(hp.prior) via $(transform_name)$(space_indicator)")
    end

    # Show fixed parameters
    if !isempty(spec.fixed)
        println(io, "  Fixed parameters ($n_fixed):")
        for (name, value) in pairs(spec.fixed)
            println(io, "    $name = $value")
        end
    end

    return nothing
end
