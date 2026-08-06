using Distributions
using Bijectors
using Printf

export Hyperparameter

"""
    Hyperparameter{T, S}

A hyperparameter specification with an explicit transformation between working and natural spaces.

# Type Parameters
- `T`: The transformation type (Bijector or identity function)
- `S`: The space in which the prior was originally specified (`:natural` or `:working`)

# Fields
- `prior::Distribution`: Prior distribution in working space (always stored in working space for numerical stability)
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
    prior::Distribution          # Always stored in working space
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
The prior is always stored internally in working space for numerical stability. When `prior_space=:natural`,
the prior is automatically transformed to working space using `transformed(prior, transform)`.

The prior may be a continuous multivariate (vector) distribution, making this a
vector-valued hyperparameter whose components share the joint prior. Vector
entries require a dimension-preserving elementwise transform — `identity` or
`elementwise(f)` — so working and natural space share one flat layout;
dimension-changing bijectors are rejected.

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
    _validate_hyperparameter_prior(prior, transform)

    if prior_space == :natural
        # Prior in natural space, transform to working space
        # natural → working uses transform
        if transform === identity
            # Identity transform: working = natural space
            return Hyperparameter(prior, transform, Val(:natural))
        end
        working_prior = transformed(prior, transform)
        return Hyperparameter(working_prior, transform, Val(:natural))
    elseif prior_space == :working
        # Prior already in working space, store as-is
        return Hyperparameter(prior, transform, Val(:working))
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

# Vector-valued hyperparameters (multivariate priors) share one flat layout
# between working and natural space, so their transforms must be
# dimension-preserving and act elementwise: `identity` or
# `Bijectors.elementwise(f)` (`Base.Fix1{typeof(broadcast)}`).
# Dimension-changing bijectors (simplex, Cholesky, ...) are rejected here
# rather than failing obscurely in space conversion or marginal extraction.
function _validate_hyperparameter_prior(prior::Distribution, transform)
    prior isa Distribution{Univariate} && return nothing
    prior isa Distribution{Multivariate} || throw(
        ArgumentError(
            "Hyperparameter priors must be univariate or multivariate " *
                "(vector-valued); got a $(typeof(prior))."
        )
    )
    _is_elementwise_transform(transform) || throw(
        ArgumentError(
            "Vector-valued hyperparameters support only dimension-preserving " *
                "elementwise transforms — `identity` or `elementwise(f)` (e.g. " *
                "`elementwise(log)`); got $(typeof(transform)). Dimension-changing " *
                "bijectors such as the simplex bijector are not supported."
        )
    )
    return nothing
end

_is_elementwise_transform(::typeof(identity)) = true
_is_elementwise_transform(::Base.Fix1{typeof(broadcast)}) = true
_is_elementwise_transform(::Any) = false
