using GaussianMarkovRandomFields
using GaussianMarkovRandomFields: LatentModel, hyperparameters, precision_matrix, constraints, model_name
using LinearAlgebra
using LinearSolve
using SparseArrays
using Distributions

import GaussianMarkovRandomFields: hyperparameters, precision_matrix, constraints, model_name

export AugmentedLatentModel, linear_predictor_indices, base_latent_indices

"""
    AugmentedLatentModel <: LatentModel

A LatentModel wrapper that automatically augments a base latent model with linear predictor
components. This enables computation of marginals for linear predictors η using full INLA
approximations when using `LinearlyTransformedObservationModel`.

# Mathematical Structure

The augmented model maintains the linear relationship η ≈ A * x_base through a joint Gaussian
construction. For a base latent field x_base ~ N(μ_base, Q_base^(-1)) and design matrix A:

**Augmented field**: x_full = [η; x_base] where:
- η: Linear predictors (size n_obs)
- x_base: Base latent components (size n_base)

**Joint distribution**: Using the linear-Gaussian formula:

Mean:
```
μ_joint = [A * μ_base; μ_base]
```

Precision (encodes η ≈ A * x_base):
```
Q_joint = [ Q_η                  -Q_η * A              ;
           -A' * Q_η    Q_base + A' * Q_η * A ]
```

where Q_η controls the tightness of the linear relationship (large Q_η → tight coupling).

# Fields
- `base_model::LatentModel`: The base latent model (returns GMRFs of size n_base)
- `design_matrix::AbstractMatrix`: Design matrix A (n_obs × n_base)
- `linear_predictor_precision::Real`: Precision Q_η for enforcing η ≈ A * x_base

# Example
```julia
# Base model for latent components
base_model = AR1Model(100)  # 100 base latent components

# Design matrix: 200 linear predictors from 100 base components
A = randn(200, 100)

# Augmented model (tight coupling with high precision)
augmented = AugmentedLatentModel(base_model, A; linear_predictor_precision=1e6)

# Use like any LatentModel
gmrf = augmented(τ=2.0, ρ=0.9)  # Returns GMRF of size 300 = [η₁...η₂₀₀; x₁...x₁₀₀]
```
"""
struct AugmentedLatentModel{M <: LatentModel, A <: AbstractMatrix, Alg} <: LatentModel
    base_model::M
    design_matrix::A
    linear_predictor_precision::Real
    alg::Alg

    function AugmentedLatentModel(
            base_model::M,
            design_matrix::A;
            linear_predictor_precision::Real = 1.0e6,
            alg = LinearSolve.CHOLMODFactorization()
        ) where {M <: LatentModel, A <: AbstractMatrix}
        n_obs, n_base = size(design_matrix)

        # Validate dimensions
        n_base == length(base_model) ||
            throw(
            DimensionMismatch(
                "Design matrix has $n_base columns but base_model has dimension $(length(base_model))"
            )
        )

        linear_predictor_precision > 0 ||
            throw(
            ArgumentError(
                "linear_predictor_precision must be positive, got $linear_predictor_precision"
            )
        )

        return new{M, A, typeof(alg)}(base_model, design_matrix, linear_predictor_precision, alg)
    end
end

"""
    length(model::AugmentedLatentModel)

Returns the total augmented dimension: n_obs + n_base.
"""
function Base.length(model::AugmentedLatentModel)
    n_obs, n_base = size(model.design_matrix)
    return n_obs + n_base
end

"""
    hyperparameters(model::AugmentedLatentModel)

Returns the hyperparameters from the base model.
The linear predictor precision is fixed at construction time.
"""
function hyperparameters(model::AugmentedLatentModel)
    return hyperparameters(model.base_model)
end

"""
    precision_matrix(model::AugmentedLatentModel; kwargs...)

Constructs the joint precision matrix encoding the linear relationship η ≈ A * x_base.

Using the linear-Gaussian formula from joint_impl_notes.org:

```
Q_joint = [ Q_η                  -Q_η * A              ;
           -A' * Q_η    Q_base + A' * Q_η * A ]
```

where:
- Q_η = linear_predictor_precision * I(n_obs) (controls tightness of η ≈ A * x_base)
- Q_base = precision_matrix(base_model; kwargs...)
- A = design_matrix

The off-diagonal blocks couple η and x_base, encoding their linear relationship.
"""
function precision_matrix(model::AugmentedLatentModel; kwargs...)
    A = model.design_matrix
    n_obs, n_base = size(A)
    Q_η_scalar = model.linear_predictor_precision

    # Precision for base latent components
    Q_base = precision_matrix(model.base_model; kwargs...)

    # Precision matrix for linear predictors (scalar * identity)
    Q_η = Q_η_scalar * I(n_obs)

    # Construct joint precision using linear-Gaussian formula:
    # Block structure:
    #   [ Q_η           -Q_η * A          ]
    #   [ -A' * Q_η     Q_base + A' * Q_η * A ]

    # Compute blocks
    # TODO: Efficiency!!! Only compute one of the off-diagonal blocks
    top_left = Q_η
    top_right = -Q_η_scalar * A  # -Q_η * A
    bottom_left = -Q_η_scalar * A'  # -A' * Q_η  (transpose of top_right)
    bottom_right = Q_base + Q_η_scalar * (A' * A)  # Q_base + A' * Q_η * A

    # Assemble sparse matrix
    return _sparse_block_matrix(top_left, top_right, bottom_left, bottom_right)
end

"""
    mean(model::AugmentedLatentModel; kwargs...)

Constructs the joint mean vector encoding E[η] = A * E[x_base]:

```
μ_joint = [A * μ_base; μ_base]
```

This ensures the linear predictors have the correct prior mean.
"""
function Distributions.mean(model::AugmentedLatentModel; kwargs...)
    A = model.design_matrix
    μ_base = mean(model.base_model; kwargs...)

    # Linear predictors have mean A * μ_base
    μ_η = A * μ_base

    # Joint mean
    return vcat(μ_η, μ_base)
end

"""
    constraints(model::AugmentedLatentModel; kwargs...)

Returns constraints from the base model, offset to account for prepended linear predictors.

If base model has constraints (A_base, e_base), returns (A_augmented, e_base) where:
- A_augmented = [zeros(n_constraints, n_obs)  A_base]

This ensures constraints only apply to the base latent components.
"""
function constraints(model::AugmentedLatentModel; kwargs...)
    base_constraints = constraints(model.base_model; kwargs...)

    # If no constraints in base model, nothing to augment
    base_constraints === nothing && return nothing

    A_base, e_base = base_constraints
    n_constraints = size(A_base, 1)
    n_obs = size(model.design_matrix, 1)

    # Augment constraint matrix with zeros for linear predictor components
    # Structure: [0_{n_constraints × n_obs}  A_base]
    A_augmented = hcat(zeros(eltype(A_base), n_constraints, n_obs), A_base)

    return (A_augmented, e_base)
end

"""
    model_name(::AugmentedLatentModel)

Returns the model name for parameter prefixing.
"""
function model_name(model::AugmentedLatentModel)
    base_name = model_name(model.base_model)
    return Symbol("augmented_$(base_name)")
end

"""
    linear_predictor_indices(model::AugmentedLatentModel)

Returns the indices corresponding to linear predictor components η in the augmented field.
"""
function linear_predictor_indices(model::AugmentedLatentModel)
    n_obs = size(model.design_matrix, 1)
    return 1:n_obs
end

"""
    base_latent_indices(model::AugmentedLatentModel)

Returns the indices corresponding to base latent components x_base in the augmented field.
"""
function base_latent_indices(model::AugmentedLatentModel)
    n_obs = size(model.design_matrix, 1)
    n_full = length(model)
    return (n_obs + 1):n_full
end

# Helper function to construct sparse block matrix from four blocks
function _sparse_block_matrix(top_left, top_right, bottom_left, bottom_right)
    # TODO: Efficiency!!! Sparse matrix should be assembled via Is, Js, Vs here
    n1, m1 = size(top_left)
    n2, m2 = size(bottom_right)

    # Build by stacking rows
    top_row = hcat(top_left, top_right)
    bottom_row = hcat(bottom_left, bottom_right)
    full_matrix = vcat(top_row, bottom_row)

    # Convert to sparse if not already
    return sparse(full_matrix)
end

function Base.show(io::IO, model::AugmentedLatentModel)
    n_obs, n_base = size(model.design_matrix)
    n_full = length(model)
    Q_η = model.linear_predictor_precision

    print(io, "AugmentedLatentModel")
    print(io, "\n  Base model: ", model.base_model)
    print(io, "\n  Design matrix: $(n_obs) × $(n_base)")
    print(io, "\n  Structure: x_full = [η₁...η_$(n_obs); x_base₁...x_base_$(n_base)]")
    print(io, "\n  Total dimension: $n_full ($(n_obs) linear predictors + $(n_base) base components)")
    print(io, "\n  Linear relationship: η ≈ A * x_base (Q_η = $(Q_η))")
    return
end
