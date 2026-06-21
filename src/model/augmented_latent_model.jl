using GaussianMarkovRandomFields
using GaussianMarkovRandomFields: LatentModel, hyperparameters, precision_matrix, constraints, model_name
using LinearAlgebra
using LinearSolve
using SparseArrays
using Distributions

import GaussianMarkovRandomFields: hyperparameters, precision_matrix, constraints, model_name

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
struct AugmentedLatentModel{M <: LatentModel, A <: AbstractMatrix, O, Alg} <: LatentModel
    base_model::M
    design_matrix::A
    offset::O
    linear_predictor_precision::Real
    alg::Alg

    function AugmentedLatentModel(
            base_model::M,
            design_matrix::A;
            offset = nothing,
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

        # A fixed offset must match the linear-predictor count; a θ-dependent
        # ParameterizedOffset's length is checked at resolution time.
        _validate_aug_offset(offset, n_obs)

        linear_predictor_precision > 0 ||
            throw(
            ArgumentError(
                "linear_predictor_precision must be positive, got $linear_predictor_precision"
            )
        )

        # Warn on poorly-scaled designs. With an intrinsic base prior the
        # augmentation can become severely ill-conditioned: column-norm
        # spread `>~ 1e3` is enough to push `Q_joint`'s condition number
        # past Float64's reliable Cholesky range, manifesting downstream
        # as `PosDefException`. Centring/standardising the offending
        # covariates is the standard mitigation.
        col_norms = map(j -> norm(view(design_matrix, :, j)), 1:n_base)
        nonzero_norms = filter(>(0), col_norms)
        if !isempty(nonzero_norms) &&
                maximum(nonzero_norms) / minimum(nonzero_norms) > 1.0e3
            @warn (
                "AugmentedLatentModel: design-matrix column norms span >1e3 " *
                    "(min=$(round(minimum(nonzero_norms), sigdigits = 3)), " *
                    "max=$(round(maximum(nonzero_norms), sigdigits = 3))). " *
                    "Sparse Cholesky on `Q_joint` may fail or be unstable. " *
                    "Consider centring/standardising covariates."
            )
        end

        return new{M, A, typeof(offset), typeof(alg)}(base_model, design_matrix, offset, linear_predictor_precision, alg)
    end
end

# The augmented linear predictor carries an optional offset b: η ≈ A·x_base + b.
# It enters purely as a mean shift (the precision is offset-invariant), so a
# θ-dependent offset makes the augmented prior MEAN θ-dependent — which the
# forward-mode IFT handles on the prior side. `nothing` ⇒ no offset (zero
# overhead, the pre-offset behaviour).
_validate_aug_offset(::Nothing, ::Int) = nothing
_validate_aug_offset(::GaussianMarkovRandomFields.ParameterizedOffset, ::Int) = nothing
function _validate_aug_offset(b::AbstractVector, n_obs::Int)
    length(b) == n_obs || throw(
        DimensionMismatch(
            "offset has length $(length(b)) but the design matrix has $n_obs rows (linear predictors)"
        )
    )
    return nothing
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
    # `hyperparameters(::LatentModel)` is a NamedTuple (name => type). A
    # θ-dependent offset adds its declared hyperparameters (consumed by the
    # mean) to that set, typed generically as `Real`.
    base = hyperparameters(model.base_model)
    off = GaussianMarkovRandomFields._offset_hp_names(model.offset)
    new_names = Tuple(s for s in off if !(s in keys(base)))
    isempty(new_names) && return base
    return merge(base, NamedTuple{new_names}(ntuple(_ -> Real, length(new_names))))
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

    # Compute blocks efficiently
    # Only compute off-diagonal block once (bottom_left = top_right')
    top_left = Q_η
    top_right = -Q_η_scalar * A  # -Q_η * A
    bottom_right = Q_base + Q_η_scalar * (A' * A)  # Q_base + A' * Q_η * A

    # Assemble sparse matrix
    Q_joint = _sparse_block_matrix(top_left, top_right, bottom_right)
    # Diagonal regularisation matching `RWModel`'s `regularization=1e-5`,
    # applied only when the base prior is intrinsic (has constraints — e.g.
    # RW1, RW2, Besag). For those, the augmentation lifts the rank-deficiency
    # through the design matrix into directions the constraint correction
    # can't reach, leaving Q_joint genuinely rank-deficient. The shift is
    # small enough that user priors at typical scales (e.g. diffuse Normal(0,
    # 100²) on fixed effects → diag entries 1e-4) are essentially untouched,
    # but large enough for sparse Cholesky to factor. Skipped for proper
    # base priors so the fast-path / AD-fallback agreement is preserved.
    if constraints(model.base_model; kwargs...) !== nothing
        n = size(Q_joint, 1)
        Q_joint = Q_joint + 1.0e-5 * sparse(I, n, n)
    end
    return Q_joint
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

    # Linear predictors have mean A * μ_base (+ offset b if present). The
    # offset enters here only — a θ-dependent b yields a θ-dependent prior
    # mean, leaving the precision untouched.
    μ_η = A * μ_base
    b = GaussianMarkovRandomFields._resolve_offset(model.offset, (; kwargs...))
    if b !== nothing
        μ_η = μ_η .+ b
    end

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

# Helper function to construct sparse block matrix from three blocks
# Bottom-left is the transpose of top-right
function _sparse_block_matrix(top_left, top_right, bottom_right)
    n_obs, n_obs2 = size(top_left)
    n_obs3, n_base = size(top_right)
    n_base2, n_base3 = size(bottom_right)

    @assert n_obs == n_obs2 == n_obs3
    @assert n_base == n_base2 == n_base3

    n_total = n_obs + n_base

    # Assemble efficiently via coordinate format (I, J, V)
    T = promote_type(eltype(top_left), eltype(top_right), eltype(bottom_right))
    Is = Int[]
    Js = Int[]
    Vs = T[]

    # Top-left block (Q_η): diagonal matrix
    # Q_η = Q_η_scalar * I
    if top_left isa UniformScaling
        Q_η_scalar = top_left.λ
        top_left_idcs = 1:n_obs
        append!(Is, top_left_idcs)
        append!(Js, top_left_idcs)
        append!(Vs, fill(Q_η_scalar, n_obs))
    else
        # General case: extract nonzeros
        I_tl, J_tl, V_tl = findnz(sparse(top_left))
        append!(Is, I_tl)
        append!(Js, J_tl)
        append!(Vs, V_tl)
    end

    # Top-right block: -Q_η * A
    I_tr, J_tr, V_tr = findnz(sparse(top_right))
    append!(Is, I_tr)
    append!(Js, J_tr .+ n_obs)  # Offset columns
    append!(Vs, V_tr)

    # Bottom-left block: transpose of top-right
    # (i, j) in top_right maps to (j + n_obs, i) in bottom_left
    append!(Is, J_tr .+ n_obs)  # Row = col of top_right + offset
    append!(Js, I_tr)           # Col = row of top_right
    append!(Vs, V_tr)           # Same values

    # Bottom-right block: Q_base + A' * Q_η * A
    I_br, J_br, V_br = findnz(sparse(bottom_right))
    append!(Is, I_br .+ n_obs)  # Offset rows
    append!(Js, J_br .+ n_obs)  # Offset columns
    append!(Vs, V_br)

    # Construct sparse matrix
    return sparse(Is, Js, Vs, n_total, n_total)
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
