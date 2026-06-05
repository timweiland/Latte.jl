using Distributions
using OrderedCollections: OrderedDict
using GaussianMarkovRandomFields
using GaussianMarkovRandomFields: LatentModel, hyperparameters, precision_matrix, constraints, model_name
using Random
using SparseArrays: SparseMatrixCSC, sparse

import GaussianMarkovRandomFields: hyperparameters, precision_matrix, constraints, model_name
import Distributions: mean

export LatentGaussianModel, LGM, FunctionLatentModel, latent_gmrf, log_joint_density

# Helper type for wrapping functions as LatentModels.
#
# Contract: `func(; kwargs...) -> (μ::AbstractVector, Q::SparseMatrixCSC)`
# returning the mean vector and sparse precision matrix directly. This avoids
# constructing a full GMRF (and its eager LinearSolve/CHOLMOD factor) on every
# hyperparameter evaluation, which is pure waste for the workspace path.
#
# Optional `constraint::Union{Nothing, Tuple{AbstractMatrix, AbstractVector}}`:
# a hard linear equality constraint `A·x = e` on the latent vector,
# hyperparameter-independent. When set, cold/warm-path calls wrap the resulting
# GMRF with the constraint (`ConstrainedGMRF` / constrained `WorkspaceGMRF`).
struct FunctionLatentModel{FuncType, C <: Union{Nothing, Tuple{AbstractMatrix, AbstractVector}}} <: LatentModel
    func::FuncType
    n::Int
    constraint::C
end

FunctionLatentModel(func, n::Int) = FunctionLatentModel(func, n, nothing)
FunctionLatentModel(func, n::Int, constraint::Tuple{AbstractMatrix, AbstractVector}) =
    FunctionLatentModel{typeof(func), typeof(constraint)}(func, n, constraint)

Base.length(flm::FunctionLatentModel) = flm.n
hyperparameters(flm::FunctionLatentModel) = NamedTuple()  # Function handles its own params
precision_matrix(flm::FunctionLatentModel; kwargs...) = last(flm.func(; kwargs...))
Distributions.mean(flm::FunctionLatentModel; kwargs...) = first(flm.func(; kwargs...))
constraints(flm::FunctionLatentModel; kwargs...) = flm.constraint

# Cold path: caller wants a full GMRF. Pay the LinearSolve/CHOLMOD init once.
function (flm::FunctionLatentModel)(; kwargs...)
    μ, Q = flm.func(; kwargs...)
    gmrf = GMRF(μ, Q)
    return flm.constraint === nothing ? gmrf :
        GaussianMarkovRandomFields.ConstrainedGMRF(gmrf, flm.constraint[1], flm.constraint[2])
end

# Warm path: caller has a workspace. Single `func` call, reuse the workspace's
# symbolic factorization, no throwaway GMRF allocation.
function (flm::FunctionLatentModel)(ws::GaussianMarkovRandomFields.GMRFWorkspace; kwargs...)
    μ, Q = flm.func(; kwargs...)
    Q_sparse = Q isa SparseMatrixCSC ? Q : sparse(Q)
    GaussianMarkovRandomFields.update_precision!(ws, Q_sparse)
    return flm.constraint === nothing ?
        GaussianMarkovRandomFields.WorkspaceGMRF(μ, Q_sparse, ws) :
        GaussianMarkovRandomFields.WorkspaceGMRF(
            μ, Q_sparse, ws, flm.constraint[1], flm.constraint[2]
        )
end

model_name(::FunctionLatentModel) = :function_latent

"""
    LatentGaussianModel{HP, F, O}

A latent Gaussian model specification with hyperparameter prior, latent field prior, and observation model.

Factors as `p(θ) · p(x | θ) · p(y | x, θ)` with `p(x | θ)` Gaussian. This is the
shared structure that INLA, TMB, HMC-on-Laplace, and variational approximations
all operate on.

The alias `LGM` is available for brevity.

# Type Parameters
- `HP`: Type of the hyperparameter specification (HyperparameterSpec)
- `F <: LatentModel`: Type of the latent prior (wrap a function with `FunctionLatentModel`)
- `O <: ObservationModel`: Type of the observation model

# Fields
- `hyperparameter_spec::HP`: Hyperparameter specification with transformations
- `latent_prior::F`: The latent field prior as a `LatentModel`, receiving natural-space hyperparameters as keyword arguments
- `observation_model::O`: Observation model linking observations to latent field

# Example
```julia
using SparseArrays

# Hyperparameter specification (free parameters via `~`)
hp_spec = @hyperparams begin
    (σ ~ Exponential(1.0), transform = log, space = natural)
end

# Latent field prior: a function of the (keyword) hyperparameters returning (mean, precision)
function latent_gmrf(; σ, kwargs...)
    n = 100
    Q = spdiagm(0 => fill(1 / σ^2, n))  # Simple white noise
    μ = zeros(n)
    return (μ, Q)
end

# Observation model
obs_model = ExponentialFamily(Normal)

# Create the latent Gaussian model — wrap the latent function with its dimension
model = LatentGaussianModel(hp_spec, FunctionLatentModel(latent_gmrf, 100), obs_model)
```
"""
struct LatentGaussianModel{HP, F <: LatentModel, O <: ObservationModel}
    hyperparameter_spec::HP
    latent_prior::F
    observation_model::O
    augmentation_info::Union{Nothing, AugmentationInfo}
    # `sym → UnitRange{Int}` mapping each named latent block (e.g. from the
    # DPPL adapter) to its position in the *augmented* latent vector. Empty
    # OrderedDict when there's no naming (hand-built LGMs).
    latent_layout::OrderedDict{Symbol, UnitRange{Int}}

    function LatentGaussianModel(hp_spec::HyperparameterSpec{FreeNT, FixedNT}, latent_prior, observation_model::O, augmentation_info::Union{Nothing, AugmentationInfo} = nothing; latent_layout::OrderedDict{Symbol, UnitRange{Int}} = OrderedDict{Symbol, UnitRange{Int}}()) where {FreeNT, FixedNT, O <: ObservationModel}
        # Catch raw functions with a helpful error message
        if latent_prior isa Function
            error(
                "Raw functions are not supported as latent priors. " *
                    "Use FunctionLatentModel(f, n) to wrap your function with its output dimension n."
            )
        end

        # Validation: check all required hyperparameters are provided (both free and fixed)
        required = Set(hyperparameters(observation_model))
        provided = Set(fieldnames(FreeNT)) ∪ Set(fieldnames(FixedNT))

        missing_params = setdiff(required, provided)
        if !isempty(missing_params)
            error("Missing required hyperparameters for $(typeof(observation_model)): $(collect(missing_params))")
        end

        return new{typeof(hp_spec), typeof(latent_prior), O}(hp_spec, latent_prior, observation_model, augmentation_info, latent_layout)
    end
end

"""
    LGM

Alias for `LatentGaussianModel`.
"""
const LGM = LatentGaussianModel

function _restrict_obs_model_to_indices(obs_model::ObservationModel, indices)
    error(
        "Prediction via missing values is not supported for $(typeof(obs_model)). " *
            "Only ExponentialFamily observation models support this, as prediction via " *
            "missing values assumes a 1:1 mapping between latent variables and observations."
    )
end

function _restrict_obs_model_to_indices(obs_model::ExponentialFamily, indices)
    # Preserve kwarg aliases (added upstream in GMRFs #106) when re-wrapping
    # the obs model with a sliced index range.
    return ExponentialFamily(obs_model.family, obs_model.link, indices, obs_model.kwarg_aliases)
end

"""
    LatentGaussianModel(
        hp_spec::HyperparameterSpec,
        base_latent_prior,
        obs_model::LinearlyTransformedObservationModel;
        augment_latent::Bool = true,
        linear_predictor_precision::Real = 1e6
    )

Specialized constructor for LinearlyTransformedObservationModel that automatically augments
the latent field with linear predictor components.

When `augment_latent=true` (default), this constructor:
1. Extracts the design matrix A and base observation model from the LinearlyTransformedObservationModel
2. Wraps the base_latent_prior in an AugmentedLatentModel
3. Creates an augmented GMRF with structure [η; x_base] where η = A * x_base
4. Stores augmentation metadata for accessing linear predictor vs base marginals later

# Arguments
- `hp_spec::HyperparameterSpec`: Hyperparameter specification
- `base_latent_prior`: Base latent prior (function or LatentModel) returning GMRF of size n_base
- `obs_model::LinearlyTransformedObservationModel`: Observation model with design matrix A
- `augment_latent::Bool = true`: Whether to automatically augment (set false to disable)
- `linear_predictor_precision::Real = 1e6`: Precision for enforcing η ≈ A * x_base (high = tight coupling)

# Returns
An LatentGaussianModel with:
- Augmented latent prior returning GMRFs of size n_obs + n_base
- Base observation model (unwrapped from LinearlyTransformedObservationModel)
- AugmentationInfo metadata for tracking which indices are linear predictors vs base components

# Example
```julia
# Base latent model
base_model = AR1Model(100)  # 100 base components

# Design matrix: 200 observations × 100 base components
A = randn(200, 100)
base_obs = ExponentialFamily(Poisson)
obs_model = LinearlyTransformedObservationModel(base_obs, A)

# Hyperparameters
hp_spec = @hyperparams begin
    (τ ~ Exponential(1.0), transform = log, space = natural)
    (ρ ~ Beta(2, 2), transform = logit, space = working)
end

# Automatic augmentation (enabled by default)
model = LatentGaussianModel(hp_spec, base_model, obs_model)
# Result: latent field has 300 components [η₁...η₂₀₀; x_base₁...x_base₁₀₀]
# model.augmentation_info contains metadata about the structure

# Opt-out of augmentation
model_no_aug = LatentGaussianModel(hp_spec, base_model, obs_model; augment_latent=false)
# Result: user must manually handle augmentation
```
"""
function LatentGaussianModel(
        hp_spec::HyperparameterSpec,
        base_latent_prior::F,
        obs_model::LinearlyTransformedObservationModel;
        augment_latent::Bool = true,
        linear_predictor_precision::Real = 1.0e6,
        latent_layout::OrderedDict{Symbol, UnitRange{Int}} = OrderedDict{Symbol, UnitRange{Int}}()
    ) where {F}
    if !augment_latent
        # User opted out - pass through to base constructor without augmentation
        return LatentGaussianModel(
            hp_spec, base_latent_prior, obs_model, nothing;
            latent_layout = latent_layout,
        )
    end

    # Extract components from LinearlyTransformedObservationModel
    design_matrix = obs_model.design_matrix
    base_obs_model = obs_model.base_model

    # Get dimensions
    n_obs, n_full = size(design_matrix)

    # Infer base latent dimension
    # If base_latent_prior is a LatentModel, use length()
    # If it's a function, we need to check what it returns (done during validation)
    latent_prior = base_latent_prior
    if base_latent_prior isa LatentModel
        n_base = length(base_latent_prior)
        if n_base != n_full
            error("Dimension mismatch: base_latent_prior has dimension $n_base, but design matrix has $(n_full) columns. Expected them to match.")
        end
    else
        # base_latent_prior is a function - wrap it in FunctionLatentModel
        n_base = n_full

        # Wrap function in FunctionLatentModel, then in AugmentedLatentModel
        latent_prior = FunctionLatentModel(base_latent_prior, n_base)
    end

    # An LTM offset (η = A·x + b) is absorbed into the augmented prior as a
    # mean shift on the linear-predictor block, so the base obs model sees η
    # directly. A θ-dependent offset thus becomes a θ-dependent prior mean,
    # which the forward-mode IFT differentiates exactly.
    augmented_latent_model = AugmentedLatentModel(
        latent_prior,
        design_matrix;
        offset = obs_model.offset,
        linear_predictor_precision = linear_predictor_precision
    )

    # Create augmentation metadata
    augmentation_info = AugmentationInfo(n_obs, n_base)

    obs_model = _restrict_obs_model_to_indices(base_obs_model, augmentation_info.linear_predictor_indices)

    # Call base constructor with augmented model and base observation model
    return LatentGaussianModel(
        hp_spec, augmented_latent_model, obs_model, augmentation_info;
        latent_layout = latent_layout,
    )
end

"""
    latent_groups(model::LatentGaussianModel) -> OrderedDict{Symbol, UnitRange{Int}}

Name → augmented-latent-range mapping for a DPPL-built LGM (empty for
hand-built LGMs). Matches `latent_groups(::INLAResult)` so lookup by
name works interchangeably on the model and its inference result.
"""
latent_groups(model::LatentGaussianModel) = model.latent_layout

"""
    latent_gmrf(model::LatentGaussianModel, θ_named)

Get the latent field GMRF for given hyperparameters θ_named.
"""
function latent_gmrf(model::LatentGaussianModel, θ_named)
    return model.latent_prior(; θ_named...)
end

"""
    latent_gmrf(model::LatentGaussianModel, ws, θ_named)

Construct the latent GMRF through a persistent workspace `ws`. Reuses the
workspace's Cholesky symbolic factorization; only the numeric values are
refactorized. Use this form inside hot loops over hyperparameters.
"""
function latent_gmrf(model::LatentGaussianModel, ws, θ_named)
    return model.latent_prior(ws; θ_named...)
end

"""
    log_joint_density(model::LatentGaussianModel, x, θ_w::WorkingHyperparameters, y)

Evaluate the joint log-density log π(x, θ, y) for the INLA model in working space.

This computes: log π(θ) + log π(x | θ) + log π(y | x, θ)

# Arguments
- `model::LatentGaussianModel`: The INLA model
- `x`: Latent field values
- `θ_w`: Hyperparameters in working space
- `y`: Observations

This is the main implementation. Creates the latent GMRF and observation likelihood internally.
"""
function log_joint_density(model::LatentGaussianModel, x, θ_w::WorkingHyperparameters, y)
    # Hyperparameter prior contribution in working space
    log_prior_θ = logpdf_prior(θ_w)

    if log_prior_θ === -Inf
        # Early return
        return -Inf
    end

    # Convert to natural space for latent GMRF and observation model
    θ_natural_nt = convert(NamedTuple, convert(NaturalHyperparameters, θ_w))

    # Create latent GMRF and materialized observation likelihood in natural space
    latent_prior = latent_gmrf(model, θ_natural_nt)
    obs_lik = model.observation_model(y; θ_natural_nt...)

    # Latent field prior contribution
    log_prior_x = logpdf(latent_prior, x)

    # Observation model contribution
    log_likelihood = loglik(x, obs_lik)

    return log_prior_θ + log_prior_x + log_likelihood
end

"""
    log_joint_density(model::LatentGaussianModel, x, θ_n::NaturalHyperparameters, y)

Evaluate the joint log-density log π(x, θ, y) for the INLA model in natural space.

Converts to working space and adds Jacobian correction term.

# Arguments
- `model::LatentGaussianModel`: The INLA model
- `x`: Latent field values
- `θ_n`: Hyperparameters in natural space
- `y`: Observations
"""
function log_joint_density(model::LatentGaussianModel, x, θ_n::NaturalHyperparameters, y)
    θ_w = convert(WorkingHyperparameters, θ_n)
    return log_joint_density(model, x, θ_w, y) + logdetjac(θ_n)
end

"""
    Base.show(io::IO, model::LatentGaussianModel)

Pretty printing for INLA models.
"""
function Base.show(io::IO, model::LatentGaussianModel{D, F, O}) where {D, F, O}
    println(io, "LatentGaussianModel")
    println(io, "  Hyperparameter spec:\n    $(repr(model.hyperparameter_spec))")
    println(io, "  Latent prior function: ", typeof(model.latent_prior))
    return println(io, "  Observation model: ", typeof(model.observation_model))
end

"""
    Random.rand([rng], model::LatentGaussianModel)

Sample from a `LatentGaussianModel`, returning a NamedTuple with hyperparameters θ, latent field x, and observations y.

The sampling process:
1. Sample hyperparameters θ from the hyperparameter prior (in working space internally)
2. Convert to natural space and generate the latent GMRF
3. Sample x from the latent GMRF
4. Sample observations y given x using the observation model

# Arguments
- `rng`: Optional random number generator (defaults to global RNG)
- `model`: The LatentGaussianModel to sample from

# Returns
A NamedTuple `(θ = θ_natural, x = x_vec, y = y_vec)` where:
- `θ_natural`: NaturalHyperparameters object representing hyperparameter values in natural space
- `x_vec`: Vector of latent field values
- `y_vec`: Vector of observation values
"""
function Random.rand(rng::AbstractRNG, model::LatentGaussianModel)
    spec = model.hyperparameter_spec

    # Sample free hyperparameters in working space
    θ_working_vec = [rand(rng, hp.prior) for hp in values(spec.free)]

    # Convert to WorkingHyperparameters, then to NaturalHyperparameters
    θ_working = WorkingHyperparameters(θ_working_vec, spec)
    θ_natural = convert(NaturalHyperparameters, θ_working)

    # Convert to NamedTuple for passing to functions that need it
    θ_natural_nt = convert(NamedTuple, θ_natural)

    # Generate latent GMRF and sample from it
    gmrf = latent_gmrf(model, θ_natural_nt)
    x = rand(rng, gmrf)

    # Sample observations given latent field using GMRF's conditional_distribution
    y_dist = GaussianMarkovRandomFields.conditional_distribution(model.observation_model, x; θ_natural_nt...)
    y = rand(rng, y_dist)

    return (θ = θ_natural, x = x, y = y)
end

# Default to global RNG
Random.rand(model::LatentGaussianModel) = rand(Random.default_rng(), model)
