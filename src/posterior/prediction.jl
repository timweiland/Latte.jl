using GaussianMarkovRandomFields: PoissonObservations, NegativeBinomialObservations

export PredictionInfo, predicted_marginals, observed_marginals, poisson_observations

"""
    PredictionInfo

Metadata about prediction setup, stored in `INLAResult` when `y` contains `missing` values.

# Fields
- `n_latent::Int`: Total latent field dimension
- `observed_indices::Vector{Int}`: Positions in latent field that have observations
- `prediction_indices::Vector{Int}`: Positions in latent field that are predicted (missing)
"""
struct PredictionInfo
    n_latent::Int
    observed_indices::Vector{Int}
    prediction_indices::Vector{Int}
    # Compact (non-augmented) LTM models don't materialize η as latent positions:
    # `*_indices` are then OBSERVATION-row indices into this full design matrix,
    # and prediction goes through A_missing·μ*. `nothing` for augmented models,
    # where `*_indices` index the η-block of the latent field directly.
    design_matrix::Union{Nothing, AbstractMatrix}
end

PredictionInfo(n_latent::Int, observed_indices::Vector{Int}, prediction_indices::Vector{Int}) =
    PredictionInfo(n_latent, observed_indices, prediction_indices, nothing)

function PredictionInfo(n_latent::Int, observed_mask::AbstractVector{Bool}; design_matrix = nothing)
    observed_indices = findall(observed_mask)
    prediction_indices = findall(.!observed_mask)
    return PredictionInfo(n_latent, observed_indices, prediction_indices, design_matrix)
end

function Base.show(io::IO, info::PredictionInfo)
    return print(io, "PredictionInfo($(length(info.observed_indices)) observed, $(length(info.prediction_indices)) predicted, $(info.n_latent) latent)")
end

"""
    poisson_observations(; counts, exposure=nothing)

Create Poisson observations that support `missing` values in counts for prediction.

Missing counts mark prediction locations where no likelihood is contributed.
Exposure values at missing positions are ignored.

# Examples
```julia
y = poisson_observations(counts=[3, missing, 7], exposure=[1.0, 2.0, 0.75])
result = inla(model, y)
```
"""
function poisson_observations(; counts::AbstractVector, exposure::Union{Nothing, AbstractVector{<:Real}} = nothing)
    if !any(ismissing, counts)
        # No missing values — return standard PoissonObservations
        if exposure === nothing
            return PoissonObservations(collect(Int, counts))
        else
            return PoissonObservations(collect(Int, counts), collect(Float64, exposure))
        end
    end

    # Has missing values — return a MissingPoissonObservations wrapper
    return MissingPoissonObservations(counts, exposure)
end

"""
    MissingPoissonObservations

Internal wrapper for Poisson observations with `missing` counts (prediction locations).
Consumed by `_prepare_for_prediction` to split into observed data + prediction indices.
"""
struct MissingPoissonObservations <: AbstractVector{Union{Missing, Tuple{Int, Float64}}}
    counts::Vector{Union{Missing, Int}}
    exposure::Union{Nothing, Vector{Float64}}
end

Base.size(y::MissingPoissonObservations) = (length(y.counts),)
function Base.getindex(y::MissingPoissonObservations, i::Int)
    ismissing(y.counts[i]) && return missing
    e = y.exposure === nothing ? 1.0 : y.exposure[i]
    return (y.counts[i], e)
end

# Normalize observation data to the type expected by the observation model.
# Passthrough for most observation models.
_normalize_observations(y, obs_model) = y

# Poisson: plain integer vectors → PoissonObservations
function _normalize_observations(y::AbstractVector{<:Integer}, obs_model::ExponentialFamily{Poisson})
    return PoissonObservations(collect(Int, y))
end

# NegativeBinomial: plain integer vectors → NegativeBinomialObservations (no exposure)
function _normalize_observations(y::AbstractVector{<:Integer}, obs_model::ExponentialFamily{NegativeBinomial})
    return NegativeBinomialObservations(collect(Int, y))
end

# LinearlyTransformedObservationModel — delegate to the base obs model.
# Without this, non-augmented LGMs that keep an LTM as their observation
# model can't normalise raw integer y vectors (the fallback no-op leaves
# a Vector{Int} that then errors at `obs_model(y; θ...)` materialisation).
function _normalize_observations(y::AbstractVector, m::LinearlyTransformedObservationModel)
    return _normalize_observations(y, m.base_model)
end


"""
    _prepare_for_prediction(model::LatentGaussianModel, y::AbstractVector)

Pre-process `y` for prediction: detect `missing` values, extract observed data,
and create a modified model with observation indices.

Returns `(y_processed, model_processed, prediction_info)`.
If no missing values, returns `(y_normalized, model, nothing)` unchanged.
"""
function _prepare_for_prediction(model::LatentGaussianModel, y::AbstractVector)
    if !any(ismissing, y)
        return _normalize_observations(y, model.observation_model), model, nothing
    end

    n_latent = length(model.latent_prior)
    observed_mask = .!ismissing.(y)

    if !any(observed_mask)
        throw(ArgumentError("All observations are missing. At least one observation must be non-missing."))
    end

    if model.augmentation_info !== nothing
        # Augmented model: latent field is [η₁...η_n_obs; x_base₁...x_base_n_base].
        # The observed_mask applies to y, which corresponds to the η part.
        # We restrict the obs model to the observed η indices and build PredictionInfo
        # so that prediction_indices point to the η's with missing observations.
        aug = model.augmentation_info
        observed_indices = findall(observed_mask)
        prediction_info = PredictionInfo(n_latent, observed_mask)

        y_obs = _extract_observed(y, observed_mask, model.observation_model)
        new_obs_model = _restrict_obs_model_to_indices(model.observation_model, observed_indices)

        model_processed = LatentGaussianModel(
            model.hyperparameter_spec,
            model.latent_prior,
            new_obs_model,
            model.augmentation_info
        )

        return y_obs, model_processed, prediction_info
    end

    if model.observation_model isa LinearlyTransformedObservationModel
        # Compact LTM (η = A·ψ): fit on the observed rows of A, then predict the
        # missing rows from A_missing·μ* (see predicted_marginals). The full A is
        # stored so the missing/observed obs rows can be reconstructed; the latent
        # ψ itself is unchanged (all columns kept).
        A_full = model.observation_model.design_matrix
        observed_indices = findall(observed_mask)
        prediction_info = PredictionInfo(n_latent, observed_mask; design_matrix = A_full)

        off = model.observation_model.offset
        new_obs_model = LinearlyTransformedObservationModel(
            model.observation_model.base_model, A_full[observed_indices, :];
            offset = off === nothing ? nothing : off[observed_indices],
        )
        y_obs = _extract_observed(y, observed_mask, model.observation_model.base_model)

        model_processed = LatentGaussianModel(
            model.hyperparameter_spec, model.latent_prior, new_obs_model, nothing,
        )
        return y_obs, model_processed, prediction_info
    end

    observed_indices = findall(observed_mask)
    prediction_info = PredictionInfo(n_latent, observed_mask)

    # Extract observed data and create modified observation model
    y_obs = _extract_observed(y, observed_mask, model.observation_model)
    new_obs_model = _restrict_obs_model_to_indices(model.observation_model, observed_indices)

    # Create modified LatentGaussianModel with indexed observation model
    model_processed = LatentGaussianModel(
        model.hyperparameter_spec,
        model.latent_prior,
        new_obs_model,
        model.augmentation_info
    )

    return y_obs, model_processed, prediction_info
end

# Extract observed (non-missing) elements from y, preserving correct type for each family.

# Plain vectors (Normal, Bernoulli)
function _extract_observed(y::AbstractVector, observed_mask::AbstractVector{Bool}, obs_model)
    return collect(skipmissing(y[observed_mask]))
end

# Poisson: integer vector → wrap into PoissonObservations
function _extract_observed(y::AbstractVector{<:Union{Missing, Integer}}, observed_mask::AbstractVector{Bool}, obs_model::ExponentialFamily{Poisson})
    return PoissonObservations(Int[y[i] for i in eachindex(y) if observed_mask[i]])
end

# MissingPoissonObservations with exposure
function _extract_observed(y::MissingPoissonObservations, observed_mask::AbstractVector{Bool}, obs_model)
    counts_obs = Int[y.counts[i] for i in eachindex(y.counts) if observed_mask[i]]
    if y.exposure === nothing
        return PoissonObservations(counts_obs)
    else
        exposure_obs = Float64[y.exposure[i] for i in eachindex(y.exposure) if observed_mask[i]]
        return PoissonObservations(counts_obs, exposure_obs)
    end
end

"""
    predicted_marginals(result::INLAResult)

Extract marginals for prediction indices (where `y` was `missing`).

These marginals represent the posterior distribution of the latent field at
locations without observations, informed by the GMRF prior (spatial/temporal correlation).
"""
function predicted_marginals(result::INLAResult)
    info = result.prediction_info
    if info === nothing
        throw(ArgumentError("No prediction was requested. Pass y with missing values to enable prediction."))
    end
    if info.design_matrix !== nothing
        # Compact LTM: the predictors aren't latent, so build the missing obs'
        # η = A_missing·μ* marginals (μ*-corrected mean + GA variance) via lincombs.
        return linear_combinations(result, info.design_matrix[info.prediction_indices, :])
    end
    return result.latent_marginals[info.prediction_indices]
end

"""
    observed_marginals(result::INLAResult)

Extract marginals for observed indices (where `y` was not `missing`).
"""
function observed_marginals(result::INLAResult)
    info = result.prediction_info
    if info === nothing
        return result.latent_marginals
    end
    if info.design_matrix !== nothing
        return linear_combinations(result, info.design_matrix[info.observed_indices, :])
    end
    return result.latent_marginals[info.observed_indices]
end
