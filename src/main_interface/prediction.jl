using GaussianMarkovRandomFields: PoissonObservations

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
end

function PredictionInfo(n_latent::Int, observed_mask::AbstractVector{Bool})
    observed_indices = findall(observed_mask)
    prediction_indices = findall(.!observed_mask)
    return PredictionInfo(n_latent, observed_indices, prediction_indices)
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


"""
    _prepare_for_prediction(model::INLAModel, y::AbstractVector)

Pre-process `y` for prediction: detect `missing` values, extract observed data,
and create a modified model with observation indices.

Returns `(y_processed, model_processed, prediction_info)`.
If no missing values, returns `(y_normalized, model, nothing)` unchanged.
"""
function _prepare_for_prediction(model::INLAModel, y::AbstractVector)
    if !any(ismissing, y)
        return _normalize_observations(y, model.observation_model), model, nothing
    end

    n_latent = length(model.latent_prior)
    observed_mask = .!ismissing.(y)

    if !any(observed_mask)
        throw(ArgumentError("All observations are missing. At least one observation must be non-missing."))
    end

    if model.augmentation_info !== nothing
        throw(
            ArgumentError(
                "Prediction via missing values is not supported for augmented models " *
                    "(LinearlyTransformedObservationModel). Prediction via missing values assumes " *
                    "a 1:1 mapping between latent variables and observations."
            )
        )
    end

    observed_indices = findall(observed_mask)
    prediction_info = PredictionInfo(n_latent, observed_mask)

    # Extract observed data and create modified observation model
    y_obs = _extract_observed(y, observed_mask, model.observation_model)
    new_obs_model = _restrict_obs_model_to_indices(model.observation_model, observed_indices)

    # Create modified INLAModel with indexed observation model
    model_processed = INLAModel(
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
    if result.prediction_info === nothing
        throw(ArgumentError("No prediction was requested. Pass y with missing values to enable prediction."))
    end
    return result.latent_marginals[result.prediction_info.prediction_indices]
end

"""
    observed_marginals(result::INLAResult)

Extract marginals for observed indices (where `y` was not `missing`).
"""
function observed_marginals(result::INLAResult)
    if result.prediction_info === nothing
        return result.latent_marginals
    end
    return result.latent_marginals[result.prediction_info.observed_indices]
end
