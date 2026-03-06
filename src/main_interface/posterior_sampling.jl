using Random
using Distributions: Categorical

"""
    rand([rng], result::INLAResult; include_y=false)
    rand([rng], result::INLAResult, n::Int; include_y=false)

Draw joint samples from the approximate posterior computed by INLA.

For each sample:
1. A hyperparameter configuration θ is drawn from the integration grid (weighted by posterior density)
2. A joint latent field x is drawn from the Gaussian approximation at that θ
3. Optionally, observations y are drawn from the observation model conditional on x and θ

Samples are batched by integration point to minimize the number of Gaussian approximation
reconstructions (one per unique θ, not one per sample).

# Returns
A NamedTuple (single sample) or `Vector{NamedTuple}` (multiple samples) with fields:
- `θ::WorkingHyperparameters`: Sampled hyperparameters
- `x::Vector{Float64}`: Joint latent field sample
- `y::Vector` (only if `include_y=true`): Observation sample conditional on x and θ

# Example
```julia
result = inla(model, y)

# Single sample
s = rand(result)

# Draw 1000 posterior samples
samples = rand(result, 1000)

# Compute a derived quantity (e.g., difference between two latent variables)
diffs = [s.x[1] - s.x[2] for s in samples]
mean(diffs), quantile(diffs, [0.025, 0.975])

# Include observation samples for posterior predictive checks
samples_with_y = rand(result, 1000; include_y=true)

# With explicit RNG for reproducibility
samples = rand(MersenneTwister(42), result, 1000)
```
"""
function Random.rand(rng::AbstractRNG, result::INLAResult, n::Int; include_y::Bool = false)
    exploration = result.exploration
    model = result.model

    # Get y_obs for GA reconstruction
    y_obs = _get_y_obs(result)

    # Compute integration weights
    integration_points = exploration.grid_points[exploration.integration_indices]
    log_weights = [p.log_density for p in integration_points]
    weights = exp.(log_weights .- maximum(log_weights))
    weights ./= sum(weights)

    # Sample integration point indices
    point_indices = rand(rng, Categorical(weights), n)

    # Batch by unique integration point to minimize GA reconstructions
    unique_indices = unique(point_indices)
    samples = Vector{NamedTuple}(undef, n)

    for idx in unique_indices
        point = integration_points[idx]
        θ = point.θ

        # Convert to natural space
        θ_natural = convert(NaturalHyperparameters, θ)
        θ_natural_nt = convert(NamedTuple, θ_natural)

        # Reconstruct Gaussian approximation
        prior_gmrf = latent_gmrf(model, θ_natural_nt)
        obs_lik = model.observation_model(y_obs; θ_natural_nt...)
        ga = gaussian_approximation(prior_gmrf, obs_lik)

        # Draw samples for all occurrences of this integration point
        for i in findall(==(idx), point_indices)
            x = rand(rng, ga)

            if include_y
                y_dist = GaussianMarkovRandomFields.conditional_distribution(
                    model.observation_model, x; θ_natural_nt...
                )
                y_sample = rand(rng, y_dist)
                samples[i] = (θ = θ, x = x, y = y_sample)
            else
                samples[i] = (θ = θ, x = x)
            end
        end
    end

    return samples
end

# Convenience methods
Random.rand(result::INLAResult, n::Int; kwargs...) = rand(Random.default_rng(), result, n; kwargs...)
Random.rand(rng::AbstractRNG, result::INLAResult; kwargs...) = rand(rng, result, 1; kwargs...)[1]
Random.rand(result::INLAResult; kwargs...) = rand(Random.default_rng(), result; kwargs...)

"""
    _get_y_obs(result::INLAResult)

Extract the processed observations used during INLA inference.
When the model has prediction (missing values), this returns the observed-only subset.
"""
function _get_y_obs(result::INLAResult)
    if haskey(result.options, :y_obs)
        return result.options.y_obs
    end
    # Fallback: re-derive from original y
    y_obs, _, _ = _prepare_for_prediction(result.model, result.options.y)
    return y_obs
end
