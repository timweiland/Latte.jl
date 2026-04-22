using Random
using Distributions: Categorical

"""
    rand([rng], result::InferenceResult; include_y=false)
    rand([rng], result::InferenceResult, n::Int; include_y=false)

Draw joint samples from the approximate posterior computed by an inference
method.

For each sample:

1. A hyperparameter configuration θ is drawn from the method's posterior
   approximation over θ (integration grid for INLA; Gaussian at MAP for TMB;
   the sample chain for HMC-Laplace).
2. A joint latent field x is drawn from the inner approximation at that θ.
3. Optionally (`include_y=true`), observations y are drawn from the
   observation model conditional on x and θ.

# Returns

- `rand(r)` → `NamedTuple{(:θ, :x)}` (or `(:θ, :x, :y)` with `include_y=true`)
  containing the single draw as vectors.
- `rand(r, n)` → [`PosteriorSamples`](@ref). Row-aligned matrices of θ and x
  (plus y when requested). Iterating / indexing yields per-draw
  `NamedTuple`s.

# Example

```julia
result = inla(model, y)

# Single joint draw of (θ, x)
s = rand(result)
s.θ           # working-space θ vector
s.x           # latent-field vector

# 1000 draws — matrices internally, iterable for per-sample access
samples = rand(result, 1000)
samples.x     # 1000 × n_latent
[diff.x for diff in samples]

# With posterior-predictive y
samples_y = rand(result, 1000; include_y = true)
samples_y.y   # 1000 × n_y
```
"""
function Random.rand(rng::AbstractRNG, result::INLAResult, n::Int; include_y::Bool = false)
    exploration = result.exploration
    m = result.model

    # Observations used during inference (prediction-aware)
    y_obs = _get_y_obs(result)

    # Integration weights over the grid
    weights = _integration_weights(exploration)
    integration_points = exploration.grid_points[exploration.integration_indices]

    # Sample integration-point indices
    point_indices = rand(rng, Categorical(weights), n)

    # One-time symbolic factorization, reused for every GA reconstruction
    θ_ref_nt = convert(NamedTuple, convert(NaturalHyperparameters, integration_points[1].θ))
    ws = make_workspace(m.latent_prior; θ_ref_nt...)

    # Pre-allocate output matrices. Determine n_hp, n_x from one sample.
    n_hp = length(integration_points[1].θ.θ)
    n_x = length(y_obs) # will re-check after first x sample
    θ_mat = Matrix{Float64}(undef, n, n_hp)
    x_mat = nothing
    y_mat = nothing

    unique_indices = unique(point_indices)
    for idx in unique_indices
        point = integration_points[idx]

        # Reconstruct Gaussian approximation at this θ
        ga, θ_natural_nt = _reconstruct_ga(m, y_obs, point.θ, ws)
        θ_natural_vec = collect(values(θ_natural_nt))

        for i in findall(==(idx), point_indices)
            θ_mat[i, :] = θ_natural_vec

            x_sample = rand(rng, ga)
            if x_mat === nothing
                x_mat = Matrix{Float64}(undef, n, length(x_sample))
            end
            x_mat[i, :] = x_sample

            if include_y
                y_dist = GaussianMarkovRandomFields.conditional_distribution(
                    m.observation_model, x_sample; θ_natural_nt...
                )
                y_sample = rand(rng, y_dist)
                if y_mat === nothing
                    y_mat = Matrix{eltype(y_sample)}(undef, n, length(y_sample))
                end
                y_mat[i, :] = y_sample
            end
        end
    end

    return PosteriorSamples(θ_mat, x_mat; y = y_mat)
end

# Single-sample form. Matches Julia's `rand(dist)` vs `rand(dist, n)`
# convention — returns a plain NamedTuple, not a PosteriorSamples.
function Random.rand(rng::AbstractRNG, result::INLAResult; include_y::Bool = false)
    samples = rand(rng, result, 1; include_y = include_y)
    return samples[1]
end

Random.rand(result::INLAResult, n::Int; kwargs...) =
    rand(Random.default_rng(), result, n; kwargs...)
Random.rand(result::INLAResult; kwargs...) =
    rand(Random.default_rng(), result; kwargs...)

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
