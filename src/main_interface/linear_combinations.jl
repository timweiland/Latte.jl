using LinearAlgebra: dot
using LinearSolve: solve!

export linear_combinations

"""
    _reconstruct_ga(model, y_obs, θ) -> (ga, θ_natural_nt)

Reconstruct the Gaussian approximation at hyperparameter configuration `θ`.

Converts θ to natural space, builds the prior GMRF and observation likelihood,
and returns the Gaussian approximation along with the natural-space NamedTuple.
Used by `linear_combinations` and `rand(::INLAResult)`.
"""
function _reconstruct_ga(model, y_obs, θ)
    θ_natural = convert(NaturalHyperparameters, θ)
    θ_natural_nt = convert(NamedTuple, θ_natural)
    prior_gmrf = latent_gmrf(model, θ_natural_nt)
    obs_lik = model.observation_model(y_obs; θ_natural_nt...)
    return gaussian_approximation(prior_gmrf, obs_lik), θ_natural_nt
end

"""
    _integration_weights(exploration::HyperparameterExploration) -> Vector{Float64}

Compute normalized integration weights from exploration results with log-sum-exp stabilization.
"""
function _integration_weights(exploration)
    integration_points = exploration.grid_points[exploration.integration_indices]
    log_weights = [p.log_density for p in integration_points]
    weights = exp.(log_weights .- maximum(log_weights))
    weights ./= sum(weights)
    return weights
end

"""
    linear_combinations(result::INLAResult, A::AbstractMatrix) -> Vector{WeightedMixture}
    linear_combinations(result::INLAResult, a::AbstractVector) -> WeightedMixture

Compute posterior marginals for linear combinations z = A * x of the latent field.

For each row aₖ of A, the marginal of zₖ = aₖᵀx is computed as a weighted mixture
of Gaussian conditionals over the hyperparameter integration points, matching
R-INLA's default `lincomb.derived.only=TRUE` approach.

At each integration point θⱼ with weight wⱼ, the conditional distribution of
zₖ = aₖᵀx is Normal(aₖᵀμⱼ, √(aₖᵀQⱼ⁻¹aₖ)), where μⱼ and Qⱼ are the mean
and precision matrix of the Gaussian approximation at θⱼ.

# Arguments
- `result::INLAResult`: Results from `inla()` inference
- `A::AbstractMatrix`: m×n matrix where each row defines a linear combination
- `a::AbstractVector`: Length-n vector defining a single linear combination

# Returns
- `Vector{WeightedMixture}`: One marginal per row of A (matrix form)
- `WeightedMixture`: Single marginal (vector form)

# Example
```julia
result = inla(model, y; progress=false)
n = length(result.latent_marginals)

# Contrast between first two latent variables
a = zeros(n); a[1] = 1; a[2] = -1
contrast_marginal = linear_combinations(result, a)
mean(contrast_marginal)  # posterior mean of x₁ - x₂

# Sum of all latent variables
sum_marginal = linear_combinations(result, ones(1, n))

# Multiple linear combinations at once
A = [1.0 -1.0 zeros(1, n-2)...;
     zeros(1, n-1)... 1.0]
marginals = linear_combinations(result, A)
```

# Notes
- For single-variable queries, prefer `result.latent_marginals[i]` which includes
  Laplace corrections. This function uses Gaussian conditionals, justified by CLT
  for multi-variable combinations.
- Sparse matrices are supported and recommended for large, sparse A.
"""
function linear_combinations(result::INLAResult, A::AbstractMatrix)
    exploration = result.exploration
    model = result.model
    y_obs = _get_y_obs(result)

    m = size(A, 1) # number of linear combinations
    n = size(A, 2) # latent field dimension

    n_latent = length(result.latent_marginals)
    if n != n_latent
        throw(
            DimensionMismatch(
                "A has $n columns but the latent field has $n_latent variables"
            )
        )
    end

    # Get integration points and weights
    weights = _integration_weights(exploration)
    integration_points = exploration.grid_points[exploration.integration_indices]
    n_points = length(integration_points)

    # For each linear combination, collect Normal components across integration points
    components = [Vector{Normal{Float64}}(undef, n_points) for _ in 1:m]

    for (j, point) in enumerate(integration_points)
        ga, _ = _reconstruct_ga(model, y_obs, point.θ)
        μ = mean(ga)

        # Reuse the GA's linsolve cache (already has the factored precision matrix)
        cache = GaussianMarkovRandomFields.linsolve_cache(ga)

        # Solve Q \ aₖ and compute mean/variance for each linear combination
        for k in 1:m
            a_k = view(A, k, :)
            cache.b .= a_k
            sol = solve!(cache)
            z_mean = dot(a_k, μ)
            z_var = dot(a_k, sol.u)
            components[k][j] = Normal(z_mean, sqrt(max(z_var, 0.0)))
        end
    end

    return [WeightedMixture(components[k], weights) for k in 1:m]
end

function linear_combinations(result::INLAResult, a::AbstractVector)
    A = reshape(a, 1, :)
    return linear_combinations(result, A)[1]
end
