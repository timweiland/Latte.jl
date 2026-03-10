using StatsFuns: logsumexp

export model_average, BMAResult

"""
    BMAResult

Result of Bayesian model averaging over multiple INLA models.

# Fields
- `latent_marginals::Vector{WeightedMixture}`: Model-averaged posterior marginals for each latent variable
- `model_weights::Vector{Float64}`: Posterior model weights (sum to 1)
- `log_marginal_likelihoods::Vector{Float64}`: Log marginal likelihood for each model

# Usage
```julia
result1 = inla(model1, y)
result2 = inla(model2, y)
bma = model_average([result1, result2])

# Access averaged marginals
summary_df(bma.latent_marginals)

# Inspect model weights
bma.model_weights  # e.g., [0.73, 0.27]
```
"""
struct BMAResult
    latent_marginals::Vector{WeightedMixture}
    model_weights::Vector{Float64}
    log_marginal_likelihoods::Vector{Float64}
end

"""
    model_average(results::Vector{<:INLAResult}; prior_weights=nothing) -> BMAResult

Bayesian model averaging over multiple INLA model fits.

Combines posterior marginals from `K` models weighted by their posterior model
probabilities `p(Mₖ | y) ∝ p(y | Mₖ) · p(Mₖ)`, where `p(y | Mₖ)` is the
marginal likelihood computed by INLA.

The averaged posterior marginal for latent variable `i` is:
```math
p(xᵢ | y) = ∑ₖ p(xᵢ | y, Mₖ) · p(Mₖ | y)
```

# Arguments
- `results`: Vector of `INLAResult` objects from different models (must have same latent dimension)
- `prior_weights`: Optional prior model probabilities (default: equal weights). Must sum to a positive value.

# Returns
A [`BMAResult`](@ref) containing model-averaged latent marginals and posterior model weights.

# Example
```julia
# Fit models with different priors or latent structures
result1 = inla(model1, y)
result2 = inla(model2, y)

# Equal prior weights (default)
bma = model_average([result1, result2])

# Prior preference for model 1
bma = model_average([result1, result2]; prior_weights = [0.8, 0.2])
```
"""
function model_average(results::Vector{<:INLAResult}; prior_weights = nothing)
    K = length(results)
    K == 0 && throw(ArgumentError("Must provide at least one INLAResult"))

    # Validate consistent latent dimensions
    n_latent = length(results[1].latent_marginals)
    for k in 2:K
        if length(results[k].latent_marginals) != n_latent
            throw(
                DimensionMismatch(
                    "Model $k has $(length(results[k].latent_marginals)) latent variables, " *
                        "expected $n_latent (from model 1)"
                )
            )
        end
    end

    # Validate prior weights
    if prior_weights !== nothing
        if length(prior_weights) != K
            throw(
                ArgumentError(
                    "prior_weights has length $(length(prior_weights)), expected $K"
                )
            )
        end
    end

    # Compute log marginal likelihoods
    log_mlls = [r.exploration.log_normalization_constant for r in results]

    # Single model: skip weight computation
    if K == 1
        return BMAResult(results[1].latent_marginals, [1.0], log_mlls)
    end

    # Compute posterior model weights: p(M_k | y) ∝ p(y | M_k) * p(M_k)
    log_weights = if prior_weights !== nothing
        log_mlls .+ log.(prior_weights)
    else
        log_mlls
    end
    weights = exp.(log_weights .- logsumexp(log_weights))

    # Construct model-averaged latent marginals
    latent_marginals = Vector{WeightedMixture}(undef, n_latent)
    for i in 1:n_latent
        components = [results[k].latent_marginals[i] for k in 1:K]
        latent_marginals[i] = WeightedMixture(components, weights)
    end

    return BMAResult(latent_marginals, weights, log_mlls)
end
