export evaluate_at_grid_point, create_weighted_mixtures

"""
    evaluate_at_grid_point(model::INLAModel, y, θ::WorkingHyperparameters; compute_marginals::Bool=false, marginalization_method=nothing, marginalization_indices=nothing)

Evaluate all quantities needed at a hyperparameter grid point.

This helper function encapsulates the expensive GMRF/GA logic and avoids code repetition.
Returns a NamedTuple with all computed quantities, designed to be splatted as kwargs to
accumulator callbacks.

# Arguments
- `model::INLAModel`: The INLA model specification
- `y`: Observed data
- `θ::WorkingHyperparameters`: Hyperparameter values in working (unconstrained) space
- `compute_marginals::Bool=false`: Whether to compute marginals (expensive)
- `marginalization_method`: Method for computing marginals (required if compute_marginals=true)
- `marginalization_indices`: Variables to marginalize (required if compute_marginals=true)

# Returns
NamedTuple with:
- `log_density`: Log posterior π(θ|y)
- `marginal_result`: Latent marginals (if compute_marginals=true, else nothing)
- `ga`: Gaussian approximation GMRF
- `x_star`: Latent mode (mean of ga)
- `obs_loglikelihoods`: Per-observation log p(yᵢ|xᵢ,θ) (or nothing if unavailable)
- `total_loglikelihood`: Sum of obs_loglikelihoods (or scalar log-likelihood)
"""
function evaluate_at_grid_point(
        model::INLAModel, y, θ::WorkingHyperparameters;
        compute_marginals::Bool = false, marginalization_method = nothing, marginalization_indices = nothing
    )
    # Convert to natural space for model evaluation
    θ_natural = convert(NaturalHyperparameters, θ)
    θ_natural_nt = convert(NamedTuple, θ_natural)

    log_prior_θ = logpdf_prior(θ)
    if log_prior_θ === -Inf
        # Return minimal NamedTuple for rejected point
        return (
            log_density = -Inf,
            marginal_result = nothing,
            ga = nothing,
            x_star = nothing,
            obs_loglikelihoods = nothing,
            total_loglikelihood = -Inf,
        )
    end

    # Perform the expensive Gaussian Approximation once
    prior_gmrf = latent_gmrf(model, θ_natural_nt)
    obs_lik = model.observation_model(y; θ_natural_nt...)
    ga = nothing
    try
        ga = gaussian_approximation(prior_gmrf, obs_lik)
    catch PosDefException
        @warn "PosDefException at hyperparams $(θ_natural_nt)"
        return (
            log_density = -Inf,
            marginal_result = nothing,
            ga = nothing,
            x_star = nothing,
            obs_loglikelihoods = nothing,
            total_loglikelihood = -Inf,
        )
    end
    x_star = mean(ga)

    # Compute log posterior density using the pre-computed GA
    log_density = hyperparameter_logpdf(model, θ, y, ga)

    # Compute per-observation log-likelihoods
    obs_loglikelihoods = pointwise_loglik(x_star, obs_lik)
    total_loglikelihood = sum(obs_loglikelihoods)

    marginal_result = nothing
    if compute_marginals
        # Reuse the GA for marginalization
        marginal_result = marginalize(
            ga, obs_lik,
            log_prior_θ, marginalization_method, marginalization_indices;
            prior_gmrf = prior_gmrf
        )
    end

    return (
        log_density = log_density,
        marginal_result = marginal_result,
        ga = ga,
        x_star = x_star,
        obs_loglikelihoods = obs_loglikelihoods,
        total_loglikelihood = total_loglikelihood,
        obs_lik = obs_lik,
    )
end

"""
    create_weighted_mixtures(exploration::HyperparameterExploration)

Creates weighted mixture distributions from the exploration results.
This function becomes cleaner due to the improved `HyperparameterExploration` struct.

# Arguments
- `exploration::HyperparameterExploration`: Results from hyperparameter exploration with marginalization

# Returns
- `Vector{WeightedMixture}`: Weighted mixture distributions, one per marginalized variable

# Example
```julia
exploration = explore_hyperparameter_posterior(model, y, θ_star, marginalization_method, marginalization_indices)
final_marginals = create_weighted_mixtures(exploration)

# Access final marginal distributions
μ₁ = mean(final_marginals[1])
ci₁ = quantile(final_marginals[1], [0.025, 0.975])
```
"""
function create_weighted_mixtures(exploration::HyperparameterExploration)
    # Get the GridPoint objects for the integration points
    integration_points = exploration.grid_points[exploration.integration_indices]

    # The log densities are already normalized, so weights are simple to calculate
    log_weights = [p.log_density for p in integration_points]
    weights = exp.(log_weights)
    weights ./= sum(weights) # Re-normalize for numerical safety

    marginal_results = [p.marginal_result for p in integration_points]

    # Determine number of marginalized variables from first marginal result
    @assert !isempty(marginal_results) && all(!isnothing, marginal_results) "No valid marginal results found at integration points"
    n_vars = length(marginal_results[1].marginals)

    # Create weighted mixtures for each variable
    final_marginals = Vector{WeightedMixture}(undef, n_vars)

    for var_idx in 1:n_vars
        # Extract component distributions for this variable across all integration points
        components = [marginal_result.marginals[var_idx] for marginal_result in marginal_results]

        # Create weighted mixture for this variable
        final_marginals[var_idx] = WeightedMixture(components, weights)
    end

    return final_marginals
end
