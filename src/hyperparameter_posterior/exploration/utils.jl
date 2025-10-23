export evaluate_logpdf_and_marginals, create_weighted_mixtures

"""
    evaluate_logpdf_and_marginals(model::INLAModel, y, θ::Vector{Float64}; compute_marginals::Bool=false, marginalization_method=nothing, marginalization_indices=nothing)

Computes the log-posterior and optionally the latent field marginals at a given θ.
This helper function encapsulates the expensive GMRF/GA logic and avoids code repetition.

# Arguments
- `model::INLAModel`: The INLA model specification
- `y`: Observed data
- `θ::Vector{Float64}`: Hyperparameter values in unconstrained space
- `compute_marginals::Bool=false`: Whether to compute marginals (expensive)
- `marginalization_method`: Method for computing marginals (required if compute_marginals=true)
- `marginalization_indices`: Variables to marginalize (required if compute_marginals=true)

# Returns
- `log_density::Float64`: Log posterior density π(θ|y)
- `marginal_result::Union{Nothing, MarginalResult}`: Marginal results (if compute_marginals=true)
"""
function evaluate_logpdf_and_marginals(
        model::INLAModel, y, θ::Vector{Float64};
        compute_marginals::Bool = false, marginalization_method = nothing, marginalization_indices = nothing
    )
    spec = model.hyperparameter_spec


    # Convert θ vector to natural space
    θ_natural = to_named_tuple(θ, spec)

    log_prior_θ = logpdf_prior(θ_natural, spec)
    if log_prior_θ === -Inf
        return -Inf, nothing
    end

    # Perform the expensive Gaussian Approximation once
    prior_gmrf = latent_gmrf(model, θ_natural)
    obs_lik = model.observation_model(y; θ_natural...)
    ga = gaussian_approximation(prior_gmrf, obs_lik)

    # Compute log posterior density using the pre-computed GA
    log_density = hyperparameter_logpdf(model, θ_natural, y, ga)

    marginal_result = nothing
    if compute_marginals
        # Reuse the GA for marginalization
        log_prior_θ = logpdf_prior(θ_natural, spec)
        # Materialize observation likelihood with data and hyperparameters
        obs_lik = model.observation_model(y; θ_natural...)
        marginal_result = marginalize(
            ga, obs_lik,
            log_prior_θ, marginalization_method, marginalization_indices;
            prior_gmrf = prior_gmrf
        )
    end

    return log_density, marginal_result
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
