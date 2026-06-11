export evaluate_at_grid_point, create_weighted_mixtures

"""
    latent_mode(model, y, θ_natural_nt, ws) -> Vector{Float64}

Latent posterior mode x*(θ) at a single hyperparameter point. Used to warm-start
the per-grid-point Gaussian-approximation Newton solves during exploration: the
latent mode varies little across the integration grid, so seeding every point's
GA from x*(θ*) cuts its Newton iterations roughly in half (6 → 2–3 here) without
changing the converged marginals at each fixed θ.
"""
function latent_mode(model::LatentGaussianModel, y, θ_natural_nt, ws)
    prior_gmrf = latent_gmrf(model, ws, θ_natural_nt)
    obs_lik = model.observation_model(y; θ_natural_nt...)
    return collect(mean(gaussian_approximation(prior_gmrf, obs_lik)))
end

"""
    evaluate_at_grid_point(model::LatentGaussianModel, y, θ::WorkingHyperparameters; compute_marginals::Bool=false, marginalization_method=nothing, marginalization_indices=nothing)

Evaluate all quantities needed at a hyperparameter grid point.

This helper function encapsulates the expensive GMRF/GA logic and avoids code repetition.
Returns a NamedTuple with all computed quantities, designed to be splatted as kwargs to
accumulator callbacks.

# Arguments
- `model::LatentGaussianModel`: The INLA model specification
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
        model::LatentGaussianModel, y, θ::WorkingHyperparameters;
        ws, x0 = nothing,
        compute_marginals::Bool = false, marginalization_method = nothing, marginalization_indices = nothing,
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
            obs_lik = nothing,
            x_star_vbc = nothing,
        )
    end

    # Perform the expensive Gaussian Approximation and downstream evaluations.
    # Wrap in try-catch to handle numerical failures (PosDefException,
    # ZeroPivotException, SingularException) that can occur at extreme
    # hyperparameter values — especially with CCD design points far from the mode.
    try
        prior_gmrf = latent_gmrf(model, ws, θ_natural_nt)
        obs_lik = model.observation_model(y; θ_natural_nt...)
        ga = gaussian_approximation(prior_gmrf, obs_lik; x0 = x0)
        x_star = mean(ga)

        # VBC (compact mode): the corrected latent mean μ*, computed once. It is the
        # predictive point estimate (feeds obs-loglik → DIC and, via x_star_vbc, the
        # WAIC/CPO predictor location) and is threaded into the marginal builder.
        # π̃(θ|y) below stays on the GA mode x_star (μ0) — VBC corrects only the mean.
        x_star_vbc = nothing
        if marginalization_method isa VBCMarginal
            model.augmentation_info === nothing || throw(
                ArgumentError(
                    "VBCMarginal requires a compact (non-augmented) model; " *
                        "build it with augment=false, or use SimplifiedLaplace()."
                )
            )
            hub_I = latent_index_set_for_vbc(model, marginalization_method.index_set)
            x_star_vbc = first(
                vbc_correction(ga, obs_lik, prior_gmrf, hub_I; n_gh = marginalization_method.n_gh)
            )
        end

        # Marginalize BEFORE the log-density. `vbc_correction` above leaves the GA
        # workspace holding the Q_post selected inverse, which `marginalize`'s
        # `std(ga)` reuses for free. `hyperparameter_logpdf` below refactorizes the
        # workspace to Q_prior (for the prior log-det), clobbering that selinv — so
        # computing it first would force `marginalize` to recompute the selinv (one
        # extra Takahashi recursion per integration point). The marginal-likelihood
        # is a sum, so the term order is otherwise immaterial.
        # mean_override carries μ* for VBC and is ignored by every other method.
        marginal_result = compute_marginals ?
            marginalize(
                ga, obs_lik, log_prior_θ, marginalization_method, marginalization_indices;
                prior_gmrf = prior_gmrf, augmentation_info = model.augmentation_info,
                mean_override = x_star_vbc,
            ) : nothing

        # Compute log posterior density (π̃(θ|y)) at the GA mode.
        log_density = hyperparameter_logpdf(model, θ, y, ga; ws = ws)

        # Per-observation log-likelihoods at the predictive point estimate
        # (μ* under VBC, else the GA mode); feeds DIC via total_loglikelihood.
        loglik_point = x_star_vbc === nothing ? x_star : x_star_vbc
        obs_loglikelihoods = pointwise_loglik(loglik_point, obs_lik)
        total_loglikelihood = sum(obs_loglikelihoods)

        return (
            log_density = log_density,
            marginal_result = marginal_result,
            ga = ga,
            x_star = x_star,
            obs_loglikelihoods = obs_loglikelihoods,
            total_loglikelihood = total_loglikelihood,
            obs_lik = obs_lik,
            x_star_vbc = x_star_vbc,
        )
    catch e
        if e isa PosDefException || e isa LinearAlgebra.ZeroPivotException ||
                e isa LinearAlgebra.SingularException || e isa DomainError
            @warn "Numerical failure at hyperparams $(θ_natural_nt): $(typeof(e))"
            return (
                log_density = -Inf,
                marginal_result = nothing,
                ga = nothing,
                x_star = nothing,
                obs_loglikelihoods = nothing,
                total_loglikelihood = -Inf,
                obs_lik = nothing,
                x_star_vbc = nothing,
            )
        else
            rethrow(e)
        end
    end
end

"""
    create_weighted_mixtures(exploration::AbstractHyperparameterExploration)

Creates weighted mixture distributions from the exploration results.
This function becomes cleaner due to the improved `HyperparameterExploration` struct.

# Arguments
- `exploration::AbstractHyperparameterExploration`: Results from hyperparameter exploration with marginalization

# Returns
- `Vector{WeightedMixture}`: Weighted mixture distributions, one per marginalized variable

# Example
```julia
exploration = explore_hyperparameter_posterior(GridExplorationStrategy(), model, y, θ_star, marginalization_method, marginalization_indices)
final_marginals = create_weighted_mixtures(exploration)

# Access final marginal distributions
μ₁ = mean(final_marginals[1])
ci₁ = quantile(final_marginals[1], [0.025, 0.975])
```
"""
function create_weighted_mixtures(exploration::AbstractHyperparameterExploration)
    # Get the GridPoint objects for the integration points
    integration_points = exploration.grid_points[exploration.integration_indices]

    # Compute weights with log-sum-exp stabilization for numerical safety
    log_weights = [p.log_density for p in integration_points]
    weights = exp.(log_weights .- maximum(log_weights))
    weights ./= sum(weights)

    marginal_results = [p.marginal_result for p in integration_points]

    # Determine number of marginalized variables from first marginal result
    @assert !isempty(marginal_results) && all(!isnothing, marginal_results) "No valid marginal results found at integration points"
    n_vars = length(marginal_results[1].marginals)

    # Create weighted mixtures and weight-averaged KLD for each variable
    final_marginals = Vector{WeightedMixture}(undef, n_vars)
    kld_values = zeros(n_vars)

    for var_idx in 1:n_vars
        # Extract component distributions for this variable across all integration points
        components = [marginal_result.marginals[var_idx] for marginal_result in marginal_results]

        # Create weighted mixture for this variable
        final_marginals[var_idx] = WeightedMixture(components, weights)

        # Weight-average KLD across integration points
        for (w, mr) in zip(weights, marginal_results)
            kld_values[var_idx] += w * mr.kld_values[var_idx]
        end
    end

    return (marginals = final_marginals, kld = kld_values)
end
