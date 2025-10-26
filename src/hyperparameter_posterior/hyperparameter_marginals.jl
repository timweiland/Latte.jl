using HCubature

export hyperparameter_marginal_logpdf

"""
    hyperparameter_marginal_logpdf(approx::HyperparameterPosteriorApproximation,
                                   marginal_dim::Int, marginal_value::Float64;
                                   rtol::Float64=1e-3, atol::Float64=1e-6)

Compute the marginal posterior log-density π(θⱼ|y) at a single point by integrating
over all other hyperparameters in working space using HCubature.

The input `marginal_value` should be in **natural space** (constrained space where users
interact with hyperparameters). The function converts to working space and integrates
in working space for numerical stability.

# Arguments
- `approx::HyperparameterPosteriorApproximation`: The interpolated posterior approximation
- `marginal_dim::Int`: The dimension to compute the marginal for (1-indexed)
- `marginal_value::Float64`: The value at which to evaluate π(θⱼ|y) in **natural space**
- `rtol::Float64=1e-3`: Relative tolerance for HCubature integration
- `atol::Float64=1e-6`: Absolute tolerance for HCubature integration

# Returns
- `Float64`: Unnormalized log marginal posterior density log π(θⱼ|y)

# Example
```julia
# For 2D posterior θ = [σ, ρ], compute π(σ=0.5|y) where 0.5 is in natural space
log_marginal = hyperparameter_marginal_logpdf(posterior_approx, 1, 0.5)
```
"""
function hyperparameter_marginal_logpdf(
        approx::HyperparameterPosteriorApproximation,
        marginal_dim::Int, marginal_value::Float64;
        rtol::Float64 = 1.0e-3, atol::Float64 = 1.0e-6
    )

    exploration = approx.exploration
    n_dims = length(exploration.grid_points[1].θ)

    if marginal_dim < 1 || marginal_dim > n_dims
        throw(BoundsError(1:n_dims, marginal_dim))
    end

    spec = exploration.transform.θ_star.spec

    if n_dims == 1
        # 1D case - no integration needed
        # marginal_value is in natural space
        θ_natural = NaturalHyperparameters([marginal_value], spec)
        # Use approx which handles working/natural space conversion and Jacobian correction
        return approx(θ_natural)
    end

    # Convert marginal_value from natural to working space
    # Get the hyperparameter from the free parameters
    free_hps = collect(values(spec.free))
    η_j = free_hps[marginal_dim].transform(marginal_value)

    # Get integration bounds (in working space)
    bounds = exploration.integration_bounds
    integration_dims = [i for i in 1:n_dims if i != marginal_dim]
    bounds_lower = [bounds[dim, 1] for dim in integration_dims]
    bounds_upper = [bounds[dim, 2] for dim in integration_dims]

    # Define integrand function that integrates in working space
    function working_space_integrand(η_other)
        # Reconstruct full working space vector
        η_full = Vector{Float64}(undef, n_dims)
        other_idx = 1
        for i in 1:n_dims
            if i == marginal_dim
                η_full[i] = η_j  # Fixed to the working space value
            else
                η_full[i] = η_other[other_idx]
                other_idx += 1
            end
        end

        try
            θ_working = WorkingHyperparameters(η_full, spec)
            log_posterior = approx(θ_working)
            return exp(log_posterior)
        catch
            return 0.0
        end
    end

    # Perform multidimensional integration in working space
    integral_result, _ = hcubature(
        working_space_integrand, bounds_lower, bounds_upper;
        rtol = rtol, atol = atol
    )

    # Convert back to log space
    log_marginal = log(integral_result)

    # Add Jacobian correction to convert from working space to natural space
    free_hps = collect(values(spec.free))
    hp_j = free_hps[marginal_dim]
    logdetjac_j = Bijectors.logabsdetjac(hp_j.transform, marginal_value)

    return log_marginal + logdetjac_j
end
