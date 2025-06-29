using HCubature

export hyperparameter_marginal_logpdf

"""
    hyperparameter_marginal_logpdf(approx::HyperparameterPosteriorApproximation, 
                                   marginal_dim::Int, marginal_value::Float64;
                                   rtol::Float64=1e-3, atol::Float64=1e-6)

Compute the marginal posterior log-density π(θⱼ|y) at a single point by integrating 
over all other hyperparameters using HCubature.

Uses precomputed integration bounds and mode logpdf for efficiency and numerical stability.

# Arguments
- `approx::HyperparameterPosteriorApproximation`: The interpolated posterior approximation
- `marginal_dim::Int`: The dimension to compute the marginal for (1-indexed)
- `marginal_value::Float64`: The value at which to evaluate π(θⱼ|y)
- `rtol::Float64=1e-3`: Relative tolerance for HCubature integration
- `atol::Float64=1e-6`: Absolute tolerance for HCubature integration

# Returns
- `Float64`: Log marginal posterior density log π(θⱼ|y)

# Example
```julia
# For 2D posterior θ = [σ, ρ], compute π(σ=0.5|y)
log_marginal = hyperparameter_marginal_logpdf(posterior_approx, 1, 0.5)
```
"""
function hyperparameter_marginal_logpdf(approx::HyperparameterPosteriorApproximation, 
                                        marginal_dim::Int, marginal_value::Float64;
                                        rtol::Float64=1e-3, atol::Float64=1e-6)
    
    exploration = approx.exploration
    n_dims = length(exploration.mode)
    
    if marginal_dim < 1 || marginal_dim > n_dims
        throw(BoundsError(1:n_dims, marginal_dim))
    end
    
    if n_dims == 1
        # 1D case - no integration needed
        return approx([marginal_value])
    end
    
    # Get precomputed integration bounds
    integration_dims = [i for i in 1:n_dims if i != marginal_dim]
    bounds_lower = [exploration.integration_bounds[dim, 1] for dim in integration_dims]
    bounds_upper = [exploration.integration_bounds[dim, 2] for dim in integration_dims]
    
    # Use precomputed mode logpdf for log-sum-exp stability
    #max_log = exploration.transformation.mode_logpdf
    
    # Define stable integrand function for HCubature
    function stable_integrand(θ_other)
        # Reconstruct full parameter vector
        θ_full = Vector{Float64}(undef, n_dims)
        other_idx = 1
        for i in 1:n_dims
            if i == marginal_dim
                θ_full[i] = marginal_value
            else
                θ_full[i] = θ_other[other_idx]
                other_idx += 1
            end
        end
        
        log_posterior = approx(θ_full)
        return exp(log_posterior)
    end
    
    # Perform multidimensional integration using HCubature
    integral_result, _ = hcubature(stable_integrand, bounds_lower, bounds_upper; 
                                  rtol=rtol, atol=atol)
    
    # Convert back to log space
    log_marginal = log(integral_result)
    
    return log_marginal
end