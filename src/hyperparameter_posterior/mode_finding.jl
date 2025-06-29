"""
Mode finding functions for hyperparameter posterior optimization.

This module contains the core functions for finding the mode of the hyperparameter
posterior π(θ | y) and computing the associated reparameterization.
"""

using LinearAlgebra
using Optim
using FiniteDiff

export hyperparameter_logpdf, find_hyperparameter_mode

"""
    hyperparameter_logpdf(model::INLAModel, θ, y)

Evaluate log π(θ | y) ∝ log π(θ) + log π(x*(θ), θ, y) - log π̃_G(x*(θ) | θ, y)

This is the INLA approximation to the hyperparameter posterior.
"""
function hyperparameter_logpdf(model::INLAModel, θ, y)
    # Check if θ is in support of the hyperparameter prior
    if !insupport(model.hyperparameter_prior.free_distribution, θ)
        return -Inf  # Return -Inf log-density for points outside support
    end

    θ_named = to_named(θ, model.hyperparameter_prior)
    # Get latent field prior for this θ
    x_prior = latent_gmrf(model, θ_named)
    
    # Find Gaussian approximation
    result = gaussian_approximation(x_prior, model.observation_model, θ_named, y)
    x_G = to_gmrf(result)
    x_star = mean(x_G)
    
    # Compute INLA approximation: log π(x*, θ, y) - log π̃_G(x* | θ, y)
    joint_logpdf = log_joint_density(model, x_star, θ, y)
    gaussian_logpdf = logpdf(x_G, x_star)
    
    return joint_logpdf - gaussian_logpdf
end

"""
    find_hyperparameter_mode(model::INLAModel, y; method=BFGS(), collect_points=true)

Find the mode θ* of the hyperparameter posterior π(θ | y).

# Arguments
- `model`: INLA model specification
- `y`: Observed data
- `method`: Optimization method (from Optim.jl)
- `collect_points`: Whether to collect intermediate points during optimization

# Returns
- `θ_star`: The posterior mode
- `mode_points`: Points evaluated during optimization (if collect_points=true)
- `mode_logdensities`: Log-densities at mode_points (if collect_points=true)
"""
function find_hyperparameter_mode(model::INLAModel, y; method=BFGS(), collect_points=true)
    
    # Storage for optimization path points
    mode_points = Vector{Float64}[]
    mode_logdensities = Float64[]
    
    # Objective function (negative log-density for minimization)
    function objective(θ)
        # Check if θ is in support of the hyperparameter prior
        if !insupport(model.hyperparameter_prior.free_distribution, θ)
            return Inf  # Return infinite objective for points outside support
        end
        
        logpdf_val = hyperparameter_logpdf(model, θ, y)
        
        if collect_points && isfinite(logpdf_val)
            push!(mode_points, copy(θ))
            push!(mode_logdensities, logpdf_val)
        end
        
        return -logpdf_val  # Minimize negative log-density
    end
    
    # Initial guess (mode of hyperparameter prior)
    θ_init = mode(model.hyperparameter_prior.free_distribution)
    if !isa(θ_init, Vector)
        θ_init = [θ_init]  # Handle scalar case
    end
    
    # Efficient INLA hyperparameter optimization tolerances
    options = Optim.Options(
        f_reltol = 1e-3,     # Relative tolerance in objective (log-likelihood) changes
        f_abstol = 1e-6,     # Absolute tolerance in objective changes  
        g_abstol = 1e-6,     # Gradient tolerance (less strict than default 1e-8)
        x_reltol = 1e-3,     # Relative tolerance in parameter changes
        iterations = 1000,   # Reasonable max iterations
        show_trace = false,  # Set to true for debugging
        allow_f_increases = true  # Allow occasional increases during search
    )
    result = Optim.optimize(objective, θ_init, method, options)
    
    if !Optim.converged(result)
        @warn "Hyperparameter mode optimization did not converge"
    end
    
    θ_star = Optim.minimizer(result)
    
    if collect_points
        return θ_star, mode_points, mode_logdensities
    else
        return θ_star, nothing, nothing
    end
end

