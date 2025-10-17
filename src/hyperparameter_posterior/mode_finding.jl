"""
Mode finding functions for hyperparameter posterior optimization.

This module contains the core functions for finding the mode of the hyperparameter
posterior π(θ | y) and computing the associated reparameterization.
"""

using LinearAlgebra
using Optim
using FiniteDiff
using Distributions

export hyperparameter_logpdf, find_hyperparameter_mode, initial_hyperparameter_guess

"""
    initial_hyperparameter_guess(spec::HyperparameterSpec)

Compute an initial guess for hyperparameter optimization in working space.

For each hyperparameter:
- If `prior_space=:natural`: Compute mode in natural space and transform to working space
- If `prior_space=:working`: Use mode directly from prior

Returns a vector of initial guesses in working space.
"""
function initial_hyperparameter_guess(spec::HyperparameterSpec)
    return [_initial_guess_for_hyperparameter(hp) for hp in values(spec.free)]
end

"""
    _robust_initial_value(dist::Distribution)

Compute a robust initial value for optimization from a distribution.
Uses mode by default, but can be specialized for distributions with boundary modes.
"""
_robust_initial_value(dist::Distribution) = mode(dist)

# Specializations for distributions with boundary modes
_robust_initial_value(dist::Exponential) = mean(dist)  # mode=0, use mean instead

function _initial_guess_for_hyperparameter(hp::Hyperparameter{T, S}) where {T, S}
    if S == :natural
        # Prior was specified in natural space and transformed to working space
        # hp.prior is a TransformedDistribution, extract the base distribution
        base_dist = hp.prior.dist
        initial_natural = _robust_initial_value(base_dist)
        # Transform to working space
        return hp.transform(initial_natural)
    else
        # Prior already in working space, use robust initial value
        return _robust_initial_value(hp.prior)
    end
end

"""
    hyperparameter_logpdf(model::INLAModel, θ, y, ga=nothing)

Evaluate log π(θ | y) ∝ log π(θ) + log π(x*(θ), θ, y) - log π̃_G(x*(θ) | θ, y)

This is the INLA approximation to the hyperparameter posterior.

# Arguments
- `model::INLAModel`: The INLA model specification
- `θ`: Hyperparameter vector in working space (unconstrained)
- `y`: Observed data
- `ga`: Optional pre-computed Gaussian approximation (GMRF object). If `nothing`, will be computed.

# Details
- `θ` is in working (unconstrained) space for optimization
- Automatically converts to natural space for model evaluation via `log_joint_density`
- Prior includes Jacobian correction for transformations
"""
function hyperparameter_logpdf(model::INLAModel, θ, y, ga = nothing)
    spec = model.hyperparameter_spec

    # Convert θ to natural space for model evaluation
    θ_working = to_named_tuple(θ, spec)
    θ_natural = to_natural(θ_working, spec)

    # Use provided Gaussian approximation or compute it
    if ga === nothing
        # Get latent field prior for this θ
        x_prior = latent_gmrf(model, θ_natural)

        # Find Gaussian approximation
        obs_lik = model.observation_model(y; θ_natural...)
        x_G = gaussian_approximation(x_prior, obs_lik)
    else
        x_G = ga
    end

    x_star = mean(x_G)

    # Compute INLA approximation: log π(x*, θ, y) - log π̃_G(x* | θ, y)
    # log_joint_density handles the transformation and Jacobian
    joint_logpdf = log_joint_density(model, x_star, θ, y)
    gaussian_logpdf = logpdf(x_G, x_star)

    return joint_logpdf - gaussian_logpdf
end

"""
    find_hyperparameter_mode(model::INLAModel, y; method=BFGS(), collect_points=true, progress_callback=nothing)

Find the mode θ* of the hyperparameter posterior π(θ | y).

# Arguments
- `model`: INLA model specification
- `y`: Observed data
- `method`: Optimization method (from Optim.jl)
- `collect_points`: Whether to collect intermediate points during optimization
- `progress_callback`: Optional function for progress updates with signature `f(; kwargs...)`

# Returns
- `θ_star`: The posterior mode in working space (for use in exploration/integration)
- `mode_points`: Points evaluated during optimization in working space (if collect_points=true)
- `mode_logdensities`: Log-densities at mode_points (if collect_points=true)

# Details
Optimization and results are in working (unconstrained) space for use in subsequent
exploration and integration steps. Convert to natural space only for final user output.
"""
function find_hyperparameter_mode(model::INLAModel, y; method = BFGS(), collect_points = true, progress_callback = nothing)
    spec = model.hyperparameter_spec

    # Storage for optimization path points
    mode_points = Vector{Float64}[]
    mode_logdensities = Float64[]

    # Objective function (negative log-density for minimization)
    function objective(θ)
        logpdf_val = 0.0
        try
            logpdf_val = hyperparameter_logpdf(model, θ, y)
        catch ZeroPivotException
            return Inf
        end


        if collect_points && isfinite(logpdf_val)
            push!(mode_points, copy(θ))
            push!(mode_logdensities, logpdf_val)
        end

        return -logpdf_val  # Minimize negative log-density
    end

    # Initial guess (mode of hyperparameter prior in working space)
    θ_init = initial_hyperparameter_guess(spec)

    # Handle progress callback
    if progress_callback === nothing
        progress_callback = (; kwargs...) -> nothing
    end

    # Create Optim callback for progress tracking
    optim_callback = function (state)
        progress_callback(
            iteration = state.iteration,
            objective = state.value,
            gradient_norm = state.g_norm
        )
        return false  # Continue optimization
    end

    # Efficient INLA hyperparameter optimization tolerances
    options = Optim.Options(
        f_reltol = 1.0e-3,     # Relative tolerance in objective (log-likelihood) changes
        f_abstol = 1.0e-6,     # Absolute tolerance in objective changes
        g_abstol = 1.0e-6,     # Gradient tolerance (less strict than default 1e-8)
        x_reltol = 1.0e-3,     # Relative tolerance in parameter changes
        iterations = 1000,   # Reasonable max iterations
        show_trace = false,  # Set to true for debugging
        allow_f_increases = true,  # Allow occasional increases during search
        callback = optim_callback  # Add progress callback
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
