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

Returns `WorkingHyperparameters` with initial guesses from the prior modes/means.

# Details
Since priors are stored in working space, we directly use the mode (or mean for distributions
with boundary modes like Exponential) from the working-space prior.
"""
function initial_hyperparameter_guess(spec::HyperparameterSpec)
    θ_init = [_initial_guess_for_hyperparameter(hp) for hp in values(spec.free)]
    return WorkingHyperparameters(θ_init, spec)
end

"""
    _robust_initial_value(dist::Distribution)

Compute a robust initial value for optimization from a distribution.
Uses mode by default, but can be specialized for distributions with boundary modes.
"""
_robust_initial_value(dist::Distribution) = mode(dist)

# Specializations for distributions with boundary modes
_robust_initial_value(dist::Exponential) = mean(dist)  # mode=0, use mean instead

# Specialization for TransformedDistribution (when prior was transformed to working space)
function _robust_initial_value(dist::Bijectors.TransformedDistribution)
    # Get robust initial value from base distribution
    base_value = _robust_initial_value(dist.dist)
    # Apply the transformation to get value in working space
    return dist.transform(base_value)
end

function _initial_guess_for_hyperparameter(hp::Hyperparameter{T, S}) where {T, S}
    # Prior is always stored in working space now
    # Just extract the mode/mean from the working-space prior
    return _robust_initial_value(hp.prior)
end

"""
    hyperparameter_logpdf(model::INLAModel, θ, y, ga=nothing)

Evaluate log π(θ | y) ∝ log π(θ) + log π(x*(θ), θ, y) - log π̃_G(x*(θ) | θ, y)

This is the INLA approximation to the hyperparameter posterior.

# Arguments
- `model::INLAModel`: The INLA model specification
- `θ`: Hyperparameters (WorkingHyperparameters or NaturalHyperparameters)
- `y`: Observed data
- `ga`: Optional pre-computed Gaussian approximation (GMRF object). If `nothing`, will be computed.

# Details
- Main implementation is for `WorkingHyperparameters` (working space)
- `NaturalHyperparameters` converts to working space and adds Jacobian correction
"""
function hyperparameter_logpdf(
        model::INLAModel, θ::WorkingHyperparameters, y, ga = nothing;
        ws, x0 = nothing,
    )
    # Compute INLA approximation: log π(x*, θ, y) - log π̃_G(x* | θ, y)

    # Evaluate prior in working space
    log_prior_θ = logpdf_prior(θ)

    if log_prior_θ === -Inf
        return -Inf
    end

    # Convert to natural space for model evaluation
    θ_nt = convert(NamedTuple, convert(NaturalHyperparameters, θ))

    obs_lik = model.observation_model(y; θ_nt...)
    latent_prior = latent_gmrf(model, ws, θ_nt)

    # Use provided Gaussian approximation or compute it
    if ga === nothing
        # Find Gaussian approximation (warm-start from x0 if provided)
        x_G = gaussian_approximation(latent_prior, obs_lik; x0 = x0)
    else
        x_G = ga
    end

    x_star = mean(x_G)

    log_prior_x = logpdf(latent_prior, x_star)
    log_likelihood = loglik(x_star, obs_lik)

    joint_logpdf = log_prior_θ + log_prior_x + log_likelihood
    if !isfinite(joint_logpdf)
        return -Inf
    end

    gaussian_logpdf = logpdf(x_G, x_star)
    if !isfinite(gaussian_logpdf)
        return -Inf
    end

    return joint_logpdf - gaussian_logpdf
end

function hyperparameter_logpdf(model::INLAModel, θ::NaturalHyperparameters, y, ga = nothing; ws)
    # Convert to working space and evaluate
    θ_working = convert(WorkingHyperparameters, θ)
    log_p_working = hyperparameter_logpdf(model, θ_working, y, ga; ws = ws)

    # Add Jacobian correction to get natural-space density
    return log_p_working + logdetjac(θ)
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
- `θ_star`: The posterior mode in working space (WorkingHyperparameters)
- `mode_points`: WorkingHyperparameters evaluated during optimization (if collect_points=true)
- `mode_logdensities`: Log-densities at mode_points (if collect_points=true)

# Details
Optimization is performed in working (unconstrained) space. The mode is returned in working space.
"""
function find_hyperparameter_mode(
        model::INLAModel, y;
        method = BFGS(), collect_points = true, progress_callback = nothing,
        diff_strategy::DifferentiationStrategy = ADStrategy()
    )
    spec = model.hyperparameter_spec

    # Storage for optimization path points
    mode_points = WorkingHyperparameters[]
    mode_logdensities = Float64[]

    # Initial guess (mode of hyperparameter prior in working space)
    θ_init = initial_hyperparameter_guess(spec)
    θ_init_nt = convert(NamedTuple, convert(NaturalHyperparameters, θ_init))

    # One-time symbolic factorization — reused across every θ iteration below.
    ws = make_workspace(model.latent_prior; θ_init_nt...)

    # Objective function (negative log-density for minimization)
    function objective(θ_vec)
        θ = WorkingHyperparameters(θ_vec, spec)
        logpdf_val = 0.0
        try
            logpdf_val = hyperparameter_logpdf(model, θ, y; ws = ws)
        catch e
            return Inf
        end

        if !isfinite(logpdf_val)
            return Inf
        end

        if collect_points
            push!(mode_points, WorkingHyperparameters(copy(θ_vec), spec))
            push!(mode_logdensities, logpdf_val)
        end

        return -logpdf_val  # Minimize negative log-density
    end

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

    result = _run_optimization(diff_strategy, objective, model, y, spec, θ_init, ws, method, options)

    if !Optim.converged(result)
        @warn "Hyperparameter mode optimization did not converge"
    end

    θ_star = WorkingHyperparameters(Optim.minimizer(result), spec)

    if collect_points
        return θ_star, mode_points, mode_logdensities
    else
        return θ_star, nothing, nothing
    end
end

function _run_optimization(::FiniteDiffStrategy, objective, model, y, spec, θ_init, ws, method, options)
    return Optim.optimize(objective, θ_init.θ, method, options)
end

function _run_optimization(strategy::ADStrategy, objective, model, y, spec, θ_init, ws, method, options)
    # Clean objective for AD (no side effects, safe for Dual numbers).
    # ws is captured by the closure; AD flows through the numeric values only.
    function objective_clean(θ_vec)
        θ = WorkingHyperparameters(θ_vec, spec)
        logpdf_val = try
            hyperparameter_logpdf(model, θ, y; ws = ws)
        catch
            oftype(θ_vec[1], -Inf)
        end
        return isfinite(logpdf_val) ? -logpdf_val : oftype(logpdf_val, Inf)
    end

    # Explicit gradient via DifferentiationInterface
    function gradient!(G, θ_vec)
        return copyto!(G, DifferentiationInterface.gradient(objective_clean, strategy.backend, θ_vec))
    end

    return Optim.optimize(objective, gradient!, θ_init.θ, method, options)
end
