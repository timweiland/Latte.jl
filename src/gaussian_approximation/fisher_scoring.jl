using LinearAlgebra
using GaussianMarkovRandomFields
using SparseArrays

export gaussian_approximation, fisher_scoring_step, compute_posterior_precision

"""
    compute_posterior_precision(prior_gmrf, obs_model, μ, θ, y) -> (Q, Q_chol)

Helper function to compute posterior precision matrix and its Cholesky factorization.
"""
function compute_posterior_precision(prior_gmrf, obs_model, μ, θ, y)
    Q_prior = precision_matrix(prior_gmrf)
    hess_obs = loghessian(obs_model, μ, θ, y)
    Q = Q_prior - hess_obs
    Q_chol = cholesky(Q)
    return Q, Q_chol
end

"""
    fisher_scoring_step(prior_gmrf, current_μ, obs_model, θ, y, options) -> (μ_new, Q_new, Q_new_chol, step_stats)

Perform one Fisher scoring step for Gaussian approximation.

# Arguments
- `prior_gmrf`: Prior GMRF distribution
- `current_μ`: Current estimate of the mode
- `obs_model`: Observation model implementing `loggrad` and `loghessian`
- `θ`: Hyperparameters for the observation model
- `y`: Observed data
- `options`: Newton options containing convergence tolerances

# Returns
- `μ_new`: Updated mode estimate
- `Q_new`: Updated precision matrix
- `Q_new_chol`: Cholesky factorization of updated precision matrix
- `step_stats`: NamedTuple with (newton_decrement, step_size, gradient_norm, early_convergence)
"""
function fisher_scoring_step(prior_gmrf, current_μ, obs_model, θ, y, options)
    # Get prior precision matrix and mean (computed once)
    Q_prior = precision_matrix(prior_gmrf)
    μ_prior = mean(prior_gmrf)

    # Compute observation model gradient (cheap)
    grad_obs = loggrad(obs_model, current_μ, θ, y)

    # Compute gradient of the posterior (negative because we minimize negative log-posterior)
    gradient = Q_prior * (current_μ - μ_prior) - grad_obs
    gradient_norm = norm(gradient)

    # Check for early convergence - avoid expensive Hessian and Cholesky if already converged
    if gradient_norm < options.tol_gradient
        # Already converged! Return current state without expensive computation
        step_stats = (
            newton_decrement = 0.0,
            step_size = 0.0,
            gradient_norm = gradient_norm,
            early_convergence = true,
        )

        return current_μ, nothing, nothing, step_stats
    end

    # Not yet converged - perform full Newton step with expensive computations
    Q_new, Q_new_chol = compute_posterior_precision(prior_gmrf, obs_model, current_μ, θ, y)

    # Solve for Newton step: Q_new * step = gradient
    newton_step = Q_new_chol \ gradient

    # Update estimate
    μ_new = current_μ - newton_step

    # Compute all statistics in one place
    newton_decrement = dot(gradient, newton_step)
    step_size = norm(newton_step)

    step_stats = (
        newton_decrement = newton_decrement,
        step_size = step_size,
        gradient_norm = gradient_norm,
        early_convergence = false,
    )

    return μ_new, Q_new, Q_new_chol, step_stats
end

"""
    gaussian_approximation(prior_gmrf, obs_model, θ, y; options=NewtonOptions()) -> NewtonResult

Find Gaussian approximation to the posterior using Fisher scoring.

This function finds the mode of the posterior distribution and constructs a Gaussian 
approximation around it using Fisher scoring (Newton-Raphson with Fisher information matrix).

# Arguments
- `prior_gmrf`: Prior GMRF distribution
- `obs_model`: Observation model implementing `loggrad` and `loghessian`
- `θ`: Hyperparameters for the observation model
- `y`: Observed data
- `options`: Optimization options (see `NewtonOptions`)

# Returns
A `NewtonResult` containing the Gaussian approximation and optimization statistics.

# Example
```julia
# Set up prior and observation model
prior = GMRF(μ_prior, Q_prior)
obs_model = ExponentialFamily(Bernoulli)

# Find Gaussian approximation
result = gaussian_approximation(prior, obs_model, Float64[], y)

# Extract results
posterior_gmrf = GMRF(result.μ, result.precision)
converged = result.converged
```
"""
function gaussian_approximation(prior_gmrf, obs_model, θ, y; options = NewtonOptions())
    # Initialize
    current_μ = copy(mean(prior_gmrf))
    stats = NewtonStats[]

    options.verbose && println("Starting Fisher scoring optimization...")

    # Keep track of previous iteration's results
    current_Q = nothing
    current_Q_chol = nothing

    for iter in 1:options.max_iterations
        # Perform Fisher scoring step (all quantities computed once here)
        μ_new, Q_new, Q_new_chol, step_stats = fisher_scoring_step(
            prior_gmrf, current_μ, obs_model, θ, y, options
        )

        # Extract statistics from the step
        newton_decrement = step_stats.newton_decrement
        step_size = step_stats.step_size
        gradient_norm = step_stats.gradient_norm
        early_convergence = get(step_stats, :early_convergence, false)

        # Check convergence (early convergence or standard criteria)
        converged = early_convergence || (
            gradient_norm < options.tol_gradient ||
                newton_decrement < options.tol_decrement
        )

        # Store statistics
        iter_stats = NewtonStats(iter, newton_decrement, step_size, gradient_norm, converged)
        push!(stats, iter_stats)

        # Handle early convergence after recording stats
        if early_convergence
            options.verbose && println("Converged early after $(iter - 1) iterations!")

            if current_Q === nothing
                current_Q, current_Q_chol = compute_posterior_precision(prior_gmrf, obs_model, current_μ, θ, y)
            end

            return NewtonResult(current_μ, current_Q, current_Q_chol, stats, true, iter)
        end

        if options.verbose
            println(
                "Iteration $iter: Newton decrement = $(newton_decrement), " *
                    "Gradient norm = $(gradient_norm), Step size = $(step_size)"
            )
        end

        # Update current state
        current_μ = μ_new
        current_Q = Q_new
        current_Q_chol = Q_new_chol

        # Check convergence
        if converged
            options.verbose && println("Converged after $iter iterations!")
            return NewtonResult(current_μ, current_Q, current_Q_chol, stats, true, iter)
        end

        # Check for too small steps
        if step_size < options.min_step_size
            options.verbose && println("Step size too small, stopping optimization.")
            return NewtonResult(current_μ, current_Q, current_Q_chol, stats, false, iter)
        end
    end

    options.verbose && println("Maximum iterations reached without convergence.")

    return NewtonResult(current_μ, current_Q, current_Q_chol, stats, false, options.max_iterations)
end
