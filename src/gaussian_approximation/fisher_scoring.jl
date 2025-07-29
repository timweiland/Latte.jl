using LinearAlgebra
using GaussianMarkovRandomFields
using SparseArrays
using NonlinearSolve

export gaussian_approximation

function neg_log_posterior(prior_gmrf, obs_lik, x)
    return -logpdf(prior_gmrf, x) - loglik(obs_lik, x)
end

function ∇ₓ_neg_log_posterior(prior_gmrf, obs_lik, x)
    return -gradlogpdf(prior_gmrf, x) - loggrad(obs_lik, x)
end

function ∇²ₓ_neg_log_posterior(prior_gmrf, obs_lik, x)
    return precision_matrix(prior_gmrf) - loghessian(obs_lik, x)
end

"""
    gaussian_approximation(prior_gmrf, obs_lik) -> AbstractGMRF

Find Gaussian approximation to the posterior using Fisher scoring.

This function finds the mode of the posterior distribution and constructs a Gaussian 
approximation around it using Fisher scoring (Newton-Raphson with Fisher information matrix).

# Arguments
- `prior_gmrf`: Prior GMRF distribution for the latent field
- `obs_lik`: Materialized observation likelihood (contains data and hyperparameters)

# Returns
- `posterior_gmrf::GMRF`: Gaussian approximation to the posterior p(x | θ, y)

# Example
```julia
# Set up components (done at higher level)
prior_gmrf = GMRF(μ_prior, Q_prior)
obs_model = ExponentialFamily(Poisson)
obs_lik = obs_model(y; rate_scale=1.2)  # Materialized once

# Find Gaussian approximation - returns a GMRF
posterior_gmrf = gaussian_approximation(prior_gmrf, obs_lik)
```
"""
function gaussian_approximation(prior_gmrf, obs_lik; initial_guess = mean(prior_gmrf))
    nlf = NonlinearFunction{true}(
        (dg, u, p) -> (dg .= ∇ₓ_neg_log_posterior(p[1], p[2], u)),
        jac = (dh, u, p) -> (dh .= ∇²ₓ_neg_log_posterior(p[1], p[2], u)),
        jac_prototype = precision_matrix(prior_gmrf)
    )
    prob = NonlinearProblem(nlf, initial_guess, (prior_gmrf, obs_lik))

    # Try to extract permutation if available, otherwise use default
    perm = try
        prior_gmrf.solver.precision_chol.p
    catch
        nothing
    end
    cho_solver = LinearSolve.CHOLMODFactorization(; perm = perm)

    sol = solve(
        prob,
        NewtonRaphson(linsolve = cho_solver), abstol = 1.0e-6, reltol = 1.0e-6
    )
    x_star = sol.u
    return GMRF(x_star, ∇²ₓ_neg_log_posterior(prior_gmrf, obs_lik, x_star))
end
