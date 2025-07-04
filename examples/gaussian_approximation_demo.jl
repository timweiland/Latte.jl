# Gaussian Approximation Demo
# This demonstrates how to use the gaussian_approximation functionality

using IntegratedNestedLaplace
using GaussianMarkovRandomFields
using LDLFactorizations
using Distributions
using LinearAlgebra
using SparseArrays

# Create a simple autoregressive prior
n = 20
ϕ = 0.8
σ² = 1.0

# Create precision matrix for AR(1) process
diag_main = [1.0; fill(1 + ϕ^2, n - 2); 1.0] ./ σ²
diag_off = fill(-ϕ, n - 1) ./ σ²
Q_prior = spdiagm(0 => diag_main, -1 => diag_off, 1 => diag_off)

# Prior GMRF
μ_prior = zeros(n)
prior_gmrf = GMRF(μ_prior, Q_prior, CholeskySolverBlueprint())

println("Prior GMRF created with dimension: ", length(μ_prior))

# Generate some synthetic data
x_true = rand(prior_gmrf)
obs_model = ExponentialFamily(Bernoulli)
θ_named = NamedTuple()  # Bernoulli has no hyperparameters
y_obs = rand(likelihood(obs_model, x_true, θ_named))

println("Generated ", length(y_obs), " Bernoulli observations")
println("Proportion of 1s: ", mean(y_obs))

# Find Gaussian approximation to posterior
println("\nFinding Gaussian approximation...")

options = NewtonOptions(
    max_iterations = 20,
    tol_gradient = 1.0e-6,
    tol_decrement = 1.0e-8,
    verbose = true
)

result = gaussian_approximation(prior_gmrf, obs_model, θ_named, y_obs; options = options)

# Display results
println("\n" * "="^50)
summary(result)

# Convert to GMRF for further use
posterior_gmrf = to_gmrf(result)

println("\nPosterior mean (first 5 elements): ", mean(posterior_gmrf)[1:5])
println("Prior mean (first 5 elements): ", mean(prior_gmrf)[1:5])

# Compare with true values
println("\nTrue values (first 5 elements): ", x_true[1:5])
println("Posterior mode (first 5 elements): ", result.μ[1:5])
println("Prior mean (first 5 elements): ", mean(prior_gmrf)[1:5])

println("\n✓ Gaussian approximation demo completed successfully!")
