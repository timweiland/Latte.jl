# Autoregressive Bernoulli Model with INLA
#
# This example demonstrates INLA inference for a Bernoulli observation model
# with an autoregressive AR(1) prior on the latent field.
#
# The model is:
# - Latent field: x ~ AR(1) with precision matrix Q
# - Observations: y_i ~ Bernoulli(logistic(x_i))
# - Hyperparameter: μ₀ (initial value of AR process)

using Latte
using GaussianMarkovRandomFields
using LDLFactorizations
using Distributions
using LinearAlgebra
using StatsFuns
using Plots
using Turing
using SparseArrays
using StatsPlots

# Model parameters
ts = 1:50
N = length(ts)
ϕ = 0.85        # AR(1) coefficient
Λ₀ = 1.0        # Initial precision
Λ = 1.0         # Process precision

# Prior for hyperparameter μ₀
μ_prior = Uniform(0, 3)
μ₀_gt = rand(μ_prior)  # True value for simulation

# Construct AR(1) precision matrix
diag_main = [Λ₀; fill(Λ + ϕ^2, N - 2); Λ]
diag_off = fill(-ϕ, N - 1)
Q = spdiagm(0 => diag_main, -1 => diag_off, 1 => diag_off)

# Function to create latent GMRF given hyperparameter μ₀
function latent_gmrf(θ)
    μ₀ = θ[1]
    μ = [ϕ^(i - 1) * μ₀ for i in eachindex(ts)]  # AR(1) mean
    return (μ, Q)
end
# Observation model (Bernoulli with logit link)
obs_model = ExponentialFamily(Bernoulli)

θ_prior = product_distribution(μ_prior)
inla_model = LatentGaussianModel(θ_prior, latent_gmrf, obs_model)

# Generate synthetic data
println("Generating synthetic data...")
x_gt = rand(GMRF(latent_gmrf(μ₀_gt)...))
θ_obs = NamedTuple()  # Bernoulli has no hyperparameters
ys_gt = rand(likelihood(obs_model, x_gt, θ_obs))

println("Generated $(length(ys_gt)) observations")
println("Proportion of 1s: $(round(mean(ys_gt), digits = 3))")
println("True μ₀: $(round(μ₀_gt, digits = 3))")

# Define log joint density for INLA
function log_joint(x, μ, y; x_prior = nothing)
    if x_prior === nothing
        x_prior = latent_gmrf(μ)
    end
    obs_lik = obs_model(y; θ_obs...)
    return logpdf(μ_prior, μ) + logpdf(x_prior, x) + loglik(obs_lik, x)
end

function xi_marginal(i, xis, θ; y)
    x_prior = latent_gmrf(θ)
    obs_lik = obs_model(y; θ_obs...)
    x_G = gaussian_approximation(x_prior, obs_lik)

    la_cache = LaplaceApproximationCache(x_G, obs_model, i)
    res = Float64[]

    logθ = logpdf(μ_prior, θ)
    for xi in xis
        push!(res, evaluate_laplace_logpdf(la_cache, xi, θ, y, logθ))
    end
    return res
end

# INLA marginal posterior for hyperparameter
function θ_posterior(θ; y)
    x_prior = latent_gmrf(θ)
    obs_lik = obs_model(y; θ_obs...)
    x_G = gaussian_approximation(x_prior, obs_lik)
    x_star = mean(x_G)

    return log_joint(x_star, θ, y; x_prior = x_prior) - logpdf(x_G, x_star)
end

# MCMC model for comparison
@model function inla_demo(y)
    μ_prior ~ Uniform(0, 3)
    x ~ latent_gmrf(μ_prior)
    y ~ likelihood(obs_model, x, θ_obs)
end

println("\n" * "="^60)
println("Running MCMC for comparison...")
println("="^60)

# Run MCMC
chain = sample(inla_demo(ys_gt), NUTS(), 1000, progress = true)

println("\n" * "="^60)
println("MCMC vs INLA Comparison")
println("="^60)

# Extract MCMC results
μ_mcmc_samples = vec(chain[:μ_prior])

println("MCMC Results for μ₀:")
mcmc_mean = mean(μ_mcmc_samples)
mcmc_std = std(μ_mcmc_samples)
mcmc_ci = quantile(μ_mcmc_samples, [0.025, 0.975])

println("  Mean: $(round(mcmc_mean, digits = 3))")
println("  Std:  $(round(mcmc_std, digits = 3))")
println("  95% CI: [$(round(mcmc_ci[1], digits = 3)), $(round(mcmc_ci[2], digits = 3))]")

# Compute INLA approximation
println("\nComputing INLA approximation...")

μ_grid = range(-3, 3, length = 100)
θ_posterior_vals = [θ_posterior(μ; y = ys_gt) for μ in μ_grid]

# Normalize to get posterior density
posterior_weights = exp.(θ_posterior_vals .- maximum(θ_posterior_vals))
posterior_weights ./= sum(posterior_weights) * step(μ_grid)

# Compute INLA moments
inla_mean = sum(μ_grid .* posterior_weights) * step(μ_grid)
inla_var = sum((μ_grid .- inla_mean) .^ 2 .* posterior_weights) * step(μ_grid)
inla_std = sqrt(inla_var)

# Compute INLA credible interval
cum_weights = cumsum(posterior_weights * step(μ_grid))
inla_q025 = μ_grid[findfirst(cum_weights .>= 0.025)]
inla_q975 = μ_grid[findfirst(cum_weights .>= 0.975)]

println("INLA Results for μ₀:")
println("  Mean: $(round(inla_mean, digits = 3))")
println("  Std:  $(round(inla_std, digits = 3))")
println("  95% CI: [$(round(inla_q025, digits = 3)), $(round(inla_q975, digits = 3))]")

# Create comparison plot
println("\nCreating comparison plot...")

p1 = density(
    μ_mcmc_samples, label = "MCMC", alpha = 0.7, linewidth = 2,
    title = "Posterior of μ₀", xlabel = "μ₀", ylabel = "Density"
)
plot!(p1, μ_grid, posterior_weights, label = "INLA", alpha = 0.7, linewidth = 2)
vline!(p1, [μ₀_gt], label = "True value", linestyle = :dash, linewidth = 2, color = :red)

# Add summary text
plot!(p1, legend = :topright)
annotate!(
    p1, [-2.5, maximum(posterior_weights) * 0.8],
    text("MCMC: μ=$(round(mcmc_mean, digits = 3)), σ=$(round(mcmc_std, digits = 3))", 10)
)
annotate!(
    p1, [-2.5, maximum(posterior_weights) * 0.7],
    text("INLA: μ=$(round(inla_mean, digits = 3)), σ=$(round(inla_std, digits = 3))", 10)
)
annotate!(
    p1, [-2.5, maximum(posterior_weights) * 0.6],
    text("True: μ=$(round(μ₀_gt, digits = 3))", 10)
)

display(p1)

# Print comparison summary
println("\nComparison Summary:")
println("Method    | Mean     | Std      | 95% CI")
println("----------|----------|----------|------------------")
println("MCMC      | $(lpad(round(mcmc_mean, digits = 3), 8)) | $(lpad(round(mcmc_std, digits = 3), 8)) | [$(round(mcmc_ci[1], digits = 3)), $(round(mcmc_ci[2], digits = 3))]")
println("INLA      | $(lpad(round(inla_mean, digits = 3), 8)) | $(lpad(round(inla_std, digits = 3), 8)) | [$(round(inla_q025, digits = 3)), $(round(inla_q975, digits = 3))]")
println("True      | $(lpad(round(μ₀_gt, digits = 3), 8)) | --       | --")

println("\n✓ MCMC vs INLA comparison completed!")
