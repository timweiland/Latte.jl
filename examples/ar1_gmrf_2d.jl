# AR-1 GMRF Example with 2D Hyperparameters
#
# This example demonstrates INLA inference for a Bernoulli observation model
# with an AR-1 GMRF based on the GaussianMarkovRandomFields.jl example.
# We use two hyperparameters:
# - σ: marginal standard deviation
# - ρ: autocorrelation coefficient
#
# The model is:
# - Latent field: x ~ AR-1 GMRF with correlation ρ and variance σ²
# - Observations: y_i ~ Bernoulli(logistic(x_i))
# - Hyperparameters: θ = [σ, ρ]

using Latte
using GaussianMarkovRandomFields
using LDLFactorizations
using Distributions
using LinearAlgebra
using StatsFuns
using Plots
using SparseArrays

# AR-1 precision matrix function (from the example)
function ar_precision(ρ, k)
    return spdiagm(-1 => -ρ * ones(k - 1), 0 => ones(k), 1 => -ρ * ones(k - 1))
end

# Model parameters
k = 5000   # number of time points (smaller for speed)

# True hyperparameter values for simulation
σ_true = 2.5   # marginal standard deviation
ρ_true = 0.4   # autocorrelation coefficient

# Priors for hyperparameters (matching the Turing example)
σ_prior = Gamma(2, 3)
ρ_prior = Uniform(0, 0.5)

# Joint hyperparameter prior
θ_prior = product_distribution([σ_prior, ρ_prior])

println("True hyperparameters:")
println("  σ = $(σ_true)")
println("  ρ = $(ρ_true)")
println("  Time series length: $(k)")

# Function to create latent GMRF given hyperparameters θ = [σ, ρ]
function latent_gmrf(θ)
    σ, ρ = θ

    # Create AR-1 precision matrix (from the example)
    Q = ar_precision(ρ, k) ./ σ^2

    # Zero mean (simpler than the example which used μ*ones(k))
    μ = zeros(k)

    return (μ, Q)
end
# Observation model (Bernoulli with logit link)
obs_model = ExponentialFamily(Bernoulli)

# Create INLA model
inla_model = LatentGaussianModel(θ_prior, latent_gmrf, obs_model)

println("\nGenerating synthetic data...")

# Generate synthetic data
x_gt = rand(GMRF(latent_gmrf([σ_true, ρ_true])...))
y_gt = rand(likelihood(obs_model, x_gt, Float64[]))

println("Generated $(length(y_gt)) observations")
println("Proportion of 1s: $(round(mean(y_gt), digits = 3))")

# Plot the time series
println("\nVisualizing AR-1 time series...")
p_ts = plot(
    x_gt, title = "True AR-1 Latent Field", xlabel = "Time", ylabel = "x(t)",
    linewidth = 2, label = "Latent field"
)
scatter!(
    p_ts, findall(y_gt .== 1), x_gt[y_gt .== 1],
    color = :red, label = "y=1", markersize = 3
)
scatter!(
    p_ts, findall(y_gt .== 0), x_gt[y_gt .== 0],
    color = :blue, label = "y=0", markersize = 3
)
display(p_ts)

println("\n" * "="^60)
println("Testing INLA Hyperparameter Exploration (2D)")
println("="^60)

# Test the hyperparameter exploration step by step
println("\nStep 1: Finding hyperparameter mode...")

θ_star, mode_points, mode_logdensities = find_hyperparameter_mode(inla_model, y_gt)

println("Found mode: θ* = [$(round(θ_star[1], digits = 3)), $(round(θ_star[2], digits = 3))]")
println("True values: θ = [$(σ_true), $(ρ_true)]")
println("Collected $(length(mode_points)) points during optimization")

println("\nStep 2: Exploring hyperparameter posterior...")

#exploration = explore_hyperparameter_posterior(inla_model, y_gt, θ_star, mode_points, mode_logdensities;
#δ_π=2.5, interpolation_factor=3)

exploration = explore_hyperparameter_posterior(
    inla_model, y_gt, θ_star, nothing, nothing;
    δ_π = 2.5, interpolation_factor = 3
)

println("Exploration results:")
println("  Total interpolation points: $(length(exploration.interpolation_points))")
println("  Integration points: $(length(exploration.integration_indices))")
println("  Mode: [$(round(exploration.mode[1], digits = 3)), $(round(exploration.mode[2], digits = 3))]")

println("\nTransformation info: Λ_inv_sqrt diagonal: $(round.(diag(exploration.transformation.Λ_inv_sqrt), digits = 3))")

println("\nStep 3: Testing interpolant construction...")

posterior_approx = build_posterior_interpolant(exploration)

println("\n" * "="^60)
println("Analyzing 2D Results")
println("="^60)

# Extract θ values for analysis
σ_vals = [θ[1] for θ in exploration.interpolation_points]
ρ_vals = [θ[2] for θ in exploration.interpolation_points]
log_densities = exploration.log_densities

println("Parameter ranges:")
println("  σ: [$(round(minimum(σ_vals), digits = 3)), $(round(maximum(σ_vals), digits = 3))]")
println("  ρ: [$(round(minimum(ρ_vals), digits = 3)), $(round(maximum(ρ_vals), digits = 3))]")

# Get integration points
integration_σ = [σ_vals[i] for i in exploration.integration_indices]
integration_ρ = [ρ_vals[i] for i in exploration.integration_indices]

println("Integration points: $(length(integration_σ))")

# Create scatter plot
println("\nCreating 2D posterior exploration plot...")

p1 = scatter(
    σ_vals, ρ_vals, zcolor = log_densities,
    xlabel = "σ (marginal std dev)", ylabel = "ρ (autocorrelation)",
    title = "2D AR-1 GMRF Hyperparameter Exploration",
    marker = :circle, markersize = 3, alpha = 0.7,
    colorbar_title = "Log Density"
)

# Mark the mode
scatter!(
    p1, [θ_star[1]], [θ_star[2]],
    marker = :star, markersize = 8, color = :red,
    label = "Mode"
)

# Mark true values
#scatter!(p1, [σ_true], [ρ_true],
#marker=:diamond, markersize=8, color=:green,
#label="True Values")

# Mark integration points
scatter!(
    p1, integration_σ, integration_ρ,
    marker = :square, markersize = 4, color = :black, alpha = 0.8,
    label = "Integration Points"
)

display(p1)

# Create contour plot for better visualization
println("\nCreating contour plot of posterior around mode...")

# Define grid around the mode for contour plotting
σ_range = range(max(0.1, θ_star[1] - 0.1), θ_star[1] + 0.1, length = 30)
ρ_range = range(max(0.05, θ_star[2] - 0.01), min(0.49, θ_star[2] + 0.1), length = 30)

# Evaluate posterior on grid
posterior_grid = zeros(length(σ_range), length(ρ_range))
println("Evaluating posterior on $(length(σ_range))×$(length(ρ_range)) grid...")

for (i, σ) in enumerate(σ_range)
    for (j, ρ) in enumerate(ρ_range)
        try
            posterior_grid[i, j] = hyperparameter_logpdf(inla_model, [σ, ρ], y_gt)
        catch
            posterior_grid[i, j] = -Inf  # Handle numerical issues
        end
    end
    if i % 5 == 0
        println("  Progress: $(round(100 * i / length(σ_range), digits = 1))%")
    end
end

# Convert to regular density (subtract max for numerical stability)
max_logpdf = maximum(posterior_grid[isfinite.(posterior_grid)])
posterior_grid_exp = exp.(posterior_grid .- max_logpdf)

# Create contour plot
p2 = contour(
    σ_range, ρ_range, posterior_grid_exp',
    xlabel = "σ (marginal std dev)", ylabel = "ρ (autocorrelation)",
    title = "Posterior Density Contours",
    levels = 10, linewidth = 2
)

# Add exploration points
scatter!(
    p2, σ_vals, ρ_vals,
    markersize = 2, alpha = 0.6, color = :gray, label = "Exploration points"
)

# Mark the mode
scatter!(
    p2, [θ_star[1]], [θ_star[2]],
    marker = :star, markersize = 10, color = :red,
    label = "Mode"
)

# Mark true values
#scatter!(p2, [σ_true], [ρ_true],
#marker=:diamond, markersize=10, color=:green,
#label="True Values")

# Mark integration points
scatter!(
    p2, integration_σ, integration_ρ,
    marker = :square, markersize = 6, color = :black,
    label = "Integration Points"
)

display(p2)

# Create combined plot
p_combined = plot(
    p1, p2, layout = (1, 2), size = (1200, 500),
    plot_title = "2D AR-1 GMRF Hyperparameter Exploration"
)
display(p_combined)

println("\n✓ 2D AR-1 GMRF exploration test completed!")
println("\nSummary:")
println("- Mode finding: ✓")
println("- 2D grid exploration: ✓")
println("- Integration point selection: ✓")
println("- Multidimensional grid building: ✓")
println("- AR-1 GMRF with $(k) time points: ✓")
