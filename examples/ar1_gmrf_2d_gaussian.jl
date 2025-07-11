# AR-1 GMRF Example with 2D Hyperparameters
#
# This example demonstrates INLA inference for a Gaussian observation model
# with an AR-1 GMRF based on the GaussianMarkovRandomFields.jl example.
# We use two hyperparameters:
# - σ: marginal standard deviation
# - ρ: autocorrelation coefficient
#
# The model is:
# - Latent field: x ~ AR-1 GMRF with correlation ρ and variance σ²
# - Observations: y_i ~ Normal(x_i, σ_y²)
# - Hyperparameters: θ = [σ, ρ]

using IntegratedNestedLaplace
using GaussianMarkovRandomFields
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
k = 1000   # number of time points

# True hyperparameter values for simulation
σ_gmrf_true = 2.5   # marginal standard deviation
ρ_true = 0.4   # autocorrelation coefficient

# Priors for hyperparameters (matching the Turing example)
#σ_prior = Gamma(2, 3)
#ρ_prior = Uniform(0, 0.5)

θ_prior = HyperparameterPrior((σ_gmrf = Gamma(2, 3), ρ = Uniform(0, 0.5)), fixed = (σ = 1.0e-6,))
# Joint hyperparameter prior
#θ_prior = product_distribution([σ_prior, ρ_prior])

println("True hyperparameters:")
println("  σ = $(σ_gmrf_true)")
println("  ρ = $(ρ_true)")
println("  Time series length: $(k)")

# Function to create latent GMRF given hyperparameters θ = [σ, ρ]
function latent_gmrf(θ)
    σ = θ.σ_gmrf
    ρ = θ.ρ

    # Create AR-1 precision matrix (from the example)
    Q = ar_precision(ρ, k) ./ σ^2

    # Zero mean (simpler than the example which used μ*ones(k))
    μ = zeros(k)

    return GMRF(μ, Q, CholeskySolverBlueprint())
end

# Observation model (Normal with identity link)
obs_model = ExponentialFamily(Normal)

# Create INLA model
inla_model = INLAModel(θ_prior, latent_gmrf, obs_model)

println("\nGenerating synthetic data...")

# Generate synthetic data
x_gt = rand(latent_gmrf((σ_gmrf = σ_gmrf_true, ρ = ρ_true)))
y_gt = rand(likelihood(obs_model, x_gt, (σ = 1.0e-6,)))

println("Generated $(length(y_gt)) observations")

# Plot the time series
println("\nVisualizing AR-1 time series...")
p_ts = plot(
    y_gt, title = "Observations", xlabel = "Time", ylabel = "y(t)",
    linewidth = 2, label = "Observations"
)
display(p_ts)

println("\n" * "="^60)
println("Testing INLA Hyperparameter Exploration (2D)")
println("="^60)

# Test the hyperparameter exploration step by step
println("\nStep 1: Finding hyperparameter mode...")

θ_star, mode_points, mode_logdensities = find_hyperparameter_mode(inla_model, y_gt)

println("Found mode: θ* = [$(round(θ_star[1], digits = 3)), $(round(θ_star[2], digits = 3))]")
println("True values: θ = [$(σ_gmrf_true), $(ρ_true)]")
println("Collected $(length(mode_points)) points during optimization")

println("\nStep 2: Exploring hyperparameter posterior...")

#exploration = explore_hyperparameter_posterior(inla_model, y_gt, θ_star, mode_points, mode_logdensities;
#δ_π=2.5, interpolation_factor=3)

exploration = explore_hyperparameter_posterior(
    inla_model, y_gt, θ_star, GaussianMarginal(), 1:k
)

println("Exploration results:")
println("  Total interpolation points: $(length(exploration.interpolation_points))")
println("  Integration points: $(length(exploration.integration_indices))")
println("  Mode: [$(round(exploration.mode[1], digits = 3)), $(round(exploration.mode[2], digits = 3))]")

println("\nTransformation info: Λ_inv_sqrt diagonal: $(round.(diag(exploration.transformation.Λ_inv_sqrt), digits = 3))")

println("\nStep 3: Testing interpolant construction...")

posterior_approx = build_posterior_interpolant(exploration, rbf_method = :thin_plate)

println("\n" * "="^60)
println("Analyzing 2D Results")
println("="^60)

named_interpolation_points = to_named.(exploration.interpolation_points, Ref(inla_model.hyperparameter_prior))

# Extract θ values for analysis
σ_vals = [θ.σ_gmrf for θ in named_interpolation_points]
ρ_vals = [θ.ρ for θ in named_interpolation_points]
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

# Test interpolant on fine grid to check for oscillatory behavior
println("\n" * "="^60)
println("Testing Interpolant on Fine Grid")
println("="^60)

println("\nEvaluating interpolant vs true posterior on fine grid...")

# Create a finer grid for interpolant testing
σ_fine = range(exploration.integration_bounds[1, 1], exploration.integration_bounds[1, 2], length = 50)
ρ_fine = range(exploration.integration_bounds[2, 1], exploration.integration_bounds[2, 2], length = 50)

# Evaluate both interpolant and true posterior on fine grid
interpolant_grid = zeros(length(σ_fine), length(ρ_fine))
true_posterior_grid = zeros(length(σ_fine), length(ρ_fine))

println("Evaluating on $(length(σ_fine))×$(length(ρ_fine)) grid...")

for (i, σ) in enumerate(σ_fine)
    for (j, ρ) in enumerate(ρ_fine)
        try
            # Evaluate interpolant
            interpolant_grid[i, j] = posterior_approx([σ, ρ])

            # Evaluate true posterior
            true_posterior_grid[i, j] = hyperparameter_logpdf(inla_model, [σ, ρ], y_gt)
        catch e
            interpolant_grid[i, j] = -Inf
            true_posterior_grid[i, j] = -Inf
        end
    end
    if i % 10 == 0
        println("  Progress: $(round(100 * i / length(σ_fine), digits = 1))%")
    end
end

# Create comparison plots
println("\nCreating interpolant comparison plots...")

# Plot 1: Interpolant surface
p_interp = heatmap(
    σ_fine, ρ_fine, interpolant_grid',
    xlabel = "σ (marginal std dev)", ylabel = "ρ (autocorrelation)",
    title = "Interpolant Log Density", color = :viridis
)

# Add interpolation points
scatter!(
    p_interp, σ_vals, ρ_vals,
    markersize = 3, color = :white, alpha = 0.8, label = "Interpolation Points"
)

# Mark the mode
scatter!(
    p_interp, [θ_star[1]], [θ_star[2]],
    marker = :star, markersize = 8, color = :red, label = "Mode"
)

# Plot 2: True posterior surface
p_true = heatmap(
    σ_fine, ρ_fine, true_posterior_grid',
    xlabel = "σ (marginal std dev)", ylabel = "ρ (autocorrelation)",
    title = "True Posterior Log Density", color = :viridis
)

# Add interpolation points
scatter!(
    p_true, σ_vals, ρ_vals,
    markersize = 3, color = :white, alpha = 0.8, label = "Interpolation Points"
)

# Mark the mode
scatter!(
    p_true, [θ_star[1]], [θ_star[2]],
    marker = :star, markersize = 8, color = :red, label = "Mode"
)

# Plot 3: Difference (interpolant - true)
difference_grid = interpolant_grid - true_posterior_grid
difference_grid[.!isfinite.(difference_grid)] .= 0  # Handle infinities

p_diff = heatmap(
    σ_fine, ρ_fine, difference_grid',
    xlabel = "σ (marginal std dev)", ylabel = "ρ (autocorrelation)",
    title = "Interpolation Error (Interp - True)"
)

# Add interpolation points
scatter!(
    p_diff, σ_vals, ρ_vals,
    markersize = 3, color = :black, alpha = 0.8, label = "Interpolation Points"
)

# Mark the mode
scatter!(
    p_diff, [θ_star[1]], [θ_star[2]],
    marker = :star, markersize = 8, color = :red, label = "Mode"
)

# Create combined comparison plot
p_comparison = plot(
    p_interp, p_true, p_diff, layout = (1, 3), size = (1800, 500),
    plot_title = "Interpolant Quality Assessment"
)
display(p_comparison)

# Compute error statistics
finite_mask = isfinite.(difference_grid)
if any(finite_mask)
    max_abs_error = maximum(abs.(difference_grid[finite_mask]))
    mean_abs_error = mean(abs.(difference_grid[finite_mask]))
    rms_error = sqrt(mean(difference_grid[finite_mask] .^ 2))

    println("\nInterpolation Error Statistics:")
    println("  Max absolute error: $(round(max_abs_error, digits = 4))")
    println("  Mean absolute error: $(round(mean_abs_error, digits = 4))")
    println("  RMS error: $(round(rms_error, digits = 4))")

    # Check for oscillatory behavior
    # Compute spatial gradients to detect rapid oscillations
    grad_σ = diff(difference_grid, dims = 1)
    grad_ρ = diff(difference_grid, dims = 2)

    # High frequency content indicator
    high_freq_indicator = std(grad_σ[isfinite.(grad_σ)]) + std(grad_ρ[isfinite.(grad_ρ)])
    println("  High frequency indicator: $(round(high_freq_indicator, digits = 4))")

    if high_freq_indicator > 0.1
        println("  ⚠️  Possible oscillatory behavior detected!")
    else
        println("  ✓ Interpolation appears smooth")
    end
else
    println("\n⚠️  Could not compute error statistics (all values non-finite)")
end

println("\n✓ 2D AR-1 GMRF exploration test completed!")
println("\nSummary:")
println("- Mode finding: ✓")
println("- 2D grid exploration: ✓")
println("- Integration point selection: ✓")
println("- Multidimensional grid building: ✓")
println("- Interpolant quality assessment: ✓")
println("- AR-1 GMRF with $(k) time points: ✓")
