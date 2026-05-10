# Plotting Latent and Observation Marginals with Makie
# =======================================================
#
# This example demonstrates plotting WeightedMixture (latent marginals) and
# TransformedWeightedMixture (observation marginals) using the Makie recipe.

using Latte
using GaussianMarkovRandomFields
using Distributions
using Random
using CairoMakie
using StatsModels
using DataFrames

# Set seed for reproducibility
Random.seed!(123)

# ==================== Prepare Data ====================
# Simple Poisson regression with a random walk prior
n = 100
x = 1:n
λ_true = exp.(0.5 .+ 0.01 .* (x .- 50) .+ 0.2 .* sin.(2π .* x ./ 30))
y = rand.(Poisson.(λ_true))

data = DataFrame(y = y, idx = 1:n)

# ==================== Fit INLA Model ====================
# Define random walk prior
rw = RandomWalk(1)

# Create formula
f = @formula(y ~ 1 + rw(idx))

# Hyperparameter specification with PC prior
hp_spec = @hyperparams begin
    (τ_rw1 ~ PCPrior.Precision(1.0, α = 0.01), transform = log)
end

# Run INLA
println("Fitting INLA model...")
result = inla(f, hp_spec, data; family = Poisson, progress = false)
println("✓ Model fitted!\n")

# Compute observation marginals (fitted values)
println("Computing observation marginals...")
obs_marginals = observation_marginals(result)
println("✓ Observation marginals computed!\n")

# ==================== Example 1: Plot a Latent Marginal ====================
# Latent marginals are WeightedMixture distributions
# For this model, latent marginals are the linear predictors η

println("Plotting latent marginal (linear predictor)...")

# Get the latent marginal for the 50th observation
latent_marginal_50 = result.latent_marginals[50]

fig1 = Figure(size = (600, 400))
ax1 = Axis(
    fig1[1, 1],
    xlabel = "Linear predictor η",
    ylabel = "Posterior density",
    title = "Latent Marginal (observation 50)"
)

# Plot with credible interval
plot!(
    ax1, latent_marginal_50;
    credible_interval = 0.95,
    ci_alpha = 0.3,
    color = :purple,
    linewidth = 2
)

save("latent_marginal_example.png", fig1)
println("✓ Saved to latent_marginal_example.png")

# ==================== Example 2: Plot an Observation Marginal ====================
# Observation marginals are TransformedWeightedMixture distributions

println("\nPlotting observation marginal (fitted value)...")

# Get the observation marginal for the 50th observation
obs_marginal_50 = obs_marginals[50]

fig2 = Figure(size = (600, 400))
ax2 = Axis(
    fig2[1, 1],
    xlabel = "Fitted value λ",
    ylabel = "Posterior density",
    title = "Observation Marginal (observation 50)"
)

# Plot with credible interval
plot!(
    ax2, obs_marginal_50;
    credible_interval = 0.95,
    ci_alpha = 0.3,
    color = :steelblue,
    linewidth = 2
)

# Add a vertical line at the observed value for reference
vlines!(ax2, [y[50]], color = :red, linestyle = :dash, label = "Observed")

axislegend(ax2; position = :rt)

save("observation_marginal_example.png", fig2)
println("✓ Saved to observation_marginal_example.png")

# ==================== Example 3: Compare Multiple Latent Marginals ====================

println("\nPlotting multiple latent marginals...")

fig3 = Figure(size = (800, 400))
ax3 = Axis(
    fig3[1, 1],
    xlabel = "Linear predictor η",
    ylabel = "Posterior density",
    title = "Comparing Latent Marginals"
)

# Plot latent marginals for observations 25, 50, 75
indices = [25, 50, 75]
colors = [:coral, :purple, :teal]

for (i, idx) in enumerate(indices)
    plot!(
        ax3, result.latent_marginals[idx];
        label = "Observation $idx",
        color = colors[i],
        credible_interval = nothing  # Disable for cleaner comparison
    )
end

axislegend(ax3; position = :rt)

save("multiple_latent_marginals.png", fig3)
println("✓ Saved to multiple_latent_marginals.png")

# ==================== Example 4: Compare Multiple Observation Marginals ====================

println("\nPlotting multiple observation marginals...")

fig4 = Figure(size = (800, 400))
ax4 = Axis(
    fig4[1, 1],
    xlabel = "Fitted value λ",
    ylabel = "Posterior density",
    title = "Comparing Observation Marginals"
)

# Plot observation marginals for observations 25, 50, 75
for (i, idx) in enumerate(indices)
    plot!(
        ax4, obs_marginals[idx];
        label = "Observation $idx",
        color = colors[i],
        credible_interval = nothing
    )
end

axislegend(ax4; position = :rt)

save("multiple_observation_marginals.png", fig4)
println("✓ Saved to multiple_observation_marginals.png")

# ==================== Example 5: Side-by-side comparison ====================

println("\nCreating side-by-side comparison...")

fig5 = Figure(size = (1200, 400))

# Left: Latent marginal
ax5a = Axis(
    fig5[1, 1],
    xlabel = "Linear predictor η",
    ylabel = "Posterior density",
    title = "Latent Marginal (obs 50)"
)

plot!(
    ax5a, latent_marginal_50;
    credible_interval = 0.95,
    color = :purple,
    linewidth = 2
)

# Right: Observation marginal
ax5b = Axis(
    fig5[1, 2],
    xlabel = "Fitted value λ",
    ylabel = "Posterior density",
    title = "Observation Marginal (obs 50)"
)

plot!(
    ax5b, obs_marginal_50;
    credible_interval = 0.95,
    color = :steelblue,
    linewidth = 2
)

# Add observed value
vlines!(ax5b, [y[50]], color = :red, linestyle = :dash)

save("side_by_side_comparison.png", fig5)
println("✓ Saved to side_by_side_comparison.png")

println("\n✓ All examples completed successfully!")
println("\nGenerated files:")
println("  1. latent_marginal_example.png")
println("  2. observation_marginal_example.png")
println("  3. multiple_latent_marginals.png")
println("  4. multiple_observation_marginals.png")
println("  5. side_by_side_comparison.png")
