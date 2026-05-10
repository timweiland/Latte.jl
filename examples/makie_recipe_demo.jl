# Plotting HyperparameterMarginalDistribution with Makie
# =========================================================
#
# This example demonstrates how to use the Makie recipe for visualizing
# hyperparameter marginal distributions from INLA results.

using Latte
using GaussianMarkovRandomFields
using Distributions
using Random
using CairoMakie  # This automatically loads the plotting extension
using StatsModels
using DataFrames

# Set seed for reproducibility
Random.seed!(42)

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

# ==================== Example 1: Simple plot() ====================
# The generic plot() function automatically uses the recipe

τ_dist = result.hyperparameter_marginals.τ_rw1

fig1 = Figure(size = (600, 400))
ax1 = Axis(
    fig1[1, 1],
    xlabel = "Precision (τ)",
    ylabel = "Posterior density",
    title = "Example 1: Simple plot()"
)

# Just call plot! with the distribution - it works automatically!
plot!(ax1, τ_dist; color = :steelblue, linewidth = 2)

save("example1_simple_plot.png", fig1)
println("✓ Example 1 saved to example1_simple_plot.png")

# ==================== Example 2: Using lines() ====================
# The type recipe also works with lines(), scatter(), etc.

fig2 = Figure(size = (600, 400))
ax2 = Axis(
    fig2[1, 1],
    xlabel = "Precision (τ)",
    ylabel = "Posterior density",
    title = "Example 2: Using lines()"
)

lines!(ax2, τ_dist; color = :coral, linewidth = 2.5)

save("example2_lines.png", fig2)
println("✓ Example 2 saved to example2_lines.png")

# ==================== Example 3: distplot with credible interval ====================
# Use the enhanced distplot() for credible interval shading

fig3 = Figure(size = (600, 400))
ax3 = Axis(
    fig3[1, 1],
    xlabel = "Precision (τ)",
    ylabel = "Posterior density",
    title = "Example 3: With 95% Credible Interval"
)

plot!(
    ax3, τ_dist;
    credible_interval = 0.95,  # Show 95% CI
    ci_alpha = 0.3,             # Transparency for shading
    color = :darkblue,
    linewidth = 2
)

save("example3_credible_interval.png", fig3)
println("✓ Example 3 saved to example3_credible_interval.png")

# ==================== Example 4: Multiple distributions ====================
# Compare multiple hyperparameter marginals

# Fit a model with multiple hyperparameters (BYM2 model example)
# For this we'd need spatial data, so let's simulate a simpler example
# with two independent random effects

iid = IID()
data2 = DataFrame(y = y, idx1 = 1:n, idx2 = 1:n)

f2 = @formula(y ~ 1 + rw(idx1) + iid(idx2))

hp_spec2 = @hyperparams begin
    (τ_rw1 ~ PCPrior.Precision(1.0, α = 0.01), transform = log)
    (τ_iid ~ PCPrior.Precision(1.0, α = 0.01), transform = log)
end

println("\nFitting second model with two hyperparameters...")
result2 = inla(f2, hp_spec2, data2; family = Poisson, progress = false)
println("✓ Second model fitted!")

fig4 = Figure(size = (800, 400))
ax4 = Axis(
    fig4[1, 1],
    xlabel = "Precision",
    ylabel = "Posterior density",
    title = "Example 4: Comparing Multiple Hyperparameters"
)

# Plot multiple distributions
plot!(
    ax4, result2.hyperparameter_marginals.τ_rw1;
    label = "τ_rw1 (Random Walk)",
    color = :steelblue,
    credible_interval = nothing  # Disable CI for cleaner comparison
)

plot!(
    ax4, result2.hyperparameter_marginals.τ_iid;
    label = "τ_iid (IID)",
    color = :coral,
    credible_interval = nothing
)

axislegend(ax4; position = :rt)

save("example4_multiple_distributions.png", fig4)
println("✓ Example 4 saved to example4_multiple_distributions.png")

# ==================== Example 5: Custom quantile range ====================
# Use a wider quantile range to show more of the tails

fig5 = Figure(size = (600, 400))
ax5 = Axis(
    fig5[1, 1],
    xlabel = "Precision (τ)",
    ylabel = "Posterior density",
    title = "Example 5: Wide Quantile Range (0.0001 to 0.9999)"
)

plot!(
    ax5, τ_dist;
    quantile_range = (0.0001, 0.9999),  # Show wider range
    credible_interval = 0.95,
    color = :purple,
    linewidth = 2
)

save("example5_wide_range.png", fig5)
println("✓ Example 5 saved to example5_wide_range.png")

# ==================== Example 6: With mode and median indicators ====================
# Show both the mode and median with vertical lines

fig6 = Figure(size = (600, 400))
ax6 = Axis(
    fig6[1, 1],
    xlabel = "Precision (τ)",
    ylabel = "Posterior density",
    title = "Example 6"
)

plot!(
    ax6, τ_dist;
    credible_interval = 0.95,
    color = :steelblue,
    linewidth = 2,
    label = "Posterior"        # Label for main curve
)

# Add legend
axislegend(ax6; position = :rt)

save("example6_mode_median_indicator.png", fig6)
println("✓ Example 6 saved to example6_mode_median_indicator.png")

println("\n✓ All examples completed successfully!")
println("\nGenerated files:")
println("  1. example1_simple_plot.png")
println("  2. example2_lines.png")
println("  3. example3_credible_interval.png")
println("  4. example4_multiple_distributions.png")
println("  5. example5_wide_range.png")
println("  6. example6_mode_median_indicator.png")
