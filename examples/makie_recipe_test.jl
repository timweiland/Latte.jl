# Test script for Makie recipe with HyperparameterMarginalDistribution
# This demonstrates the usage of the plotting recipes

using Latte
using GaussianMarkovRandomFields
using Distributions
using Random
using CairoMakie  # This loads the extension

# Set seed for reproducibility
Random.seed!(123)

# ==================== Create test data ====================
# Simple Poisson regression with random walk prior
n = 50
x = 1:n
λ_true = exp.(0.5 .+ 0.02 .* (x .- 25) .+ 0.3 .* sin.(2π .* x ./ 20))
y = rand.(Poisson.(λ_true))

# ==================== Fit model with INLA ====================
# Define latent model: random walk
rw = RW1()

# Create formula
using StatsModels
using DataFrames

data = DataFrame(y = y, idx = 1:n)
f = @formula(y ~ 1 + rw(idx))

# Hyperparameter specification
hp_spec = @hyperparams begin
    (τ_rw ~ PCPrior.Precision(1.0, α = 0.01), transform = log)
end

# Run INLA
println("Fitting INLA model...")
result = inla(f, hp_spec, data; family = Poisson, progress = false)
println("✓ Model fitted successfully\n")

# ==================== Test Makie Recipes ====================

# Get the hyperparameter marginal distribution
τ_dist = result.hyperparameter_marginals.τ_rw

println("Testing Makie recipes:")
println("  Distribution type: ", typeof(τ_dist))
println("  Mode: ", round(mode(τ_dist), digits = 4))
println("  Mean: ", round(mean(τ_dist), digits = 4))
println(
    "  95% CI: [", round(quantile(τ_dist, 0.025), digits = 4), ", ",
    round(quantile(τ_dist, 0.975), digits = 4), "]\n"
)

# ==================== Test 1: Simple lines plot ====================
println("Test 1: Basic lines() plot")
try
    fig1 = Figure(size = (600, 400))
    ax1 = Axis(
        fig1[1, 1],
        xlabel = "Precision (τ_rw)",
        ylabel = "Posterior density",
        title = "Test 1: lines(distribution)"
    )

    lines!(ax1, τ_dist; color = :steelblue, linewidth = 2)

    save("test_recipe_lines.png", fig1)
    println("  ✓ lines() works! Saved to test_recipe_lines.png\n")
catch e
    println("  ✗ Error: ", e, "\n")
end

# ==================== Test 2: Generic plot ====================
println("Test 2: Generic plot() - should use distplot automatically")
try
    fig2 = Figure(size = (600, 400))
    ax2 = Axis(
        fig2[1, 1],
        xlabel = "Precision (τ_rw)",
        ylabel = "Posterior density",
        title = "Test 2: plot(distribution) with 95% CI"
    )

    plot!(ax2, τ_dist; color = :coral, linewidth = 2)

    save("test_recipe_plot.png", fig2)
    println("  ✓ plot() works! Saved to test_recipe_plot.png\n")
catch e
    println("  ✗ Error: ", e, "\n")
end

# ==================== Test 3: distplot with credible interval ====================
println("Test 3: distplot() with credible interval shading")
try
    fig3 = Figure(size = (600, 400))
    ax3 = Axis(
        fig3[1, 1],
        xlabel = "Precision (τ_rw)",
        ylabel = "Posterior density",
        title = "Test 3: distplot with 95% CI shading"
    )

    distplot!(
        ax3, τ_dist;
        credible_interval = 0.95,
        ci_alpha = 0.3,
        color = :darkblue,
        linewidth = 2
    )

    save("test_recipe_distplot.png", fig3)
    println("  ✓ distplot() works! Saved to test_recipe_distplot.png\n")
catch e
    println("  ✗ Error: ", e, "\n")
end

# ==================== Test 4: distplot with mode indicator ====================
println("Test 4: distplot() with mode indicator")
try
    fig4 = Figure(size = (600, 400))
    ax4 = Axis(
        fig4[1, 1],
        xlabel = "Precision (τ_rw)",
        ylabel = "Posterior density",
        title = "Test 4: distplot with mode line"
    )

    distplot!(
        ax4, τ_dist;
        credible_interval = 0.95,
        show_mode = true,
        color = :steelblue,
        linewidth = 2
    )

    save("test_recipe_mode.png", fig4)
    println("  ✓ distplot() with mode works! Saved to test_recipe_mode.png\n")
catch e
    println("  ✗ Error: ", e, "\n")
end

# ==================== Test 5: Custom quantile range ====================
println("Test 5: Custom quantile range")
try
    fig5 = Figure(size = (600, 400))
    ax5 = Axis(
        fig5[1, 1],
        xlabel = "Precision (τ_rw)",
        ylabel = "Posterior density",
        title = "Test 5: Wide quantile range (0.0001 to 0.9999)"
    )

    distplot!(
        ax5, τ_dist;
        quantile_range = (0.0001, 0.9999),
        credible_interval = 0.95,
        color = :purple,
        linewidth = 2
    )

    save("test_recipe_quantile.png", fig5)
    println("  ✓ Custom quantile range works! Saved to test_recipe_quantile.png\n")
catch e
    println("  ✗ Error: ", e, "\n")
end

println("✓ All recipe tests completed!")
println("\nGenerated files:")
println("  - test_recipe_lines.png")
println("  - test_recipe_plot.png")
println("  - test_recipe_distplot.png")
println("  - test_recipe_mode.png")
println("  - test_recipe_quantile.png")
