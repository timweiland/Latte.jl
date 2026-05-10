# Joy Plot Demo for INLA Marginals
# ===================================
#
# This example demonstrates creating joy plots (ridgeline plots) for INLA marginals,
# inspired by the famous Joy Division "Unknown Pleasures" album cover.

using Latte
using GaussianMarkovRandomFields
using Distributions
using Random
using CairoMakie
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
rw = RandomWalk(1)
f = @formula(y ~ 1 + rw(idx))

hp_spec = @hyperparams begin
    (τ_rw1 ~ PCPrior.Precision(1.0, α = 0.01), transform = log)
end

println("Fitting INLA model...")
result = inla(f, hp_spec, data; family = Poisson, progress = false)
println("✓ Model fitted!\n")

# ==================== Example 1: Basic Joy Plot ====================
println("Creating basic joy plot...")

# Select latent marginals at regular intervals
indices = 10:10:90
dists = [result.latent_marginals[i] for i in indices]
labels = ["Location $i" for i in indices]

fig1 = joyplot(
    dists;
    labels = labels,
    title = "Latent Field Marginals (Joy Plot)",
    xlabel = "Linear predictor η",
    spacing = 1.0
)

save("joyplot_basic.png", fig1)
println("✓ Saved to joyplot_basic.png")

# ==================== Example 2: Custom Colors ====================
println("\nCreating joy plot with custom colors...")

# Cycle through Wong colors for the distributions
wong_colors = Makie.wong_colors()
n_dists = length(dists)
custom_colors = [wong_colors[mod1(i, length(wong_colors))] for i in 1:n_dists]

fig2 = joyplot(
    dists;
    labels = labels,
    title = "Latent Field Marginals - Custom Colors",
    xlabel = "Linear predictor η",
    spacing = 1.2,
    colors = custom_colors,
    strokewidth = 2
)

save("joyplot_custom_colors.png", fig2)
println("✓ Saved to joyplot_custom_colors.png")

# ==================== Example 3: Observation Marginals ====================
println("\nCreating joy plot with observation marginals...")

obs_marginals = observation_marginals(result)
obs_dists = [obs_marginals[i] for i in indices]

fig3 = joyplot(
    obs_dists;
    labels = labels,
    title = "Observation Marginals (Fitted Values)",
    xlabel = "Fitted value λ",
    spacing = 1.0
)

save("joyplot_observations.png", fig3)
println("✓ Saved to joyplot_observations.png")

# ==================== Example 4: More Distributions ====================
println("\nCreating dense joy plot...")

# Use more distributions for a denser look
indices_dense = 5:5:95
dists_dense = [result.latent_marginals[i] for i in indices_dense]
labels_dense = ["Loc $i" for i in indices_dense]

fig4 = joyplot(
    dists_dense;
    labels = labels_dense,
    title = "Dense Joy Plot (Joy Division Style)",
    xlabel = "Linear predictor η",
    spacing = 0.8,
    strokewidth = 1.5
)

save("joyplot_dense.png", fig4)
println("✓ Saved to joyplot_dense.png")

# ==================== Example 5: Monochrome Style ====================
println("\nCreating monochrome joy plot...")

# All black for classic Joy Division look
black_colors = fill(:black, length(dists))

fig5 = joyplot(
    dists;
    labels = labels,
    title = "Monochrome Joy Plot",
    xlabel = "Linear predictor η",
    spacing = 1.0,
    colors = black_colors,
    strokewidth = 2
)

save("joyplot_monochrome.png", fig5)
println("✓ Saved to joyplot_monochrome.png")

println("\n✓ All joy plot examples completed successfully!")
println("\nGenerated files:")
println("  1. joyplot_basic.png")
println("  2. joyplot_custom_colors.png")
println("  3. joyplot_observations.png")
println("  4. joyplot_dense.png")
println("  5. joyplot_monochrome.png")
