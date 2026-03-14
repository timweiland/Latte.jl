# # Spatio-temporal disease surveillance
#
# Disease incidence varies across both space and time. An additive model can capture
# *where* risk is high and *when* it peaks, but it assumes every region follows the
# same temporal trend. In reality, spatial patterns often shift over time — an
# emerging epidemic may spread outward from a focus, or public-health interventions
# may reduce risk in some regions earlier than others.
#
# In this tutorial we use **separable space-time models** to capture these interactions.
# You will learn:
# - How to specify a **Separable** interaction via the formula interface
# - How the **Kronecker product** structure `Q_time ⊗ Q_space` enforces smoothness
#   in both dimensions simultaneously
# - How to compare an additive model (no interaction) with a space-time interaction model
# - How to visualise space-time effects using **heatmaps** and **spatial snapshots**
#
# ## Simulating spatio-temporal data
#
# We work with a simulated disease surveillance scenario: case counts across a
# 5×5 grid of regions over 12 time periods. Using simulated data lets us know
# the truth and check whether the models recover it.
using Random, SparseArrays, Distributions, Statistics, DataFrames
Random.seed!(42)

n_rows, n_cols = 5, 5
n_regions = n_rows * n_cols  # 25
n_time = 12

# First, we build a rook-adjacency matrix for the grid (each cell connects to its
# horizontal and vertical neighbours):
function grid_adjacency(nrow, ncol)
    n = nrow * ncol
    W = spzeros(n, n)
    for i in 1:nrow, j in 1:ncol
        node = (i - 1) * ncol + j
        if j < ncol  # right neighbour
            W[node, node + 1] = 1.0
            W[node + 1, node] = 1.0
        end
        if i < nrow  # bottom neighbour
            W[node, node + ncol] = 1.0
            W[node + ncol, node] = 1.0
        end
    end
    return W
end
W = grid_adjacency(n_rows, n_cols)

# Now we simulate the ground truth. The linear predictor for region $r$ at time $t$ is
#
# ```math
# \eta_{t,r} = \mu + u_r + v_t + \delta_{t,r}
# ```
#
# where $u_r$ is a spatial gradient, $v_t$ is a temporal trend, and $\delta_{t,r}$ is
# a space-time interaction — a wave that sweeps across the grid over time.
intercept = 0.5

## Spatial effect: gradient from northwest (low) to southeast (high)
u_true = [(i + j) / (n_rows + n_cols) for i in 1:n_rows for j in 1:n_cols]
u_true .-= mean(u_true)
u_true .*= 0.6

## Temporal effect: inverted-U trend peaking around the middle
v_true = [-0.04 * (t - n_time / 2)^2 + 0.3 for t in 1:n_time]
v_true .-= mean(v_true)

## Interaction: a sinusoidal wave sweeping across the grid
delta_true = zeros(n_time, n_regions)
for t in 1:n_time
    for i in 1:n_rows, j in 1:n_cols
        r = (i - 1) * n_cols + j
        phase = (i + j) / (n_rows + n_cols)
        delta_true[t, r] = 0.4 * sin(2π * (t / n_time - phase))
    end
end

## Population at risk per region (gives expected counts of ~10–40)
pop = round.(Int, exp.(randn(n_regions) .* 0.3 .+ 3.0))

## Generate Poisson counts
df = DataFrame(time = Int[], region = Int[], y = Int[], expected = Float64[])
for t in 1:n_time, r in 1:n_regions
    η = intercept + u_true[r] + v_true[t] + delta_true[t, r]
    count = rand(Poisson(pop[r] * exp(η)))
    push!(df, (time = t, region = r, y = count, expected = Float64(pop[r])))
end
println("$(nrow(df)) observations, mean count = $(round(mean(df.y), digits = 1))")

# ## Exploratory visualisation
#
# A heatmap of the standardised rate (observed / expected) gives a first look at
# the space-time structure:
using CairoMakie

rate_matrix = reshape(df.y ./ df.expected, n_regions, n_time)'
fig = Figure(size = (800, 400))
ax = Axis(
    fig[1, 1],
    xlabel = "Region", ylabel = "Time period",
    title = "Standardised rate (observed / expected)"
)
hm = heatmap!(ax, 1:n_regions, 1:n_time, rate_matrix', colormap = Reverse(:RdYlBu))
Colorbar(fig[1, 2], hm)
fig

# You can see both a spatial gradient (left vs right) and a temporal arc (middle rows
# brighter). Importantly, the bright patch shifts across regions over time — this is
# the interaction signal that an additive model cannot capture.
#
# ## Model 1: Additive main effects
#
# We start with a model that assumes spatial and temporal effects are independent:
#
# ```math
# \log \lambda_{t,r} = \mu + u_r + v_t
# ```
#
# The spatial component is a Besag (ICAR) model and the temporal component is a
# first-order random walk:
using GaussianMarkovRandomFields, StatsModels
using IntegratedNestedLaplace

spatial = Besag(W, normalize_var = true)
temporal = RandomWalk()
f_additive = @formula(y ~ 1 + spatial(region) + temporal(time))

hp_additive = @hyperparams begin
    (τ_besag ~ PCPrior.Precision(1.0, α = 0.01), transform = log)
    (τ_rw1 ~ PCPrior.Precision(1.0, α = 0.01), transform = log)
end

result_additive = inla(
    f_additive, hp_additive, df;
    family = Poisson, exposure = :expected, progress = false
)

# Let's visualise the fitted rates:
obs_add = observation_marginals(result_additive)
fit_add = summary_df(obs_add)

fitted_add = reshape(fit_add.median, n_regions, n_time)'
fig = Figure(size = (800, 400))
ax = Axis(
    fig[1, 1],
    xlabel = "Region", ylabel = "Time period",
    title = "Additive model — fitted rates"
)
hm = heatmap!(ax, 1:n_regions, 1:n_time, fitted_add', colormap = Reverse(:RdYlBu))
Colorbar(fig[1, 2], hm)
fig

# The additive model captures the overall spatial gradient and temporal arc, but
# the fitted surface is separable by construction — every region gets the same
# temporal pattern, just shifted up or down. It cannot reproduce the sweeping
# wave visible in the raw data.
#
# ## Model 2: Separable space-time interaction
#
# The additive model forces every region to follow the same temporal pattern.
# A **separable** model relaxes this: each region gets its own smooth temporal
# curve. The precision matrix is a Kronecker product
#
# ```math
# Q = Q_{\text{time}} \otimes Q_{\text{space}}
# ```
#
# With `RandomWalk()` for time and `IID()` for space, this gives each region an
# independent first-order random walk — so regions are free to have different
# temporal evolutions. The RW1 structure still enforces smoothness *within*
# each region's trend.
st = Separable(RandomWalk(), IID())
f_interaction = @formula(y ~ 1 + st(time, region))

hp_interaction = @hyperparams begin
    (τ_rw1_separable ~ PCPrior.Precision(1.0, α = 0.01), transform = log)
    (τ_iid_separable ~ PCPrior.Precision(1.0, α = 0.01), transform = log)
end

result_interaction = inla(
    f_interaction, hp_interaction, df;
    family = Poisson, exposure = :expected, progress = false
)

# Fitted rates from the interaction model:
obs_int = observation_marginals(result_interaction)
fit_int = summary_df(obs_int)

fitted_int = reshape(fit_int.median, n_regions, n_time)'
fig = Figure(size = (800, 400))
ax = Axis(
    fig[1, 1],
    xlabel = "Region", ylabel = "Time period",
    title = "Separable model — fitted rates"
)
hm = heatmap!(ax, 1:n_regions, 1:n_time, fitted_int', colormap = Reverse(:RdYlBu))
Colorbar(fig[1, 2], hm)
fig

# The Separable model captures the shifting pattern. Each region now has its own
# smooth temporal trend, so the model can represent the wave-like interaction
# that the additive model misses.
#
# ## Spatial snapshots at selected time points
#
# To make the interaction even more vivid, let's look at spatial maps at four
# time points. We reshape each time slice back into the 5×5 grid:
snapshot_times = [1, 4, 8, 12]

fig = Figure(size = (900, 250))
for (k, t) in enumerate(snapshot_times)
    local ax = Axis(
        fig[1, k],
        title = "t = $t",
        xlabel = k == 1 ? "Column" : "", ylabel = k == 1 ? "Row" : "",
        aspect = DataAspect()
    )
    rates = fit_int.median[df.time .== t]
    grid = reshape(rates[sortperm(df.region[df.time .== t])], n_cols, n_rows)
    heatmap!(ax, 1:n_cols, 1:n_rows, grid, colormap = Reverse(:RdYlBu))
end
fig

# You can see the high-risk area migrating across the grid over time — exactly
# the wave we simulated.
#
# ## Side-by-side comparison
#
# Let's put the two models next to each other:
fig = Figure(size = (900, 400))
ax1 = Axis(
    fig[1, 1],
    xlabel = "Region", ylabel = "Time period",
    title = "Additive (main effects only)"
)
ax2 = Axis(
    fig[1, 2],
    xlabel = "Region", ylabel = "Time period",
    title = "Separable (space-time interaction)"
)
heatmap!(ax1, 1:n_regions, 1:n_time, fitted_add', colormap = Reverse(:RdYlBu))
heatmap!(ax2, 1:n_regions, 1:n_time, fitted_int', colormap = Reverse(:RdYlBu))
fig

# ## Model comparison
#
# INLA computes several model comparison criteria. Let's see which model the data
# prefer:
println("Model comparison:")
println("─"^60)
for (name, res) in [("Additive", result_additive), ("Separable", result_interaction)]
    dic = res.accumulators[1].DIC
    p_d = res.accumulators[1].p_D
    mll = res.accumulators[2].log_marginal_likelihood
    waic = res.accumulators[3].WAIC
    println(
        "$name:  DIC = $(round(dic, digits = 1)) (p_D = $(round(p_d, digits = 1))),  " *
            "WAIC = $(round(waic, digits = 1)),  " *
            "log ML = $(round(mll, digits = 1))"
    )
end

# The separable model should have lower DIC and WAIC and higher marginal likelihood,
# confirming that the space-time interaction is real and worth modelling.
#
# ## Hyperparameter posteriors
#
# Let's examine what the data tell us about the smoothness in each dimension:
summary_df(result_interaction.hyperparameter_marginals)

# We can visualise them:
fig = Figure(size = (900, 400))
ax1 = Axis(
    fig[1, 1],
    xlabel = "Precision",
    title = "Temporal precision (τ_rw1)"
)
ax2 = Axis(
    fig[1, 2],
    xlabel = "Precision",
    title = "Region precision (τ_iid)"
)
plot!(ax1, result_interaction.hyperparameter_marginals.τ_rw1_separable)
plot!(ax2, result_interaction.hyperparameter_marginals.τ_iid_separable)
fig

# Higher temporal precision means smoother temporal evolution within regions.
# Higher IID precision means less variation between regions' temporal patterns.
#
# ## Summary
#
# Separable space-time models capture interactions that additive models cannot.
# The key ideas:
#
# - **Additive models** assume every region follows the same temporal pattern.
#   They are simpler and work well when there is no space-time interaction.
# - **Separable models** use a Kronecker product `Q_time ⊗ Q_space` to allow
#   region-specific temporal patterns while retaining temporal smoothness.
# - The `Separable()` constructor makes this easy: just wrap the temporal and
#   spatial components and use them in a formula.
# - **DIC**, **WAIC**, and the **marginal likelihood** help decide whether the
#   extra flexibility of the interaction model is justified by the data.
#
# These models are fundamental tools in spatial epidemiology, environmental
# monitoring, and any application where spatial patterns evolve over time.
# For background on the interaction taxonomy (Types I–IV), see
# Knorr-Held (2000), *Bayesian modelling of inseparable space-time variation
# in disease risk*.
