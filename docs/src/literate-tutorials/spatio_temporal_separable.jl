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
# - Why a structured interaction term **requires main effects** — and what goes
#   wrong if you leave them out
# - How to compare models using **DIC**, **WAIC**, and the **marginal likelihood**
# - How to visualise space-time effects using **heatmaps** and **animations**
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

## --- Animation utilities (hidden from rendered output) ---                         #hide
function _to_grids(fit_df, df, n_cols, n_rows, n_time)                               #hide
    return [                                                                         #hide
        let mask = df.time .== t                                                     #hide
                order = sortperm(df.region[mask])                                    #hide
                reshape(fit_df.median[mask][order], n_cols, n_rows)                  #hide
        end for t in 1:n_time                                                        #hide
    ]                                                                                #hide
end                                                                                  #hide
#hide
true_grids = [                                                                       #hide
    reshape(                                                                         #hide
            [                                                                        #hide
                exp(intercept + u_true[r] + v_true[t] + delta_true[t, r])            #hide
                for r in 1:n_regions                                                 #hide
            ], n_cols, n_rows                                                        #hide
        ) for t in 1:n_time                                                          #hide
]                       #hide
#hide
function _animate_vs_truth(true_g, fitted_g, filename, fitted_title)                  #hide
    cmin = min(minimum(minimum.(true_g)), minimum(minimum.(fitted_g)))                #hide
    cmax = max(maximum(maximum.(true_g)), maximum(maximum.(fitted_g)))                #hide
    t_obs = Observable(1)                                                            #hide
    fig = Figure(size = (700, 350))                                                  #hide
    a1 = Axis(                                                                       #hide
        fig[1, 1], title = "True rate",                                        #hide
        xlabel = "Column", ylabel = "Row", aspect = DataAspect()                     #hide
    )                    #hide
    a2 = Axis(                                                                       #hide
        fig[1, 2], title = fitted_title,                                       #hide
        xlabel = "Column", aspect = DataAspect()                                     #hide
    )                                    #hide
    hideydecorations!(a2, grid = false)                                              #hide
    hm = heatmap!(                                                                   #hide
        a1, 1:size(true_g[1], 1), 1:size(true_g[1], 2),                   #hide
        @lift(true_g[$t_obs]),                                                       #hide
        colormap = Reverse(:RdYlBu), colorrange = (cmin, cmax)                       #hide
    )                      #hide
    heatmap!(                                                                        #hide
        a2, 1:size(true_g[1], 1), 1:size(true_g[1], 2),                         #hide
        @lift(fitted_g[$t_obs]),                                                     #hide
        colormap = Reverse(:RdYlBu), colorrange = (cmin, cmax)                       #hide
    )                      #hide
    Colorbar(fig[1, 3], hm, label = "Rate")                                          #hide
    Label(                                                                           #hide
        fig[0, :], @lift("Time period $($t_obs) of $(length(true_g))"),             #hide
        fontsize = 18, font = :bold                                                  #hide
    )                                                 #hide
    return record(fig, filename, 1:length(true_g); framerate = 2) do t                      #hide
        t_obs[] = t                                                                  #hide
    end                                                                              #hide
end                                                                                  #hide

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
# The animation below makes this limitation vivid — the additive model's spatial
# pattern is essentially static, while the true pattern shifts over time:

_animate_vs_truth(                                                                   #hide
    true_grids, _to_grids(fit_add, df, n_cols, n_rows, n_time),        #hide
    "anim_additive.gif", "Additive model"                                            #hide
)                                           #hide
#md # ![True rates vs additive model fitted rates, animated over time](anim_additive.gif)

#
# ## Model 2: Interaction-only (a cautionary tale)
#
# Our first instinct might be to replace the additive model with a single
# **separable** interaction term. The precision matrix is a Kronecker product
#
# ```math
# Q = Q_{\text{time}} \otimes Q_{\text{space}}
# ```
#
# With `RandomWalk()` for time and `Besag(W)` for space, this enforces smoothness
# in *both* dimensions simultaneously. Let's try it as the sole random effect:
st = Separable(RandomWalk(), Besag(W, normalize_var = true))
f_interaction_only = @formula(y ~ 1 + st(time, region))

hp_interaction_only = @hyperparams begin
    (τ_rw1_separable ~ PCPrior.Precision(1.0, α = 0.01), transform = log)
    (τ_besag_separable ~ PCPrior.Precision(1.0, α = 0.01), transform = log)
end

result_interaction_only = inla(
    f_interaction_only, hp_interaction_only, df;
    family = Poisson, exposure = :expected, progress = false
)

obs_int_only = observation_marginals(result_interaction_only)
fit_int_only = summary_df(obs_int_only)

fitted_int_only = reshape(fit_int_only.median, n_regions, n_time)'
fig = Figure(size = (800, 400))
ax = Axis(
    fig[1, 1],
    xlabel = "Region", ylabel = "Time period",
    title = "Interaction-only model — fitted rates"
)
hm = heatmap!(ax, 1:n_regions, 1:n_time, fitted_int_only', colormap = Reverse(:RdYlBu))
Colorbar(fig[1, 2], hm)
fig

# Surprisingly, this model fits *worse* than the additive model! The fitted rates
# are over-smoothed and barely capture any structure:

_animate_vs_truth(                                                                   #hide
    true_grids, _to_grids(fit_int_only, df, n_cols, n_rows, n_time),   #hide
    "anim_interaction_only.gif", "Interaction only"                                  #hide
)                                  #hide
#md # ![True rates vs interaction-only model, animated over time](anim_interaction_only.gif)

# What went wrong?
#
# ### Why interaction-only fails: the constraint story
#
# The answer lies in how **constraints** work in Kronecker product models.
#
# Both the RW1 and the Besag components are **rank-deficient** — each has a
# sum-to-zero constraint. When combined via `Q_time ⊗ Q_space`, these constraints
# compose:
#
# - From RW1: $\sum_t x_{t,s} = 0$ for each region $s$ (25 constraints)
# - From Besag: $\sum_s x_{t,s} = 0$ for each time $t$ (12 constraints, minus 1 redundant)
#
# That gives **36 constraints** total. The crucial consequence: the Besag constraints
# force the **spatial mean to be zero at every time point**. This means the
# interaction field $\delta_{t,r}$ cannot capture any temporal main effect $v_t$
# (which shifts all regions up or down together) or spatial main effect $u_r$
# (which shifts all time points). It can only represent *pure interaction* —
# deviations from additivity.
#
# Without separate main effect terms to capture $u_r$ and $v_t$, the model is
# left trying to explain the entire signal through pure interaction alone, which
# it cannot do.
#
# ## Model 3: Main effects + interaction (the right way)
#
# The correct formulation adds the interaction term **alongside** main effects,
# following Knorr-Held's (2000) Type IV interaction structure:
#
# ```math
# \log \lambda_{t,r} = \mu + u_r + v_t + \delta_{t,r}
# ```
#
# where $u_r \sim \text{Besag}$, $v_t \sim \text{RW1}$, and
# $\delta_{t,r} \sim \text{Separable}(\text{RW1}, \text{Besag})$.
# The main effects absorb the marginal spatial and temporal patterns, freeing
# the interaction to capture only what changes across both dimensions:
f_full = @formula(y ~ 1 + spatial(region) + temporal(time) + st(time, region))

hp_full = @hyperparams begin
    (τ_besag ~ PCPrior.Precision(1.0, α = 0.01), transform = log)
    (τ_rw1 ~ PCPrior.Precision(1.0, α = 0.01), transform = log)
    (τ_rw1_separable ~ PCPrior.Precision(1.0, α = 0.01), transform = log)
    (τ_besag_separable ~ PCPrior.Precision(1.0, α = 0.01), transform = log)
end

result_full = inla(
    f_full, hp_full, df;
    family = Poisson, exposure = :expected, progress = false
)

# With 4 hyperparameters, INLA automatically switches from a grid exploration to a
# **Central Composite Design** (CCD), which scales as $O(2d^2 + 1)$ instead of
# exponentially. Let's see the fitted surface:
obs_full = observation_marginals(result_full)
fit_full = summary_df(obs_full)

fitted_full = reshape(fit_full.median, n_regions, n_time)'
fig = Figure(size = (800, 400))
ax = Axis(
    fig[1, 1],
    xlabel = "Region", ylabel = "Time period",
    title = "Full model — fitted rates"
)
hm = heatmap!(ax, 1:n_regions, 1:n_time, fitted_full', colormap = Reverse(:RdYlBu))
Colorbar(fig[1, 2], hm)
fig

# The full model captures the shifting wave pattern beautifully. Main effects
# handle the spatial gradient and temporal arc, while the interaction term adds
# the region-specific temporal deviations:

_animate_vs_truth(                                                                   #hide
    true_grids, _to_grids(fit_full, df, n_cols, n_rows, n_time),                     #hide
    "anim_full.gif", "Main effects + interaction"                                    #hide
)                                                                                    #hide
#md # ![True rates vs full model fitted rates, animated over time](anim_full.gif)

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
    rates = fit_full.median[df.time .== t]
    grid = reshape(rates[sortperm(df.region[df.time .== t])], n_cols, n_rows)
    heatmap!(ax, 1:n_cols, 1:n_rows, grid, colormap = Reverse(:RdYlBu))
end
fig

# You can see the high-risk area migrating across the grid over time — exactly
# the wave we simulated.
#
# ## Side-by-side comparison
#
# Let's put all three models next to each other:
fig = Figure(size = (1200, 400))
ax1 = Axis(
    fig[1, 1],
    xlabel = "Region", ylabel = "Time period",
    title = "Additive (main effects only)"
)
ax2 = Axis(
    fig[1, 2],
    xlabel = "Region", ylabel = "Time period",
    title = "Interaction only (no main effects)"
)
ax3 = Axis(
    fig[1, 3],
    xlabel = "Region", ylabel = "Time period",
    title = "Main effects + interaction"
)
heatmap!(ax1, 1:n_regions, 1:n_time, fitted_add', colormap = Reverse(:RdYlBu))
heatmap!(ax2, 1:n_regions, 1:n_time, fitted_int_only', colormap = Reverse(:RdYlBu))
heatmap!(ax3, 1:n_regions, 1:n_time, fitted_full', colormap = Reverse(:RdYlBu))
fig

# The difference is even more striking in animation — watch how only the full
# model tracks the true spatial pattern as it evolves:

grids_add = _to_grids(fit_add, df, n_cols, n_rows, n_time)                          #hide
grids_int_only = _to_grids(fit_int_only, df, n_cols, n_rows, n_time)                 #hide
grids_full = _to_grids(fit_full, df, n_cols, n_rows, n_time)                         #hide
all_vals = vcat(                                                                     #hide
    vec.(true_grids)..., vec.(grids_add)...,                              #hide
    vec.(grids_int_only)..., vec.(grids_full)...                                     #hide
)                                    #hide
cmin, cmax = extrema(all_vals)                                                       #hide
t_obs = Observable(1)                                                                #hide
fig_anim = Figure(size = (1100, 380))                                                #hide
a1 = Axis(                                                                           #hide
    fig_anim[1, 1], title = "Additive",                                       #hide
    xlabel = "Column", ylabel = "Row", aspect = DataAspect()                         #hide
)                        #hide
a2 = Axis(                                                                           #hide
    fig_anim[1, 2], title = "Interaction only",                                #hide
    xlabel = "Column", aspect = DataAspect()                                         #hide
)                                        #hide
a3 = Axis(                                                                           #hide
    fig_anim[1, 3], title = "Main effects +\ninteraction",                     #hide
    xlabel = "Column", aspect = DataAspect()                                         #hide
)                                        #hide
hideydecorations!(a2, grid = false)                                                  #hide
hideydecorations!(a3, grid = false)                                                  #hide
hm = heatmap!(                                                                       #hide
    a1, 1:n_cols, 1:n_rows, @lift(grids_add[$t_obs]),                     #hide
    colormap = Reverse(:RdYlBu), colorrange = (cmin, cmax)                           #hide
)                          #hide
heatmap!(                                                                            #hide
    a2, 1:n_cols, 1:n_rows, @lift(grids_int_only[$t_obs]),                      #hide
    colormap = Reverse(:RdYlBu), colorrange = (cmin, cmax)                           #hide
)                          #hide
heatmap!(                                                                            #hide
    a3, 1:n_cols, 1:n_rows, @lift(grids_full[$t_obs]),                          #hide
    colormap = Reverse(:RdYlBu), colorrange = (cmin, cmax)                           #hide
)                          #hide
Colorbar(fig_anim[1, 4], hm, label = "Rate")                                        #hide
Label(                                                                               #hide
    fig_anim[0, :], @lift("Time period $($t_obs) of $n_time"),                     #hide
    fontsize = 18, font = :bold                                                      #hide
)                                                     #hide
record(fig_anim, "anim_comparison.gif", 1:n_time; framerate = 2) do t                #hide
    t_obs[] = t                                                                      #hide
end                                                                                  #hide
#md # ![All three models compared side-by-side, animated over time](anim_comparison.gif)

#
# ## Model comparison
#
# INLA computes several model comparison criteria. Let's see which model the data
# prefer:
println("Model comparison:")
println("─"^70)
for (name, res) in [
        ("Additive        ", result_additive),
        ("Interaction only ", result_interaction_only),
        ("Full (main+inter)", result_full),
    ]
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

# The full model dominates on every criterion: lowest DIC and WAIC, highest marginal
# likelihood. The interaction-only model is the worst — confirming that a structured
# interaction term cannot substitute for main effects.
#
# ## Hyperparameter posteriors
#
# The full model has four hyperparameters. Let's examine what the data tell us
# about the smoothness in each component:
summary_df(result_full.hyperparameter_marginals)

# The main-effect precisions ($\tau_\text{besag}$, $\tau_\text{rw1}$) control the
# smoothness of the spatial and temporal main effects. The interaction precisions
# ($\tau_\text{rw1,sep}$, $\tau_\text{besag,sep}$) control the smoothness of the
# space-time deviations.
#
# ## Summary
#
# Separable space-time models capture interactions that additive models cannot,
# but they must be specified correctly. The key ideas:
#
# - **Additive models** assume every region follows the same temporal pattern.
#   They are simpler and work well when there is no space-time interaction.
# - **Separable interaction terms** use a Kronecker product $Q_\text{time} \otimes
#   Q_\text{space}$ to enforce smoothness in both dimensions while allowing
#   region-specific temporal patterns.
# - Structured spatial components like **Besag** impose **sum-to-zero constraints**
#   that make the interaction a *pure* interaction — it can only represent deviations
#   from additive structure. **Main effects must be included separately** to capture
#   the marginal spatial and temporal patterns.
# - The `Separable()` constructor makes specification easy: wrap the temporal and
#   spatial components, then combine with main effects in the formula.
# - **DIC**, **WAIC**, and the **marginal likelihood** help decide whether the
#   extra flexibility of the interaction model is justified by the data.
#
# These models are fundamental tools in spatial epidemiology, environmental
# monitoring, and any application where spatial patterns evolve over time.
# For background on the interaction taxonomy (Types I–IV), see
# Knorr-Held (2000), *Bayesian modelling of inseparable space-time variation
# in disease risk*.
