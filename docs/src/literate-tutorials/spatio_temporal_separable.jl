# # Spatio-temporal disease surveillance
#
# Disease incidence varies across both space and time. An additive model can capture
# where risk is high and when it peaks, but it assumes every region follows the
# same temporal trend. In practice spatial patterns often shift over time: an
# emerging epidemic may spread outward from a focus, or an intervention may reduce
# risk in some regions earlier than others.
#
# This tutorial uses separable space-time models to capture those interactions. It
# covers how to specify a separable interaction in a `@latte` model, how the
# Kronecker-product structure `Q_time ⊗ Q_space` enforces smoothness in both
# dimensions at once, why a structured interaction term needs accompanying main
# effects, and how DIC, WAIC, and the marginal likelihood help choose between the
# fitted models.
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
(observations = nrow(df), mean_count = round(mean(df.y), digits = 1))

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

# Both a spatial gradient (left vs right) and a temporal arc (middle rows brighter)
# are visible. The bright patch also shifts across regions over time, which is the
# interaction signal that an additive model cannot capture.

using Latte
using Distributions
using GaussianMarkovRandomFields: BesagModel, RWModel, SeparableModel
using LinearAlgebra

# The three multi-hyperparameter models below share a set of `inla` keyword
# arguments: automatic exploration-strategy selection, a Gaussian latent
# marginalisation, and an explicit set of accumulator strategies.
# `AutoExplorationStrategy` matters most here. A Cartesian grid scales as
# (points-per-dim)^(n_hp), which is manageable for the 2-hyperparameter models
# but reaches hundreds of points for the 4-hyperparameter `full_model`. Auto keeps
# a grid in 2-D and switches to a Central Composite Design (~25 points) in 4-D.
# The strategies are immutable configs; `inla()` materialises them into fresh
# accumulator state on each call, so this tuple is safely reused.
const INLA_KWARGS = (
    progress = false,
    exploration_strategy = AutoExplorationStrategy(),
    latent_marginalization_method = GaussianMarginal(),
    accumulators = (
        DICStrategy(),
        MarginalLogLikelihoodStrategy(),
        WAICStrategy(),
    ),
)

# ## Model 1: Additive main effects
#
# We start with a model that assumes spatial and temporal effects are independent:
#
# ```math
# \log \lambda_{t,r} = \mu + u_r + v_t
# ```
#
# The spatial component is a [Besag (ICAR)](#ref-besag) model and the temporal
# component is a first-order random walk, each with a [PC prior](#ref-pc-priors)
# on its precision:
@latte function additive_model(y, expected, region, time, n_regions, n_time, W)
    τ_besag ~ PCPrior.Precision(1.0, α = 0.01)
    τ_rw1 ~ PCPrior.Precision(1.0, α = 0.01)
    β ~ MvNormal(zeros(1), 100.0 * I(1))
    u ~ BesagModel(W; normalize_var = Val{true}())(τ = τ_besag)
    v ~ RWModel{1}(n_time)(τ = τ_rw1)
    for i in eachindex(y)
        y[i] ~ Poisson(
            expected[i] * exp(β[1] + u[region[i]] + v[time[i]])
        )
    end
end

lgm_add = additive_model(df.y, df.expected, df.region, df.time, n_regions, n_time, W)
result_additive = inla(lgm_add, df.y; INLA_KWARGS...)

# The fitted rates come from `observation_marginals`, which reports fitted counts
# with the exposure included; dividing by `expected` recovers a relative-risk
# equivalent for the heatmap.
obs_add = observation_marginals(result_additive)
fit_add = summary_df(obs_add)
fitted_add = reshape(fit_add.median ./ df.expected, n_regions, n_time)'
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
# ## Model 2: Interaction-only (a cautionary tale)
#
# A tempting alternative is to replace the additive model with a single separable
# interaction term. Its precision matrix is a Kronecker product
#
# ```math
# Q = Q_{\text{time}} \otimes Q_{\text{space}}
# ```
#
# With `RWModel{1}` for time and `BesagModel(W)` for space, this enforces
# smoothness in both dimensions at once. We try it here as the sole random effect.
#
# The Separable field has size `n_time × n_regions`, flattened in
# row-major order (`δ[(t-1)*n_regions + r]` for region `r` at time `t`). This
# matches the `kron(Q_time, Q_space)` convention.
@latte function interaction_only_model(y, expected, region, time, n_regions, n_time, W)
    τ_rw1_separable ~ PCPrior.Precision(1.0, α = 0.01)
    τ_besag_separable ~ PCPrior.Precision(1.0, α = 0.01)
    β ~ MvNormal(zeros(1), 100.0 * I(1))
    δ ~ SeparableModel(
        RWModel{1}(n_time),
        BesagModel(W; normalize_var = Val{true}()),
    )(τ_rw1 = τ_rw1_separable, τ_besag = τ_besag_separable)
    for i in eachindex(y)
        δ_idx = (time[i] - 1) * n_regions + region[i]
        y[i] ~ Poisson(
            expected[i] * exp(β[1] + δ[δ_idx])
        )
    end
end

lgm_int_only = interaction_only_model(df.y, df.expected, df.region, df.time, n_regions, n_time, W)
result_interaction_only = inla(lgm_int_only, df.y; INLA_KWARGS...)

obs_int_only = observation_marginals(result_interaction_only)
fit_int_only = summary_df(obs_int_only)
fitted_int_only = reshape(fit_int_only.median ./ df.expected, n_regions, n_time)'
fig = Figure(size = (800, 400))
ax = Axis(
    fig[1, 1],
    xlabel = "Region", ylabel = "Time period",
    title = "Interaction-only model — fitted rates"
)
hm = heatmap!(ax, 1:n_regions, 1:n_time, fitted_int_only', colormap = Reverse(:RdYlBu))
Colorbar(fig[1, 2], hm)
fig

# This model fits worse than the additive one. The fitted rates are over-smoothed
# and barely capture any structure.
#
# ### Why interaction-only fails: the constraint story
#
# The explanation lies in how the constraints of the two components compose under
# the Kronecker product.
#
# Both the RW1 and the Besag components are rank-deficient: each carries a
# sum-to-zero constraint. Combining them through `Q_time ⊗ Q_space` makes those
# constraints compose. The RW1 part contributes $\sum_t x_{t,s} = 0$ for each region
# $s$ (25 constraints), and the Besag part contributes $\sum_s x_{t,s} = 0$ for each
# time $t$ (12 constraints, one of them redundant).
#
# That leaves 36 constraints in total. The consequence that matters is that the
# Besag constraints force the spatial mean to zero at every time point. The
# interaction field $\delta_{t,r}$ then cannot carry any temporal main effect $v_t$
# (which shifts all regions up or down together) or spatial main effect $u_r$
# (which shifts all time points), so it represents pure interaction: deviations
# from additivity and nothing else.
#
# With no separate main-effect terms for $u_r$ and $v_t$, the model is left trying
# to explain the entire signal through pure interaction, which it cannot do.
#
# ## Model 3: Main effects + interaction (the right way)
#
# The correct formulation adds the interaction term alongside main effects,
# following [Knorr-Held's (2000)](#ref-knorr-held) Type IV interaction structure:
#
# ```math
# \log \lambda_{t,r} = \mu + u_r + v_t + \delta_{t,r}
# ```
#
# where $u_r \sim \text{Besag}$, $v_t \sim \text{RW1}$, and
# $\delta_{t,r} \sim \text{Separable}(\text{RW1}, \text{Besag})$.
# The main effects absorb the marginal spatial and temporal patterns, freeing
# the interaction to capture only what changes across both dimensions:
@latte function full_model(y, expected, region, time, n_regions, n_time, W)
    τ_besag ~ PCPrior.Precision(1.0, α = 0.01)
    τ_rw1 ~ PCPrior.Precision(1.0, α = 0.01)
    τ_rw1_separable ~ PCPrior.Precision(1.0, α = 0.01)
    τ_besag_separable ~ PCPrior.Precision(1.0, α = 0.01)
    β ~ MvNormal(zeros(1), 100.0 * I(1))
    u ~ BesagModel(W; normalize_var = Val{true}())(τ = τ_besag)
    v ~ RWModel{1}(n_time)(τ = τ_rw1)
    δ ~ SeparableModel(
        RWModel{1}(n_time),
        BesagModel(W; normalize_var = Val{true}()),
    )(τ_rw1 = τ_rw1_separable, τ_besag = τ_besag_separable)
    for i in eachindex(y)
        δ_idx = (time[i] - 1) * n_regions + region[i]
        y[i] ~ Poisson(
            expected[i] * exp(β[1] + u[region[i]] + v[time[i]] + δ[δ_idx])
        )
    end
end

lgm_full = full_model(df.y, df.expected, df.region, df.time, n_regions, n_time, W)
result_full = inla(lgm_full, df.y; INLA_KWARGS...)

# Let's see the fitted surface:
obs_full = observation_marginals(result_full)
fit_full = summary_df(obs_full)
fitted_full = reshape(fit_full.median ./ df.expected, n_regions, n_time)'
fig = Figure(size = (800, 400))
ax = Axis(
    fig[1, 1],
    xlabel = "Region", ylabel = "Time period",
    title = "Full model — fitted rates"
)
hm = heatmap!(ax, 1:n_regions, 1:n_time, fitted_full', colormap = Reverse(:RdYlBu))
Colorbar(fig[1, 2], hm)
fig

# The full model recovers the shifting wave. Main effects handle the spatial
# gradient and temporal arc, and the interaction term adds the region-specific
# temporal deviations.
#
# ## Spatial snapshots at selected time points
#
# Spatial maps at four time points show the interaction directly. We reshape each
# time slice back into the 5×5 grid:
snapshot_times = [1, 4, 8, 12]

fig = Figure(size = (900, 250))
for (k, t) in enumerate(snapshot_times)
    local ax = Axis(
        fig[1, k],
        title = "t = $t",
        xlabel = k == 1 ? "Column" : "", ylabel = k == 1 ? "Row" : "",
        aspect = DataAspect()
    )
    rates = (fit_full.median ./ df.expected)[df.time .== t]
    grid = reshape(rates[sortperm(df.region[df.time .== t])], n_cols, n_rows)
    heatmap!(ax, 1:n_cols, 1:n_rows, grid, colormap = Reverse(:RdYlBu))
end
fig

# The high-risk area migrates across the grid over time, tracing the wave we
# simulated.
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

# ## Model comparison
#
# INLA computes several model comparison criteria as part of inference, and they
# land in `result.accumulators`. DIC and WAIC both weigh fit against complexity,
# with lower values preferred; the log marginal likelihood is a model-selection
# score where higher is preferred. The accumulator tuple we configured in
# `INLA_KWARGS` orders them as DIC, log marginal likelihood, then WAIC.
comparison = DataFrame(
    model = String[], DIC = Float64[], p_D = Float64[],
    WAIC = Float64[], log_ML = Float64[],
)
for (name, res) in [
        ("Additive", result_additive),
        ("Interaction only", result_interaction_only),
        ("Full (main+inter)", result_full),
    ]
    push!(
        comparison, (
            name,
            round(res.accumulators[1].DIC, digits = 1),
            round(res.accumulators[1].p_D, digits = 1),
            round(res.accumulators[3].WAIC, digits = 1),
            round(res.accumulators[2].log_marginal_likelihood, digits = 1),
        )
    )
end
comparison

# The full model leads on every criterion: lowest DIC and WAIC, highest marginal
# likelihood. The interaction-only model trails both others, confirming that a
# structured interaction term cannot substitute for main effects.
#
# ## Hyperparameter posteriors
#
# The full model has four hyperparameters. The summary table reads each marginal
# off the natural (declared) scale; we add a column naming the components in
# declared order:
hp_full = summary_df(hyperparameter_marginals(result_full))
insertcols!(hp_full, 1, :parameter => collect(keys(hyperparameter_groups(result_full))))

# The main-effect precisions ($\tau_\text{besag}$, $\tau_\text{rw1}$) control the
# smoothness of the spatial and temporal main effects. The interaction precisions
# ($\tau_\text{rw1,sep}$, $\tau_\text{besag,sep}$) control the smoothness of the
# space-time deviations.
#
# ## Summary
#
# Separable space-time models capture interactions that additive models cannot,
# but they have to be specified correctly. The points worth keeping:
#
# - An additive model assumes every region follows the same temporal pattern. It
#   is simpler and works well when there is no space-time interaction.
# - A separable interaction term uses a Kronecker product $Q_\text{time} \otimes
#   Q_\text{space}$ to enforce smoothness in both dimensions while allowing
#   region-specific temporal patterns.
# - A structured spatial component like Besag imposes sum-to-zero constraints that
#   make the interaction pure: it represents only deviations from additive
#   structure, so the main effects have to be included separately to capture the
#   marginal spatial and temporal patterns.
# - `SeparableModel(time_component, space_component)` composes the two precision
#   matrices via Kronecker product; inside the `@latte` block it lives on an
#   `n_time × n_regions` flat vector indexed by `(time-1)*n_regions + region`.
# - DIC, WAIC, and the marginal likelihood help decide whether the extra
#   flexibility of the interaction model is justified by the data.
#
# These models see use in spatial epidemiology, environmental monitoring, and other
# settings where spatial patterns evolve over time. For background on the
# interaction taxonomy (Types I–IV), see [Knorr-Held (2000)](#ref-knorr-held).
#
# ## References
#
# ```@raw html
# <div class="ref-grid-2">
# <PaperCite
#   tag="Knorr-Held"
#   title="Bayesian Modelling of Inseparable Space-Time Variation in Disease Risk"
#   authors="L. Knorr-Held"
#   venue="Statistics in Medicine" year="2000"
#   url="https://pubmed.ncbi.nlm.nih.gov/10960871/"
#   abstract="Introduces the Type I–IV taxonomy of space-time interaction priors as Kronecker products of the spatial and temporal main-effect structures; the Type IV (Besag ⊗ RW) interaction used in the full model here." />
# <PaperCite
#   tag="Besag"
#   title="Spatial Interaction and the Statistical Analysis of Lattice Systems"
#   authors="J. Besag"
#   venue="J. R. Statist. Soc. B" year="1974"
#   doi="10.1111/j.2517-6161.1974.tb00999.x"
#   url="https://doi.org/10.1111/j.2517-6161.1974.tb00999.x"
#   abstract="The foundational paper on conditional autoregressive (CAR) models, the intrinsic Gaussian Markov random field that the spatial (Besag/ICAR) component is built from." />
# <PaperCite
#   tag="PC priors"
#   title="Penalising Model Component Complexity: A Principled, Practical Approach to Constructing Priors"
#   authors="D. Simpson, H. Rue, A. Riebler, T. G. Martins & S. H. Sørbye"
#   venue="Statistical Science" year="2017"
#   doi="10.1214/16-STS576"
#   url="https://doi.org/10.1214/16-STS576"
#   abstract="Penalised Complexity priors shrink each model component toward a simpler base model, with interpretable scaling such as P(σ > U) = α; the priors used on every precision hyperparameter here." />
# </div>
# ```
