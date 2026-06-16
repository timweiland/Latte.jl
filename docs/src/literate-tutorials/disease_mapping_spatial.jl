# # Disease mapping with spatial models
#
# This tutorial analyzes lung cancer mortality across Pennsylvania counties with the
# BYM model, estimating disease risk while accounting for spatial correlation between
# neighboring counties.
#
# ## What is disease mapping?
#
# Disease mapping estimates disease risk across geographic areas. Crude rates are
# unstable in areas with small populations, and neighboring areas often share risk
# through common environmental or socioeconomic factors.
#
# The [BYM model (Besag-York-Mollié)](#ref-bym) is a standard choice for areal data. It
# splits the county effect into a spatial component (a [Besag/ICAR model](#ref-besag)) for
# spatially structured variation and an unstructured component (IID) for region-specific
# deviations.
#
# ## The Pennsylvania lung cancer dataset
#
# The data record lung cancer mortality across Pennsylvania counties, with observed
# cases and population broken down by age, race, and gender. That breakdown is what
# lets us compute expected counts by indirect standardization.
#
# Loading the data:
using CodecBzip2  # needed to decompress .rda files
using RData
using DataFrames
using StatsBase
data_dir = joinpath(@__DIR__, "data")
mkpath(data_dir)
local_rda = joinpath(data_dir, "pennLC_sf.rda")
if !isfile(local_rda)
    repo_url = "https://github.com/rudeboybert/SpatialEpi/raw/refs/heads/master/data/pennLC_sf.rda"
    try
        download(repo_url, local_rda)
    catch err
        error(
            "Could not download dataset (are you offline?). " *
                "Place an RData file at $(local_rda) or pass your own DataFrame."
        )
    end
end

pennLC_sf = load(local_rda)["pennLC_sf"]
first(pennLC_sf, 5)

# ### Preprocessing
# We aggregate the data to the county level and add the quantities the model needs:
# - integer county IDs for the spatial model,
# - expected counts from indirect standardization (population times the overall rate across all counties),
# - the standardized incidence ratio (SIR), observed over expected,
# - and county geometries converted to LibGEOS polygons.
county_data = combine(
    groupby(pennLC_sf, :county),
    :cases => sum => :cases,
    :population => sum => :population,
    :geometry => first => :geometry  # keep geometry for spatial adjacency
)
county_data.county_id = 1:nrow(county_data)
total_cases = sum(county_data.cases)
total_pop = sum(county_data.population)
overall_rate = total_cases / total_pop
county_data.expected = county_data.population .* overall_rate
county_data.SIR = county_data.cases ./ county_data.expected
using LibGEOS
county_data.geometry = [
    LibGEOS.Polygon([[mat[i, :] for i in 1:size(mat, 1)]])
        for mat in county_data.geometry
]
sort!(county_data, :county)
first(county_data, 5)

# The scale of the data:

println("Pennsylvania lung cancer data:")
println("  Counties: ", nrow(county_data))
println("  Total cases: ", total_cases)
println("  Total population: ", total_pop)
println("  Overall rate: ", round(overall_rate * 100000, digits = 2), " per 100,000")

# ### SIR summary
#
# Counties with SIR > 1 have higher than expected lung cancer mortality, those with
# SIR < 1 lower than expected. Mapping the SIR shows the spread across the state.
using AlgebraOfGraphics, CairoMakie
data(county_data) * mapping(:geometry, color = :SIR) * visual(Poly) |> draw

# The variability is large: some counties sit well above 1, others well below. The
# spatial model stabilizes these estimates by borrowing strength from neighboring
# counties.

# ## The BYM model
#
# For each county i we model the observed cases as $Y_i \sim \text{Poisson}(E_i \cdot
# \exp(\eta_i))$ with linear predictor $\eta_i = \alpha + \text{spatial}_i +
# \text{unstructured}_i$. Here $E_i$ is the expected count (the exposure), $\alpha$ the
# overall log-risk intercept, $\text{spatial}_i$ the Besag spatially structured effect,
# and $\text{unstructured}_i$ the IID effect.
#
# The Besag model needs an adjacency structure that says which counties are neighbors.
# GaussianMarkovRandomFields.jl builds a contiguity-based adjacency matrix from the
# county geometries:
using GaussianMarkovRandomFields, SparseArrays
geom_collection = LibGEOS.GeometryCollection(county_data.geometry)
W = contiguity_adjacency(geom_collection)
size(W), nnz(W) ÷ 2  # counties and number of shared borders

# We write the BYM model as an `@latte` block with three latent pieces. The intercept
# `β` is a one-element `MvNormal`, which gives the adapter a simple random-effect
# structure to recognise. The spatial component `spatial ~ BesagModel(W; …)(τ =
# τ_besag)` returns a `ConstrainedGMRF`: `BesagModel` enforces a sum-to-zero constraint
# per connected component of the adjacency graph, which keeps the intercept identified
# from the spatial field. The unstructured component is `u ~ IIDModel(n)(τ = τ_iid)`.
using Latte
using Distributions
using LinearAlgebra
n = nrow(county_data)

@latte function bym_model(cases, expected, n, W)
    τ_besag ~ PCPrior.Precision(1.0, α = 0.01)
    τ_iid ~ PCPrior.Precision(1.0, α = 0.01)
    β ~ MvNormal(zeros(1), 100.0 * I(1))
    spatial ~ BesagModel(W; normalize_var = Val{true}())(τ = τ_besag)
    u ~ IIDModel(n)(τ = τ_iid)
    for i in eachindex(cases)
        cases[i] ~ Poisson(
            expected[i] * exp(β[1] + spatial[i] + u[i])
        )
    end
end

# ## Prior specification
#
# Both precisions get [PC (Penalized Complexity) priors](#ref-pc-priors), which favor the
# simpler model unless the data pull away from it. `PCPrior.Precision(1.0, α = 0.01)`
# calibrates that preference as $P(\sigma > 1) = 0.01$: a 1% prior probability that the
# standard deviation exceeds 1.0 on the log-risk scale, roughly a three-fold change in risk.
#
# ## Running INLA
#
# The `expected` exposure enters the likelihood through `expected[i] * exp(…)`,
# which the adapter picks up as a log-exposure offset automatically.
lgm = bym_model(county_data.cases, county_data.expected, n, W)
inla_result = inla(lgm, county_data.cases; progress = false)

# This returns posterior marginals for every parameter.
#
# ## Hyperparameter posteriors
#
# The two precision parameters summarize how much spatial and unstructured variation
# the model attributes to the data:
summary_df(hyperparameter_marginals(inla_result))

# Evaluating each marginal density on its own grid gives a tidy frame we can hand to
# AlgebraOfGraphics, faceted by component:
function density_frame(dist, label)
    grid = range(quantile(dist, 0.001), quantile(dist, 0.999); length = 200)
    return DataFrame(τ = grid, density = pdf.(Ref(dist), grid), component = label)
end
hyper_density = vcat(
    density_frame(hyperparameter_marginals(inla_result, :τ_besag)[1], "Spatial (τ_besag)"),
    density_frame(hyperparameter_marginals(inla_result, :τ_iid)[1], "Unstructured (τ_iid)"),
)
data(hyper_density) *
    mapping(:τ => "Precision", :density => "Posterior density", layout = :component) *
    visual(Lines) |> draw(; facet = (; linkxaxes = :none))

# ## Visualizing the latent spatial and unstructured effects
#
# Before turning to relative risk, it helps to look at the model components directly.
# The BYM model splits each county's effect into a spatial part (shared with neighbors)
# and an unstructured part (county-specific). These base latent marginals are the
# building blocks that combine into the linear predictor.
#
# `base_latent_marginals(result)` returns them in the order the variables appear in the
# model body: `β`, then `spatial`, then `u`. Index 1 is `β`, indices 2:(1+n) are the
# `spatial` effects, and the next n are `u`.
n_counties = nrow(county_data)
spatial_offset = 1
iid_offset = 1 + n_counties
base_marginals = base_latent_marginals(inla_result)
length(base_marginals)  # β (1) + spatial (n) + u (n)

# A joyplot shows how these effects vary across a selection of counties:
sorted_idx = sortperm(county_data.SIR, rev = true)
component_indices = vcat(
    sorted_idx[1:3],              # 3 highest SIR
    sorted_idx[33:35],            # 3 middle SIR
    sorted_idx[(end - 2):end]
)        # 3 lowest SIR

spatial_dists = [base_marginals[spatial_offset + i] for i in component_indices]
spatial_labels = [county_data.county[i] * " (spatial)" for i in component_indices]
iid_dists = [base_marginals[iid_offset + i] for i in component_indices]
iid_labels = [county_data.county[i] * " (IID)" for i in component_indices]

# Raw Makie: `joyplot!` is Latte's ridgeline recipe for marginal objects; AlgebraOfGraphics
# has no equivalent and cannot consume opaque distributions directly.
fig_components = Figure(size = (1200, 600))
ax_spatial = Axis(
    fig_components[1, 1],
    title = "Spatial Component (Besag)",
    xlabel = "Effect",
    ylabel = ""
)
ax_iid = Axis(
    fig_components[1, 2],
    title = "Unstructured Component (IID)",
    xlabel = "Effect",
    ylabel = ""
)
joyplot!(
    ax_spatial, spatial_dists;
    labels = spatial_labels,
)
joyplot!(
    ax_iid, iid_dists;
    labels = iid_labels,
)
fig_components

# The spatial component varies smoothly and is shared between neighboring counties,
# while the IID component captures county-specific deviations. The spatial effects sit
# in tighter distributions because the spatial structure regularizes them; the IID
# effects vary more freely. This is the separation of smooth spatial signal from local
# noise that the BYM model is built to do.

# ## Posterior relative risk estimates
#
# The quantity of interest is the relative risk for each county. `observation_marginals`
# returns marginals of the fitted count $E_i \cdot \exp(\beta + \text{spatial}_i + u_i)$,
# following R-INLA's convention that fitted values include the offset. Dividing each
# summary column by the expected count puts the summaries back on the relative-risk
# scale:
obs_marginals = observation_marginals(inla_result)
fitted_summary = summary_df(obs_marginals)
risk_summary = DataFrame(
    county = county_data.county,
    SIR = county_data.SIR,
    median = fitted_summary.median ./ county_data.expected,
    mean = fitted_summary.mean ./ county_data.expected,
    q2_5 = fitted_summary.q2_5 ./ county_data.expected,
    q97_5 = fitted_summary.q97_5 ./ county_data.expected,
    geometry = county_data.geometry,
)
first(select(risk_summary, :county, :SIR, :median, :q2_5, :q97_5), 10)

# ## Comparing smoothed vs crude estimates
#
# Overlaying the crude SIR and the smoothed posterior estimate on a shared county axis
# shows what the spatial model buys us. The posterior layer carries its 95% credible
# interval as range bars:
risk_summary.county_id = 1:nrow(risk_summary)
risk_layers =
    mapping(:county_id => "County", :q2_5, :q97_5) *
    visual(Rangebars, color = (:steelblue, 0.3), whiskerwidth = 0) +
    mapping(:county_id => "County", :SIR => "Risk ratio", color = direct("Crude SIR")) *
    visual(Scatter, markersize = 6) +
    mapping(
    :county_id => "County", :median => "Risk ratio",
    color = direct("Posterior median (BYM)"),
) * visual(Scatter, markersize = 8)
draw(
    data(risk_summary) * risk_layers;
    axis = (title = "Crude SIR vs posterior relative risk",),
)

# The posterior estimates are more stable than the crude SIRs. Extreme values shrink
# toward 1.0, spatial smoothing pulls neighboring counties toward similar estimates, and
# the 95% credible intervals quantify the remaining uncertainty.

# ## Visualizing posterior distributions with joyplots
#
# Credible intervals report two numbers; a joyplot shows the full posterior. Picking the
# 5 highest- and 5 lowest-risk counties brings out how the distributions differ.
#
# The fitted-count marginals live on the county-specific count scale, so their densities
# span very different widths (a county with thousands of cases has a much flatter density
# than one with fifty). Stacked as-is, every ridge would be nearly flat. Dividing each
# marginal by its expected count puts all of them on the common relative-risk scale, where
# the curves are comparable and legible. `pushforward` carries the full distribution
# through the `Y_i / E_i` map, not just its summaries:
using Bijectors: Scale
sorted_by_risk = sortperm(risk_summary.median, rev = true)
selected_indices = vcat(
    sorted_by_risk[1:5],      # 5 highest risk
    sorted_by_risk[(end - 4):end]
) # 5 lowest risk

selected_dists = [
    pushforward(obs_marginals[i], Scale(1.0 / county_data.expected[i]))
        for i in selected_indices
]
selected_labels = [county_data.county[i] for i in selected_indices]

fig_joy = joyplot(
    selected_dists;
    labels = selected_labels,
    title = "Posterior relative-risk distributions (selected counties)",
    xlabel = "Relative risk",
)
fig_joy

# The high-risk counties have relative-risk distributions sitting above 1.0, the low-risk
# counties below it. The width of each distribution reflects uncertainty: counties with
# smaller populations have wider, less certain distributions.

# ## Identifying high-risk counties
#
# A county shows significantly elevated risk when its entire 95% credible interval lies
# above 1.0:
high_risk = risk_summary[
    risk_summary.q2_5 .> 1.0,
    [:county, :median, :q2_5, :q97_5],
]

# Reduced risk is the mirror image: the whole interval below 1.0.
low_risk = risk_summary[
    risk_summary.q97_5 .< 1.0,
    [:county, :median, :q2_5, :q97_5],
]

# ## Exceedance probabilities
#
# For each county we can compute the probability that relative risk exceeds a threshold,
# say 1.1 for 10% elevated risk. The marginals are on the fitted-count scale, so
# $\mathbb{P}(\text{RR} > t)$ equals $\mathbb{P}(\text{fitted} > t \cdot E_i)$:
threshold = 1.1
risk_summary.exc_prob = [
    1 - cdf(obs_marginals[i], threshold * county_data.expected[i])
        for i in 1:nrow(risk_summary)
]
data(risk_summary) * mapping(:geometry, color = :exc_prob) * visual(Poly) |> draw

# The counties where that probability exceeds 0.95:
risk_summary[
    risk_summary.exc_prob .> 0.95,
    [:county, :median, :exc_prob],
]

# ## Model diagnostics and comparison
#
# The default accumulators report the DIC (with its effective parameter count `p_D`) and
# the log marginal likelihood. To judge whether the spatial component earns its place, we
# fit a second model with only unstructured random effects:
@latte function iid_only(cases, expected, n)
    τ_iid ~ PCPrior.Precision(1.0, α = 0.01)
    β ~ MvNormal(zeros(1), 100.0 * I(1))
    u ~ IIDModel(n)(τ = τ_iid)
    for i in eachindex(cases)
        cases[i] ~ Poisson(expected[i] * exp(β[1] + u[i]))
    end
end

lgm_iid = iid_only(county_data.cases, county_data.expected, n)
inla_result_iid = inla(lgm_iid, county_data.cases; progress = false)

# Putting the two fits side by side:
DataFrame(
    model = ["BYM (spatial + unstructured)", "IID only"],
    DIC = round.(
        [inla_result.accumulators[1].DIC, inla_result_iid.accumulators[1].DIC],
        digits = 2,
    ),
    p_D = round.(
        [inla_result.accumulators[1].p_D, inla_result_iid.accumulators[1].p_D],
        digits = 2,
    ),
    log_ML = round.(
        [
            inla_result.accumulators[2].log_marginal_likelihood,
            inla_result_iid.accumulators[2].log_marginal_likelihood,
        ],
        digits = 2,
    ),
)

# A lower DIC and higher marginal likelihood for the BYM model would indicate that the
# data carry spatial structure the IID model cannot capture.

# ## Summary
#
# We started from raw Pennsylvania lung cancer counts and used the BYM model to separate
# spatial pattern from noise. Borrowing strength across neighboring counties produced
# more stable risk estimates than the crude SIRs, which matters most where populations
# are small.
#
# The spatial component captured smooth geographic variation and the unstructured
# component the county-specific deviations. Looking at the full posteriors through
# joyplots showed how uncertainty changes across the risk spectrum, and exceedance
# probabilities turned the posteriors into direct statements about which counties are
# likely above a given risk threshold.
#
# The BYM model is a standard tool in spatial epidemiology for count data over small
# areas where neighboring regions are expected to be similar.
#
# To go further, see [Getting started with Latte.jl](@ref) for the INLA basics, or the
# main documentation for how hyperparameter priors work.
#
# ## References
#
# ```@raw html
# <div class="ref-grid-2">
# <PaperCite
#   tag="BYM"
#   title="Bayesian Image Restoration, with Two Applications in Spatial Statistics"
#   authors="J. Besag, J. York & A. Mollié"
#   venue="Ann. Inst. Statist. Math." year="1991"
#   doi="10.1007/BF00116466"
#   url="https://doi.org/10.1007/BF00116466"
#   abstract="Introduces the Besag-York-Mollié model: an areal count model combining a spatially structured intrinsic CAR effect with an unstructured IID effect, the standard decomposition for disease mapping." />
# <PaperCite
#   tag="Besag"
#   title="Spatial Interaction and the Statistical Analysis of Lattice Systems"
#   authors="J. Besag"
#   venue="J. R. Statist. Soc. B" year="1974"
#   doi="10.1111/j.2517-6161.1974.tb00999.x"
#   url="https://doi.org/10.1111/j.2517-6161.1974.tb00999.x"
#   abstract="The foundational paper on conditional autoregressive (CAR) models, the intrinsic Gaussian Markov random field that the spatial component of the BYM model is built from." />
# <PaperCite
#   tag="PC priors"
#   title="Penalising Model Component Complexity: A Principled, Practical Approach to Constructing Priors"
#   authors="D. Simpson, H. Rue, A. Riebler, T. G. Martins & S. H. Sørbye"
#   venue="Statistical Science" year="2017"
#   doi="10.1214/16-STS576"
#   url="https://doi.org/10.1214/16-STS576"
#   abstract="Penalised Complexity priors shrink each model component toward a simpler base model, with interpretable scaling such as P(σ > U) = α; the priors used on both BYM precisions here." />
# </div>
# ```
