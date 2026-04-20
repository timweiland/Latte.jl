# # Disease mapping with spatial models
#
# Welcome! In this tutorial we'll analyze lung cancer mortality across Pennsylvania counties
# using spatial disease mapping. We'll use the BYM model to estimate disease risk while
# accounting for spatial correlation between neighboring counties.
#
# ## What is disease mapping?
#
# Disease mapping estimates disease risk across geographic areas. The challenge is that
# crude rates are unstable in areas with small populations, and neighboring areas often
# have similar risk due to shared environmental or socioeconomic factors.
#
# The **BYM model** (Besag-York-Mollié) is a classic choice for areal data. It combines:
# - A **spatial component** (Besag/ICAR model) for spatially structured variation
# - An **unstructured component** (IID) for region-specific random effects
#
# ## The Pennsylvania lung cancer dataset
#
# We'll use lung cancer mortality data from Pennsylvania counties. The dataset includes
# observed cases and population broken down by age, race, and gender, which allows us
# to compute expected counts based on indirect standardization.
#
# First, let's load the data:
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
# Next, we preprocess the data. Concretely, we:
# - Aggregate data to the county level
# - Create integer county IDs (needed for spatial model)
# - Compute expected counts using indirect standardization (expected = population × overall rate across all counties)
# - Compute the standardized incidence ratio (SIR) = observed / expected
# - Convert geometries to LibGEOS polygons
using DataFrames
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

# Let's get a rough idea of the scale of the data:

println("Pennsylvania lung cancer data:")
println("  Counties: ", nrow(county_data))
println("  Total cases: ", total_cases)
println("  Total population: ", total_pop)
println("  Overall rate: ", round(overall_rate * 100000, digits = 2), " per 100,000")

# ### SIR summary
#
# The SIRs show substantial variation across counties.

# Counties with SIR > 1 have higher than expected lung cancer mortality,
# while SIR < 1 indicates lower than expected mortality.
using AlgebraOfGraphics, CairoMakie
data(county_data) * mapping(:geometry, color = :SIR) * visual(Poly) |> draw

# Notice the high variability! Some counties have SIRs well above 1 (higher risk),
# others well below (lower risk). The spatial model will stabilize these estimates
# by borrowing strength from neighboring counties.

# ## The BYM model
#
# For each county i, we model:
# - Observed cases: Y_i ~ Poisson(E_i × exp(η_i))
# - Linear predictor: η_i = α + spatial_i + unstructured_i
#
# where:
# - E_i is the expected count (exposure)
# - α is the overall log-risk intercept
# - spatial_i is the Besag spatially structured effect
# - unstructured_i is the IID unstructured effect
#
# Let's set this up.
#
# For the Besag spatial model, we need to define which counties are neighbors.
# We'll construct a contiguity-based adjacency matrix from the county geometries.
# There is a helper method for this in GaussianMarkovRandomFields.jl.
using GaussianMarkovRandomFields, SparseArrays
geom_collection = LibGEOS.GeometryCollection(county_data.geometry)
W = contiguity_adjacency(geom_collection)

# Now we're ready to specify our model through the formula interface:
using StatsModels
using Distributions
using Latte

spatial = Besag(W, normalize_var = true)
unstructured = IID()
f = @formula(cases ~ 1 + spatial(county_id) + unstructured(county_id))

# ## Prior specification
#
# We use PC (Penalized Complexity) priors for the precision parameters.
# These priors prefer simpler models unless data strongly suggests otherwise:
hp_spec = @hyperparams begin
    (τ_besag ~ PCPrior.Precision(1.0, α = 0.01), transform = log)
    (τ_iid ~ PCPrior.Precision(1.0, α = 0.01), transform = log)
end

# This says: "I believe there's only a 1% chance that the standard deviation
# exceeds 1.0 on the log-risk scale" (roughly a 3-fold change in risk).

# ## Running INLA
#
# Now we're ready to run INLA! Note that we pass `exposure = :expected`
# to account for the varying expected counts:
inla_result = inla(
    f, hp_spec, county_data;
    family = Poisson,
    exposure = :expected,  # Use expected counts as exposure
    progress = false
)

# INLA has computed approximate posterior marginals for all parameters!

# ## Hyperparameter posteriors
#
# Let's examine the precision parameters:
summary_df(inla_result.hyperparameter_marginals)

# We can visualize these posterior distributions:
fig = Figure(size = (900, 400))
ax1 = Axis(
    fig[1, 1],
    xlabel = "Spatial precision (τ_besag)",
    ylabel = "Posterior density",
    title = "Spatial component precision"
)
ax2 = Axis(
    fig[1, 2],
    xlabel = "Unstructured precision (τ_iid)",
    ylabel = "Posterior density",
    title = "Unstructured component precision"
)
plot!(ax1, inla_result.hyperparameter_marginals.τ_besag)
plot!(ax2, inla_result.hyperparameter_marginals.τ_iid)
fig

# ## Visualizing the latent spatial and unstructured effects
#
# Before we look at relative risk, let's visualize the actual model components themselves.
# The BYM model decomposes each county's effect into two parts: a spatial component
# (shared with neighbors) and an unstructured component (county-specific). These are
# the base latent marginals - the raw building blocks before they're combined into
# the linear predictor.
#
# Let's make a joyplot showing how these effects vary across a selection of counties:
# We're going to extract components for selected counties.
sorted_idx = sortperm(county_data.SIR, rev = true)
component_indices = vcat(
    sorted_idx[1:3],              # 3 highest SIR
    sorted_idx[33:35],            # 3 middle SIR
    sorted_idx[(end - 2):end]
)        # 3 lowest SIR
n_counties = nrow(county_data)

spatial_dists = [inla_result.base_latent_marginals[i] for i in component_indices]
spatial_labels = [county_data.county[i] * " (spatial)" for i in component_indices]
iid_dists = [inla_result.base_latent_marginals[n_counties + i] for i in component_indices]
iid_labels = [county_data.county[i] * " (IID)" for i in component_indices]

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

# This decomposition is super informative! The spatial component shows smoothly varying
# effects that are shared between neighboring counties, while the IID component captures
# county-specific deviations. Notice how the spatial effects tend to be more constrained
# (tighter distributions) because they're regularized by the spatial structure, whereas
# the IID effects can vary more freely. This is exactly what the BYM model is designed
# to do - separate the smooth spatial signal from local noise.

# ## Posterior relative risk estimates
#
# Now the key results: posterior estimates of relative risk for each county!
obs_marginals = observation_marginals(inla_result)
risk_summary = summary_df(obs_marginals)
risk_summary.county = county_data.county
risk_summary.SIR = county_data.SIR
risk_summary.geometry = county_data.geometry
first(select(risk_summary, :county, :SIR, :median, :q2_5, :q97_5), 10)

# ## Comparing smoothed vs crude estimates
#
# The power of spatial modeling becomes clear when we compare smoothed posterior
# estimates with crude SIRs:
fig = Figure(size = (1000, 400))
ax1 = Axis(
    fig[1, 1],
    title = "Crude SIR",
    xlabel = "County",
    ylabel = "Risk ratio"
)
ax2 = Axis(
    fig[1, 2],
    title = "Posterior mean relative risk (BYM)",
    xlabel = "County",
    ylabel = "Risk ratio"
)

scatter!(
    ax1, 1:nrow(county_data), risk_summary.SIR,
    color = :gray60, markersize = 6
)
hlines!(ax1, [1.0], color = :black, linestyle = :dash, linewidth = 1)

rangebars!(
    ax2, 1:nrow(county_data),
    risk_summary.q2_5, risk_summary.q97_5,
    color = (:steelblue, 0.3), whiskerwidth = 0
)
scatter!(
    ax2, 1:nrow(county_data), risk_summary.median,
    color = :steelblue, markersize = 4
)
hlines!(ax2, [1.0], color = :black, linestyle = :dash, linewidth = 1)
fig

# The difference is striking! The posterior estimates are much more stable than the crude SIRs.
# Extreme values get shrunk toward 1.0 (the overall mean), neighboring counties end up with
# similar estimates due to spatial smoothing, and we get proper uncertainty quantification
# through the 95% credible intervals.

# ## Visualizing posterior distributions with joyplots
#
# Credible intervals are useful, but they only show two numbers. Let's visualize the full
# posterior distributions for selected counties using a joyplot. We'll pick the 5 highest
# and 5 lowest risk counties to see how the distributions differ:
sorted_by_risk = sortperm(risk_summary.median, rev = true)
selected_indices = vcat(
    sorted_by_risk[1:5],      # 5 highest risk
    sorted_by_risk[(end - 4):end]
) # 5 lowest risk

selected_dists = [obs_marginals[i] for i in selected_indices]
selected_labels = [county_data.county[i] for i in selected_indices]

fig_joy = joyplot(
    selected_dists;
    labels = selected_labels,
    title = "Relative Risk Distributions (Selected Counties)",
    xlabel = "Relative risk",
)
fig_joy

# You can see how the high-risk counties have distributions shifted to the right,
# while low-risk counties cluster near or below 1.0. The width of each distribution tells
# you about uncertainty - counties with smaller populations tend to have wider, more uncertain
# distributions.

# ## Identifying high-risk counties
#
# Which counties have significantly elevated risk? Those where the entire
# 95% credible interval is above 1.0:
high_risk = risk_summary[
    risk_summary.q2_5 .> 1.0,
    [:county, :median, :q2_5, :q97_5],
]

println("Counties with significantly elevated lung cancer risk:")
println(high_risk)

# Similarly for low-risk counties:
low_risk = risk_summary[
    risk_summary.q97_5 .< 1.0,
    [:county, :median, :q2_5, :q97_5],
]

println("\nCounties with significantly reduced lung cancer risk:")
println(low_risk)

# ## Exceedance probabilities
#
# For each county, we can compute the probability that relative risk exceeds
# a threshold (e.g., 1.1 for 10% elevated risk):
threshold = 1.1
risk_summary.exc_prob = [1 - cdf(observation_marginals(inla_result)[i], threshold) for i in 1:nrow(risk_summary)]
data(risk_summary) * mapping(:geometry, color = :exc_prob) * visual(Poly) |> draw

# Counties with high probability of elevated risk:
high_prob = findall(risk_summary.exc_prob .> 0.95)
println("\nCounties with >95% probability of RR > ", threshold, ":")
for i in high_prob
    println(
        "  ", county_data.county[i], ": P(RR > $threshold) = ",
        round(risk_summary.exc_prob[i], digits = 3)
    )
end

# ## Model diagnostics and comparison
#
# Let's examine model fit statistics:
println("\nBYM model fit:")
println("  DIC: ", round(inla_result.accumulators[1].DIC, digits = 2))
println("  Effective parameters (p_D): ", round(inla_result.accumulators[1].p_D, digits = 2))
println("  Log marginal likelihood: ", round(inla_result.accumulators[2].log_marginal_likelihood, digits = 2))

# ## Comparing with simpler models
#
# To appreciate the spatial component's value, let's fit a model with only
# unstructured random effects (no spatial structure):
f_iid = @formula(cases ~ 1 + unstructured(county_id))

hp_spec_iid = @hyperparams begin
    (τ_iid ~ PCPrior.Precision(1.0, α = 0.01), transform = log)
end

inla_result_iid = inla(
    f_iid, hp_spec_iid, county_data;
    family = Poisson,
    exposure = :expected,
    progress = false
)

# Compare model fit:
println("\nModel comparison:")
println("BYM (spatial + unstructured):")
println("  DIC: ", round(inla_result.accumulators[1].DIC, digits = 2))
println("  Log ML: ", round(inla_result.accumulators[2].log_marginal_likelihood, digits = 2))
println("\nIID only (no spatial structure):")
println("  DIC: ", round(inla_result_iid.accumulators[1].DIC, digits = 2))
println("  Log ML: ", round(inla_result_iid.accumulators[2].log_marginal_likelihood, digits = 2))

# The BYM model should have lower DIC (better fit) and higher marginal likelihood
# if spatial structure is present in the data.

# ## Summary
#
# And that's disease mapping with Latte.jl! We started with raw lung cancer data from
# Pennsylvania counties and used the BYM model to separate genuine spatial patterns
# from noise. The model gave us stable risk estimates by borrowing strength from
# neighboring counties, which is especially important when dealing with small populations.
#
# We saw how the spatial component captures smooth geographic variation while the
# unstructured component picks up county-specific quirks. The joyplots really brought
# this decomposition to life, showing how uncertainty varies across the risk spectrum.
# And by computing exceedance probabilities, we could make concrete statements about
# which counties are likely experiencing elevated risk - exactly what public health
# officials need for resource allocation.
#
# The BYM model shows up everywhere in spatial epidemiology, from disease surveillance
# to environmental health studies. Any time you're working with count data in small
# areas and suspect your neighbors matter, this is the model to reach for.
#
# Want to dig deeper? Check out [Getting started with Latte.jl](@ref)
# for the INLA basics, or explore how hyperparameter priors work in the main documentation.
