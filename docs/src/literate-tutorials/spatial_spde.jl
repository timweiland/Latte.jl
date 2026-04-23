# # Earthquake Intensity: Spatial Modelling with the SPDE Approach
#
# The **SPDE approach** (Lindgren, Rue & Lindström, 2011) is one of the most powerful
# features of INLA. It lets us model continuously indexed spatial random fields using
# Gaussian Markov random fields on a triangular mesh — combining the flexibility of
# geostatistical models with the computational efficiency of sparse precision matrices.
#
# In this tutorial we fit a **log-Gaussian Cox process** to real earthquake data:
# given the locations of M ≥ 4.5 earthquakes around Japan in 2023, we estimate the
# underlying seismic intensity surface. The Matérn spatial field discovers the plate
# boundary geometry — the Japan Trench, the Izu-Bonin arc, the Kuril arc — from
# data alone, without any tectonic information.
#
# You will learn:
# 1. How to download and prepare spatial point pattern data for INLA
# 2. How a Poisson observation model with a Matérn spatial field gives a **log-Gaussian
#    Cox process** — the standard model for spatial point patterns
# 3. How to evaluate the fitted Matérn field at new locations via `linear_combinations`
# 4. How to overlay results on a real coastline map

# ## Downloading earthquake data
#
# We query the [USGS Earthquake Catalog](https://earthquake.usgs.gov/) for all M ≥ 4.5
# earthquakes in the Japan region during 2023. The API returns a CSV that we read
# directly into a DataFrame.
using Downloads, CSV, DataFrames

usgs_url = "https://earthquake.usgs.gov/fdsnws/event/1/query?" *
    "format=csv&starttime=2023-01-01&endtime=2024-01-01&minmagnitude=4.5" *
    "&minlatitude=25&maxlatitude=50&minlongitude=125&maxlongitude=150"
eq = CSV.read(Downloads.download(usgs_url), DataFrame)
nrow(eq)

# ## Binning into a spatial grid
#
# A log-Gaussian Cox process models the *intensity* (expected events per unit area)
# as a function of space. To turn point locations into observations for INLA, we
# lay down a regular grid and count events per cell. Each cell becomes one Poisson
# observation with the cell area as exposure:
#
# ```math
# y_i \sim \text{Poisson}(a_i \cdot \lambda(s_i)), \qquad
# \log \lambda(s_i) = \alpha + u(s_i)
# ```
#
# where ``a_i`` is the cell area, ``\alpha`` is a global intercept, and ``u(s)``
# is a Matérn spatial field.
n_grid = 15
lat_range = (25.0, 50.0)
lon_range = (125.0, 150.0)
dlat = (lat_range[2] - lat_range[1]) / n_grid
dlon = (lon_range[2] - lon_range[1]) / n_grid

grid_lats = [lat_range[1] + (i - 0.5) * dlat for i in 1:n_grid]
grid_lons = [lon_range[1] + (j - 0.5) * dlon for j in 1:n_grid]

cell_counts = zeros(Int, n_grid, n_grid)
for row in eachrow(eq)
    i = clamp(ceil(Int, (row.latitude - lat_range[1]) / dlat), 1, n_grid)
    j = clamp(ceil(Int, (row.longitude - lon_range[1]) / dlon), 1, n_grid)
    cell_counts[i, j] += 1
end

df = DataFrame(
    lat = vec([lat for lat in grid_lats, _ in grid_lons]),
    lon = vec([lon for _ in grid_lats, lon in grid_lons]),
    count = vec(cell_counts),
    area = fill(dlat * dlon, n_grid^2),
)
first(df, 5)

# Let's visualise the raw counts on the grid:
using AlgebraOfGraphics, CairoMakie

fig = Figure(size = (600, 500))
ax = Axis(
    fig[1, 1]; title = "Earthquake counts per grid cell",
    xlabel = "Longitude", ylabel = "Latitude", aspect = DataAspect()
)
hm = heatmap!(ax, grid_lons, grid_lats, cell_counts'; colormap = :YlOrRd)
scatter!(ax, eq.longitude, eq.latitude; color = :black, markersize = 2)
Colorbar(fig[1, 2], hm; label = "Count")
fig

# The earthquakes cluster along narrow bands — these are the subduction zones
# where tectonic plates collide. Most of the domain has zero or very few events.
# This is exactly the kind of strongly structured spatial pattern that the SPDE
# approach handles well.

# ## Building the Matérn mesh
#
# The SPDE approach discretises the Matérn field on a triangular finite-element
# mesh. `MaternModel(points; smoothness = 1)` builds the mesh from a set of
# observation points (via their convex hull) and stores a projection matrix
# `A_obs` from mesh degrees of freedom to those points. At inference time the
# latent `field` lives on the mesh DOFs; at each observation, the linear
# predictor reads `field` through the projection.
using GaussianMarkovRandomFields

obs_points = hcat(df.lon, df.lat)   # N × 2 — lon first, then lat
base_matern = MaternModel(obs_points; smoothness = 1)
A_obs = evaluation_matrix(base_matern)
n_mesh = length(base_matern)

println("Mesh DOFs: ", n_mesh)
println("A_obs size: ", size(A_obs))

# ## The model
#
# The Matérn field has two hyperparameters: the field precision `τ_matern`
# (controlling amplitude) and `range_matern` (controlling spatial correlation
# distance). We use a PC prior for precision and an `Exponential` prior for range.
using Latte
using DynamicPPL: @model
using Distributions
using LinearAlgebra

@model function spde_model(counts, area, base_matern, A_obs)
    τ_matern ~ PCPrior.Precision(1.0, α = 0.01)
    range_matern ~ Exponential(5.0)
    β ~ MvNormal(zeros(1), 100.0 * I(1))
    field ~ base_matern(τ = τ_matern, range = range_matern)
    η = β[1] .+ A_obs * field
    for i in eachindex(counts)
        counts[i] ~ Poisson(area[i] * exp(η[i]); check_args = false)
    end
end

# ## Running INLA
#
# We use `SimplifiedLaplace` for the latent marginals — it adds a skewness
# correction over the basic Gaussian approximation, which matters for count data
# where the posterior can be asymmetric (especially at cells with zero counts).
lgm = latte_from_dppl(
    spde_model(df.count, df.area, base_matern, A_obs);
    random = (:β, :field),
)
result = inla(
    lgm, df.count;
    progress = false,
    latent_marginalization_method = SimplifiedLaplace(),
)

# ## Hyperparameter posteriors
#
# Let's see what the model learned about the spatial structure.
hp_df = summary_df(result.hyperparameter_marginals)
hp_df

# The estimated spatial range tells us the characteristic scale of seismic
# intensity variation — how far the influence of a plate boundary extends.

fig = Figure(size = (700, 300))
for (i, (name, label)) in enumerate(
        zip([:τ_matern, :range_matern], ["Field precision τ", "Spatial range (degrees)"])
    )
    local ax = Axis(fig[1, i]; title = label, xlabel = "value", ylabel = "density")
    d = result.hyperparameter_marginals[name]
    μ, s = mean(d), std(d)
    xs = range(max(1.0e-3, μ - 4s), μ + 4s; length = 200)
    lines!(ax, xs, pdf.(d, xs); color = :steelblue, linewidth = 2)
end
fig

# ## Predicting the intensity surface
#
# To evaluate the fitted Matérn field on a fine prediction grid, we build a
# projection matrix `A_pred` from the same FEM discretization to the new
# locations, and ask INLA for the marginal of `β + A_pred_row · field` at
# each prediction point. `linear_combinations(result; kwargs...)` does the
# plumbing for us — it looks up each symbol's range in the augmented latent
# and pads the η positions with zeros.
n_fine = 50
fine_lats = range(lat_range[1], lat_range[2]; length = n_fine)
fine_lons = range(lon_range[1], lon_range[2]; length = n_fine)

pred_points = hcat(
    vec([lon for _ in fine_lats, lon in fine_lons]),
    vec([lat for lat in fine_lats, _ in fine_lons]),
)
A_pred = evaluation_matrix(base_matern.discretization, pred_points)

pred_marginals = linear_combinations(result; β = 1.0, field = A_pred)

# The predictions are on the linear predictor scale (log-intensity).
# We exponentiate to get the actual intensity (expected events per degree²).
pred_means = mean.(pred_marginals)
intensity = reshape(exp.(pred_means), n_fine, n_fine)'

# ## Visualising the intensity surface
#
# We overlay the predicted intensity on a coastline map to see how the model's
# spatial structure aligns with real tectonic features. The coastline data comes
# from [Natural Earth](https://www.naturalearthdata.com/) (public domain, ~137 KB).
coast_url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/" *
    "master/geojson/ne_110m_coastline.geojson"
coast_str = read(Downloads.download(coast_url), String)

function extract_coastline_segments(geojson_str, lon_range, lat_range)
    segments = Vector{Tuple{Vector{Float64}, Vector{Float64}}}()
    for m in eachmatch(r"\[\[[-\d.]+,[-\d.]+\](?:,\[[-\d.]+,[-\d.]+\])*\]", geojson_str)
        lons, lats = Float64[], Float64[]
        for coord in eachmatch(r"\[([-\d.]+),([-\d.]+)\]", m.match)
            lon = parse(Float64, coord.captures[1])
            lat = parse(Float64, coord.captures[2])
            if lon_range[1] - 5 <= lon <= lon_range[2] + 5 &&
                    lat_range[1] - 5 <= lat <= lat_range[2] + 5
                push!(lons, lon)
                push!(lats, lat)
            else
                length(lons) > 1 && push!(segments, (copy(lons), copy(lats)))
                empty!(lons); empty!(lats)
            end
        end
        length(lons) > 1 && push!(segments, (copy(lons), copy(lats)))
    end
    return segments
end
coastlines = extract_coastline_segments(coast_str, lon_range, lat_range)

#

fig = Figure(size = (700, 600))
ax = Axis(
    fig[1, 1]; title = "Predicted seismic intensity — Japan, 2023 (M ≥ 4.5)",
    xlabel = "Longitude", ylabel = "Latitude", aspect = DataAspect()
)
hm = heatmap!(
    ax, collect(fine_lons), collect(fine_lats), intensity;
    colormap = :inferno, colorscale = log10
)
for (lons, lats) in coastlines
    lines!(ax, lons, lats; color = :white, linewidth = 1.5)
end
scatter!(
    ax, eq.longitude, eq.latitude;
    color = :cyan, markersize = 3, strokewidth = 0.5, strokecolor = :black
)
xlims!(ax, lon_range...)
ylims!(ax, lat_range...)
Colorbar(fig[1, 2], hm; label = "Events per deg²")
fig

# The model discovers the major tectonic features from earthquake counts alone:
#
# - **Japan Trench** (140–145°E, 35–43°N): the main subduction zone where the
#   Pacific Plate dives under northeastern Japan
# - **Izu-Bonin arc** (~140°E, 28–33°N): a highly active volcanic arc extending
#   south from Tokyo
# - **Kuril arc** (~148°E, 44°N): the northern continuation of the subduction zone
# - **Ryukyu arc** (~126–128°E, 25–30°N): the southwestern island chain
# - **Sea of Japan interior**: correctly identified as a low-intensity region
#
# The Matérn field smoothly interpolates between the observed grid cells, borrowing
# spatial information to estimate intensity even where no earthquakes occurred.

# ## Posterior uncertainty
#
# One of INLA's strengths is that we get full posterior uncertainty, not just
# point estimates. Let's visualise the posterior standard deviation of the
# intensity field:
pred_sds = std.(pred_marginals)
sd_grid = reshape(pred_sds, n_fine, n_fine)'

fig = Figure(size = (700, 600))
ax = Axis(
    fig[1, 1]; title = "Posterior uncertainty (std. dev. of log-intensity)",
    xlabel = "Longitude", ylabel = "Latitude", aspect = DataAspect()
)
hm = heatmap!(ax, collect(fine_lons), collect(fine_lats), sd_grid; colormap = :YlOrRd)
for (lons, lats) in coastlines
    lines!(ax, lons, lats; color = :black, linewidth = 1)
end
xlims!(ax, lon_range...)
ylims!(ax, lat_range...)
Colorbar(fig[1, 2], hm; label = "Posterior SD")
fig

# Uncertainty is lowest in areas with many observations (the active subduction
# zones) and highest in data-sparse regions (open ocean, continental interior).

# ## Model diagnostics
println("Model fit:")
println("  DIC:  $(round(result.accumulators[1].DIC, digits = 1))")
println("  WAIC: $(round(result.accumulators[3].WAIC, digits = 1))")
println("  Log marginal likelihood: $(round(result.exploration.log_normalization_constant, digits = 1))")

# ## Summary
#
# In this tutorial we used INLA with the SPDE approach to fit a **log-Gaussian
# Cox process** to real earthquake data. Starting from ~600 epicentre locations,
# we estimated a continuous seismic intensity surface that recovers the geometry
# of major tectonic plate boundaries — without any geological prior knowledge.
#
# Key takeaways:
#
# - `MaternModel(points; smoothness = ν)` builds the FEM mesh, the SPDE
#   discretisation and the projection matrix in one step; `evaluation_matrix`
#   is the explicit, reusable design-matrix operator
# - A **Poisson family with exposure** turns gridded counts into a proper
#   log-Gaussian Cox process — the exposure is just a log-offset in the
#   linear predictor
# - `linear_combinations(result; β = 1.0, field = A_pred)` evaluates the
#   fitted spatial field at arbitrary new locations: each keyword names a
#   random-effect block in the DPPL model and supplies its coefficients,
#   and the posterior marginals of the linear combination come back
# - INLA provides full **posterior uncertainty** on both the intensity surface
#   and the hyperparameters (spatial range and field precision)
#
# ## References
#
# - Lindgren, F., Rue, H. & Lindström, J. (2011). An explicit link between Gaussian
#   fields and Gaussian Markov random fields: the stochastic partial differential
#   equation approach. *JRSS-B*, 73(4), 423–498.
# - USGS Earthquake Hazards Program. [earthquake.usgs.gov](https://earthquake.usgs.gov/)
# - Natural Earth. [naturalearthdata.com](https://www.naturalearthdata.com/)
