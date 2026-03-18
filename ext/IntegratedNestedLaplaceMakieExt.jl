module IntegratedNestedLaplaceMakieExt

using IntegratedNestedLaplace
using Makie
using Distributions

# Import the distribution types we're adding recipes for
import IntegratedNestedLaplace: SplineMarginalDistribution,
    WeightedMixture,
    TransformedWeightedMixture,
    joyplot,
    joyplot!

# Convenience union type for all supported distributions
const PlottableINLADistribution = Union{
    <:SplineMarginalDistribution,
    <:WeightedMixture,
    <:TransformedWeightedMixture,
}

# ==================== Recipe for Plotting INLA Marginal Distributions ====================

"""
    distplot(distribution; kwargs...)
    distplot!(ax, distribution; kwargs...)

Plot INLA marginal distributions with optional credible interval shading.

Supports:
- `SplineMarginalDistribution` (hyperparameter marginals)
- `WeightedMixture` (latent field marginals)
- `TransformedWeightedMixture` (observation marginals)

# Arguments
- `distribution`: An INLA marginal distribution to plot (SplineMarginalDistribution, WeightedMixture, or TransformedWeightedMixture)

# Attributes
- `quantile_range = (0.001, 0.999)`: Initial quantile range for x-axis (refined for heavy tails)
- `n_points = 200`: Number of points for smooth curve
- `max_log_drop = 5.0`: Maximum log-density drop from mode for plot range (handles heavy tails)
- `credible_interval = 0.95`: Credible interval level for shading (set to `nothing` to disable)
- `ci_color = :gray`: Color for credible interval shading
- `ci_alpha = 0.2`: Transparency for credible interval
- `show_mode = false`: Show vertical line at mode
- `mode_color = :red`: Color for mode line
- `mode_linestyle = :dash`: Line style for mode indicator
- `show_median = false`: Show vertical line at median
- `median_color = :orange`: Color for median line
- `median_linestyle = :dash`: Line style for median indicator
- `add_special_xticks = true`: Add x-axis ticks at mode/median when shown
- Standard line attributes: `color`, `linewidth`, `linestyle`, etc.

# Examples
```julia
using IntegratedNestedLaplace
using CairoMakie

result = inla(...)
τ_dist = result.hyperparameter_marginals.τ_besag

# Basic plot with credible interval
distplot(τ_dist)

# Customized appearance
distplot(τ_dist;
    quantile_range = (0.0001, 0.9999),
    credible_interval = 0.95,
    ci_alpha = 0.3,
    color = :steelblue,
    linewidth = 2,
    show_mode = true
)

# Multiple distributions
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "Parameter value", ylabel = "Density")
distplot!(ax, result.hyperparameter_marginals.τ_besag; label = "τ_besag", color = :steelblue)
distplot!(ax, result.hyperparameter_marginals.τ_iid; label = "τ_iid", color = :coral)
axislegend(ax)
fig
```
"""
@recipe(DistPlot) do scene
    Makie.Theme(
        # Data generation
        quantile_range = (0.001, 0.999),
        n_points = 200,
        max_log_drop = 6.0,  # Maximum log-density drop from mode for plot range

        # Credible interval
        credible_interval = 0.95,  # Set to nothing to disable
        ci_color = :gray,
        ci_alpha = 0.2,

        # Add xticks at special values
        add_special_xticks = true,

        # Fill options
        fill = false,        # Fill area under curve
        fillalpha = 0.6,     # Opacity for filled area

        # Line appearance (standard attributes)
        color = :black,
        linewidth = 2,
        linestyle = :solid
    )
end

# Define convert_arguments to pass through the distribution unchanged
function Makie.convert_arguments(::Type{<:DistPlot}, d::SplineMarginalDistribution)
    return (d,)
end

function Makie.convert_arguments(::Type{<:DistPlot}, d::WeightedMixture)
    return (d,)
end

function Makie.convert_arguments(::Type{<:DistPlot}, d::TransformedWeightedMixture)
    return (d,)
end

function Makie.plot!(plot::DistPlot{<:Tuple{<:PlottableINLADistribution}})
    # Extract the distribution from first argument (it's an Observable)
    d = plot[1][]

    # Get attributes
    quantile_range = plot[:quantile_range][]
    n_points = plot[:n_points][]
    credible_interval = plot[:credible_interval][]

    # Start with quantile-based range
    q_min, q_max = quantile_range
    x_lower = quantile(d, q_min)
    x_upper = quantile(d, q_max)

    # Refine based on log-density drop from mode for heavy-tailed distributions
    mode_val = mode(d)
    log_mode = logpdf(d, mode_val)
    max_log_drop = plot[:max_log_drop][]  # Only show region within this many log units of mode

    # Check if lower quantile is too far in the tail
    if log_mode - logpdf(d, x_lower) > max_log_drop
        # Binary search for better lower bound
        search_lower = x_lower
        search_upper = mode_val
        for _ in 1:20
            x_test = (search_lower + search_upper) / 2
            if abs(log_mode - logpdf(d, x_test)) > max_log_drop
                search_lower = x_test
            else
                search_upper = x_test
            end
        end
        x_lower = search_lower
    end

    # Check if upper quantile is too far in the tail
    if log_mode - logpdf(d, x_upper) > max_log_drop
        # Binary search for better upper bound
        search_lower = mode_val
        search_upper = x_upper
        for _ in 1:20
            x_test = (search_lower + search_upper) / 2
            if abs(log_mode - logpdf(d, x_test)) > max_log_drop
                search_upper = x_test
            else
                search_lower = x_test
            end
        end
        x_upper = search_upper
    end

    # Generate x-values in the refined range
    x = range(x_lower, x_upper; length = n_points)

    # Compute PDF at those points
    y = pdf.(Ref(d), x)

    # Plot credible interval shading if enabled
    if !isnothing(credible_interval) && 0 < credible_interval < 1
        # Compute credible interval bounds
        α = 1 - credible_interval
        ci_lower = quantile(d, α / 2)
        ci_upper = quantile(d, 1 - α / 2)

        # Find indices in the x range that fall within CI
        ci_mask = ci_lower .<= x .<= ci_upper

        if any(ci_mask)
            x_ci = x[ci_mask]
            y_ci = y[ci_mask]

            # Create shaded region using band!
            # band! requires upper and lower bounds, so we use 0 as lower
            band!(
                plot,
                x_ci,
                zeros(length(x_ci)),
                y_ci;
                color = (plot[:ci_color][], plot[:ci_alpha][])
            )
        end
    end

    # Plot the main PDF curve - either as filled band or line
    fill_enabled = plot[:fill][]

    if fill_enabled
        # Create filled area under curve
        band_attrs = Dict{Symbol, Any}(
            :color => (plot[:color][], plot[:fillalpha][])
        )

        # Forward label if it exists
        if haskey(plot.attributes, :label)
            band_attrs[:label] = plot[:label][]
        end

        band!(plot, x, zeros(length(x)), y; band_attrs...)

        # Add outline with line
        line_attrs = Dict{Symbol, Any}(
            :color => plot[:color][],
            :linewidth => plot[:linewidth][],
            :linestyle => plot[:linestyle][]
        )
        lines!(plot, x, y; line_attrs...)
    else
        # Just draw line
        line_attrs = Dict{Symbol, Any}(
            :color => plot[:color][],
            :linewidth => plot[:linewidth][],
            :linestyle => plot[:linestyle][]
        )

        # Forward label if it exists
        if haskey(plot.attributes, :label)
            line_attrs[:label] = plot[:label][]
        end

        lines!(plot, x, y; line_attrs...)
    end

    return plot
end

# ==================== Make `plot()` work nicely ====================

# When users call plot(distribution), use the enhanced distplot by default
function Makie.plottype(::SplineMarginalDistribution)
    return DistPlot
end

function Makie.plottype(::WeightedMixture)
    return DistPlot
end

function Makie.plottype(::TransformedWeightedMixture)
    return DistPlot
end

# ==================== Joy Plot (Ridgeline Plot) ====================

"""
    joyplot(distributions::Vector; kwargs...)
    joyplot!(ax, distributions::Vector; kwargs...)

Create a joy plot (ridgeline plot) for multiple INLA marginal distributions, inspired
by the famous Joy Division "Unknown Pleasures" album cover.

# Arguments
- `distributions::Vector`: Vector of INLA distributions (SplineMarginalDistribution, WeightedMixture, or TransformedWeightedMixture)

# Attributes
- `labels::Union{Nothing, Vector{String}} = nothing`: Labels for each distribution (shown on y-axis)
- `spacing::Float64 = 1.0`: Vertical spacing between distributions
- `colors::Union{Nothing, Vector} = nothing`: Vector of colors for each distribution (default: Wong colors)
- `strokewidth = 1`: Width of the outline stroke
- `fill = true`: Fill area under curves (default true for joyplots)
- `fillalpha = 0.6`: Opacity for filled areas (0 = transparent, 1 = opaque)
- `reverse_order = true`: Plot in reverse order (bottom to top, like Joy Division cover)
- `credible_interval = nothing`: Disable CI shading by default for cleaner joyplots
- Any other attributes from `distplot` (quantile_range, n_points, max_log_drop, etc.)

# Examples
```julia
using IntegratedNestedLaplace
using CairoMakie

result = inla(...)

# Joy plot of latent marginals at different locations
indices = 10:10:100
dists = [result.latent_marginals[i] for i in indices]
labels = ["Location \$i" for i in indices]
joyplot(dists; labels = labels)

# Custom styling with specific colors
using Colors
my_colors = [colorant"steelblue", colorant"coral", colorant"purple"]
joyplot(dists;
    labels = labels,
    spacing = 1.5,
    colors = my_colors,
    strokewidth = 2
)
```
"""
function joyplot(
        distributions::AbstractVector{<:PlottableINLADistribution};
        labels::Union{Nothing, Vector{String}} = nothing,
        spacing::Float64 = 1.0,
        title::String = "",
        xlabel::String = "",
        ylabel::String = "",
        kwargs...
    )
    f = Figure()
    ax = Axis(f[1, 1], title = title, xlabel = xlabel, ylabel = ylabel)
    joyplot!(ax, distributions; labels = labels, spacing = spacing, kwargs...)
    return f
end

function joyplot!(
        ax::Axis,
        distributions::AbstractVector{<:PlottableINLADistribution};
        labels::Union{Nothing, Vector{String}} = nothing,
        spacing::Float64 = 1.0,
        colors::Union{Nothing, Vector} = nothing,
        strokewidth = 1,
        fill = true,
        fillalpha = 0.6,
        reverse_order = true,
        credible_interval = nothing,  # Disable CI by default for cleaner joyplots
        kwargs...
    )
    n_dists = length(distributions)

    if n_dists == 0
        error("Cannot create joy plot with empty distribution vector")
    end

    # Set up labels
    if labels === nothing
        labels = ["Distribution $i" for i in 1:n_dists]
    elseif length(labels) != n_dists
        error("Number of labels ($(length(labels))) must match number of distributions ($n_dists)")
    end

    # Set up colors (default to a gradient)
    if colors === nothing
        # Create a color gradient from the current theme
        colors = [Makie.wong_colors()[mod1(i, 7)] for i in 1:n_dists]
    elseif length(colors) != n_dists
        error("Number of colors ($(length(colors))) must match number of distributions ($n_dists)")
    end

    # Determine plotting order
    plot_order = reverse_order ? (n_dists:-1:1) : (1:n_dists)

    # Plot each distribution with vertical offset using our existing recipe
    for (plot_idx, i) in enumerate(plot_order)
        d = distributions[i]
        offset = (plot_idx - 1) * spacing

        # Use our existing DistPlot recipe with offset transformation and fill options
        p = plot!(
            ax, d;
            credible_interval = credible_interval,
            color = colors[i],
            linewidth = strokewidth,
            fill = fill,
            fillalpha = fillalpha,
            kwargs...
        )

        # Apply vertical offset by transforming the plot
        translate!(p, 0, offset, -0.1 * plot_idx)
    end

    # Set up y-axis ticks
    ytick_positions = [(i - 1) * spacing for i in 1:n_dists]
    if reverse_order
        ytick_labels = reverse(labels)
    else
        ytick_labels = labels
    end
    ax.yticks = (ytick_positions, ytick_labels)

    return ax
end

end # module
