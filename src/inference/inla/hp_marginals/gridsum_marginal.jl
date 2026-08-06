"""
GridSumMarginal: rectangle rule on exploration grid -> SplineMarginalDistribution.

For D=1, the exploration grid directly gives us 1D density evaluations.
Sort by θ (working space), then feed to `_build_spline_marginal`.
"""
function _marginalize_impl(
        method::GridSumMarginal,
        exploration::AbstractHyperparameterExploration,
        model::LatentGaussianModel,
        y,
        progress_callback
    )
    if progress_callback === nothing
        progress_callback = (; kwargs...) -> nothing
    end

    progress_callback(status = "Computing GridSum spline marginals")

    spec = exploration.transform.θ_star.spec
    n_dim = length(exploration.transform.θ_star)
    # One marginal per flat coordinate; vector blocks expand to `name[i]`.
    param_names = _expanded_hp_names(spec)

    # Extract (θ, log_density) from grid points
    grid_points = exploration.grid_points

    marginals = map(1:n_dim) do d
        # Get values for this dimension
        θ_vals = [p.θ[d] for p in grid_points]
        log_densities = [p.log_density for p in grid_points]

        # Sort by θ value in working space
        perm = sortperm(θ_vals)
        η_grid = θ_vals[perm]
        log_marginal = log_densities[perm]

        # Remove duplicates (can happen if mode is repeated in grid).
        unique_mask = Bool[
            i == 1 || η_grid[i] != η_grid[i - 1] for i in 1:length(η_grid)
        ]
        η_grid = η_grid[unique_mask]
        log_marginal = log_marginal[unique_mask]

        _build_spline_marginal(η_grid, log_marginal, spec, d)
    end

    progress_callback(status = "GridSum marginals complete")

    return NamedTuple(param_names[i] => marginals[i] for i in 1:n_dim)
end
