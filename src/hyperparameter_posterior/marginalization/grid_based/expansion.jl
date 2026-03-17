"""
Asymmetric expansion logic for grid-based hyperparameter marginalization.

Implements efficient region extension that only evaluates new points beyond current boundaries.
"""

using Printf

export extend_exploration_asymmetric, update_log_drop_limits

"""
    update_log_drop_limits(current_limits, diagnostics, increment, cap, allow_asymmetric)

Update log-drop limits based on diagnostic results.

# Arguments
- `current_limits::AsymmetricLogDropLimits`: Current limits
- `diagnostics::Vector{MarginalDiagnostics}`: Diagnostic results for each dimension
- `increment::Float64`: How much to increase when expanding
- `cap::Float64`: Maximum allowed limit
- `allow_asymmetric::Bool`: If false, use symmetric limits

# Returns
New `AsymmetricLogDropLimits` with updated values where diagnostics indicated issues.
"""
function update_log_drop_limits(
        current_limits::AsymmetricLogDropLimits,
        diagnostics::Vector{MarginalDiagnostics},
        increment::Float64,
        cap::Float64,
        allow_asymmetric::Bool
    )
    n_dim = size(current_limits.limits, 1)
    new_limits = copy(current_limits.limits)

    for diag in diagnostics
        dim = diag.dimension

        # Update left (negative direction) limit if needed
        if !diag.left_tail_ok
            suggested_increase = max(increment, diag.suggested_left_extension)
            new_limits[dim, 1] = min(new_limits[dim, 1] + suggested_increase, cap)

            if new_limits[dim, 1] >= cap
                @warn "Dimension $dim left tail: reached max_log_drop_cap=$cap. " *
                    "Tail mass estimate: $(@sprintf("%.2e", diag.left_tail_mass)). " *
                    "Consider increasing cap or check model specification."
            end
        end

        # Update right (positive direction) limit if needed
        if !diag.right_tail_ok
            suggested_increase = max(increment, diag.suggested_right_extension)
            new_limits[dim, 2] = min(new_limits[dim, 2] + suggested_increase, cap)

            if new_limits[dim, 2] >= cap
                @warn "Dimension $dim right tail: reached max_log_drop_cap=$cap. " *
                    "Tail mass estimate: $(@sprintf("%.2e", diag.right_tail_mass)). " *
                    "Consider increasing cap or check model specification."
            end
        end

        # If not allowing asymmetric, take the max of left and right
        if !allow_asymmetric
            max_limit = max(new_limits[dim, 1], new_limits[dim, 2])
            new_limits[dim, 1] = max_limit
            new_limits[dim, 2] = max_limit
        end
    end

    return AsymmetricLogDropLimits(new_limits)
end

"""
    extend_exploration_asymmetric(exploration, model, y, new_limits, progress_callback)

Extend the exploration region according to new asymmetric limits.

This function only evaluates NEW points beyond the current boundaries.
All existing grid points are reused to avoid redundant computation.

# Algorithm
1. For each dimension and direction (negative/positive):
   - Check if new limit exceeds old limit
   - If yes, explore the extended region in that direction
   - Add new points to the exploration
2. Merge new points with existing exploration

# Arguments
- `exploration::AbstractHyperparameterExploration`: Current exploration
- `model::INLAModel`: Model specification
- `y`: Observed data
- `new_limits::AsymmetricLogDropLimits`: Updated exploration limits
- `progress_callback`: Progress tracking

# Returns
New `HyperparameterExploration` with extended grid.
"""
function extend_exploration_asymmetric(
        exploration::AbstractHyperparameterExploration,
        model::INLAModel,
        y,
        new_limits::AsymmetricLogDropLimits,
        progress_callback
    )
    # Extract current limits from exploration
    # (Inferred from the furthest points in each direction)
    current_limits = infer_current_limits(exploration)

    n_dim = length(exploration.transform.θ_star)
    transform = exploration.transform

    # Collect new points to add
    new_grid_points = GridPoint[]

    # Check each dimension and direction
    for dim in 1:n_dim
        for direction in [-1, 1]
            dir_idx = direction == -1 ? 1 : 2  # 1=negative, 2=positive

            new_limit = new_limits.limits[dim, dir_idx]
            old_limit = current_limits.limits[dim, dir_idx]

            if new_limit > old_limit
                # Need to extend in this direction
                progress_callback(
                    status = "Extending dimension $dim, direction $(direction > 0 ? "+" : "-")",
                    old_limit = old_limit,
                    new_limit = new_limit
                )

                # Explore extended region
                extended_points = explore_extended_half_axis(
                    exploration,
                    model,
                    y,
                    dim,
                    direction,
                    old_limit,
                    new_limit
                )

                append!(new_grid_points, extended_points)
            end
        end
    end

    if isempty(new_grid_points)
        # No extension needed
        return exploration
    end

    # Merge new points with existing exploration
    return merge_with_new_points(exploration, new_grid_points)
end

"""
    infer_current_limits(exploration)

Infer the current max_log_drop limits from the exploration grid.

Finds the furthest explored point in each dimension/direction and computes
the log-density drop from the mode.
"""
function infer_current_limits(exploration::AbstractHyperparameterExploration)
    θ_star = exploration.transform.θ_star
    n_dim = length(θ_star)

    # Find mode log-density
    mode_log_density = maximum(p.log_density for p in exploration.grid_points)

    limits = zeros(n_dim, 2)

    for dim in 1:n_dim
        # Find furthest point in negative direction (θ < θ_star)
        neg_points = filter(p -> p.θ[dim] < θ_star[dim], exploration.grid_points)
        if !isempty(neg_points)
            furthest_neg = argmin(p.θ[dim] for p in neg_points)
            limits[dim, 1] = mode_log_density - neg_points[furthest_neg].log_density
        else
            limits[dim, 1] = 0.0
        end

        # Find furthest point in positive direction (θ > θ_star)
        pos_points = filter(p -> p.θ[dim] > θ_star[dim], exploration.grid_points)
        if !isempty(pos_points)
            furthest_pos = argmax(p.θ[dim] for p in pos_points)
            limits[dim, 2] = mode_log_density - pos_points[furthest_pos].log_density
        else
            limits[dim, 2] = 0.0
        end
    end

    return AsymmetricLogDropLimits(limits)
end

"""
    explore_extended_half_axis(exploration, model, y, dim, direction, old_limit, new_limit)

Explore the region beyond the current boundary in one dimension/direction.

Only evaluates points in the extended region (between old_limit and new_limit).

# Arguments
- `exploration`: Current exploration
- `model`, `y`: Model and data
- `dim`: Which dimension to extend
- `direction`: -1 for negative, +1 for positive
- `old_limit`: Current max_log_drop in this direction
- `new_limit`: New max_log_drop to extend to

# Returns
Vector of new `GridPoint` objects in the extended region.
"""
function explore_extended_half_axis(
        exploration::AbstractHyperparameterExploration,
        model::INLAModel,
        y,
        dim::Int,
        direction::Int,
        old_limit::Float64,
        new_limit::Float64
    )
    transform = exploration.transform
    θ_star = transform.θ_star
    n_dim = length(θ_star)

    # Determine step size from existing grid (if possible)
    # For simplicity, use a fixed step in z-space
    # In practice, this should match the integration_step_z from exploration
    evaluation_step_z = 0.5  # Default; could be inferred from existing grid

    mode_log_density = maximum(p.log_density for p in exploration.grid_points)

    new_points = GridPoint[]

    # Start from just beyond the old limit and step outward
    # We need to find the z-coordinate corresponding to old_limit and new_limit

    # Strategy: Step outward in z-space until log-drop exceeds new_limit
    step_count = 1
    while true
        # Construct z-vector: all zeros except dimension `dim`
        z_vec = zeros(n_dim)
        z_vec[dim] = direction * (step_count * evaluation_step_z)

        # Transform to θ-space
        θ_test = transform(z_vec)

        # Evaluate log-density
        log_density = hyperparameter_logpdf(model, θ_test, y) - exploration.log_normalization_constant

        log_drop = mode_log_density - log_density

        # Check if we've reached the new limit
        if log_drop > new_limit
            break
        end

        # Check if this point is in the extended region (beyond old limit)
        if log_drop > old_limit
            # This is a new point - add it
            push!(new_points, GridPoint(θ_test, log_density, nothing))
        end

        step_count += 1

        # Safety check: don't go too far
        if step_count > 1000
            @warn "Extension reached maximum step count (1000) for dim=$dim, direction=$direction"
            break
        end
    end

    return new_points
end

"""
    merge_with_new_points(exploration, new_points)

Merge new grid points with existing exploration.

Creates a new `HyperparameterExploration` object with the combined grid.
Updates integration indices to include the new points for hyperparameter marginal computation.

Note: This does not recompute the normalization constant, which is acceptable
since we're only using this for interpolation in the marginalization step.
"""
function merge_with_new_points(exploration::AbstractHyperparameterExploration, new_points::Vector{GridPoint})
    # Combine grid points
    combined_points = vcat(exploration.grid_points, new_points)

    # Update integration indices to include the new points
    # New points are added at the end, starting from length(exploration.grid_points) + 1
    n_old_points = length(exploration.grid_points)
    n_new_points = length(new_points)
    new_indices = collect((n_old_points + 1):(n_old_points + n_new_points))

    # Combine old and new integration indices
    combined_integration_indices = vcat(exploration.integration_indices, new_indices)

    return GridExploration(
        combined_points,
        combined_integration_indices,
        exploration.transform,
        exploration.log_normalization_constant  # Keep existing (approximate)
    )
end
