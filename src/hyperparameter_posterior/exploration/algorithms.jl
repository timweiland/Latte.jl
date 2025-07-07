export explore_hyperparameter_posterior, explore_half_axis_by_steps, explore_dimension_and_build_lookup

"""
    explore_half_axis_by_steps(model, y, transform, mode_logpdf, dim, direction, evaluation_step_z, max_log_drop, interpolation_subdivisions, marginalization_method, marginalization_indices)

Explores one direction along a dimension using integer step indices.
Returns a list of (key, GridPoint) tuples where key is an integer coordinate tuple.
"""
function explore_half_axis_by_steps(
        model, y, transform::ReparameterizationTransform, mode_logpdf,
        dim::Int, direction::Int, evaluation_step_z::Float64, max_log_drop::Float64, interpolation_subdivisions::Int,
        marginalization_method, marginalization_indices
    )
    n_dim = length(transform.θ_star)
    keyed_points = Tuple{NTuple{n_dim, Int}, GridPoint}[]

    step_count = 1
    while true
        key_vec = zeros(Int, n_dim)
        key_vec[dim] = direction * step_count
        key = Tuple(key_vec)

        z_vec = evaluation_step_z .* collect(key)
        θ_test = transform(z_vec)

        is_integration_point = (step_count % interpolation_subdivisions == 0)

        log_density, marginal_result = evaluate_logpdf_and_marginals(
            model, y, θ_test;
            compute_marginals = is_integration_point,
            marginalization_method = marginalization_method,
            marginalization_indices = marginalization_indices
        )

        if mode_logpdf - log_density > max_log_drop
            break
        end

        point = GridPoint(θ_test, log_density, marginal_result)
        push!(keyed_points, (key, point))
        step_count += 1
    end

    return keyed_points
end

"""
    explore_dimension_and_build_lookup(model, y, transform, mode_logpdf, dim, evaluation_step_z, max_log_drop, interpolation_subdivisions, marginalization_method, marginalization_indices)

Explore a single dimension and build lookup table for integer-based grid coordinates.
Calls explore_half_axis_by_steps for both directions and collects results.

Returns (point_lookup, step_range) where:
- point_lookup: Dictionary mapping integer tuple keys to GridPoint objects
- step_range: Range of integer steps found for this dimension
"""
function explore_dimension_and_build_lookup(
        model, y, transform::ReparameterizationTransform, mode_logpdf,
        dim::Int, evaluation_step_z::Float64, max_log_drop::Float64, interpolation_subdivisions::Int,
        marginalization_method, marginalization_indices
    )
    # Call our helper function for both directions
    pos_points = explore_half_axis_by_steps(
        model, y, transform, mode_logpdf,
        dim, 1, evaluation_step_z, max_log_drop, interpolation_subdivisions,
        marginalization_method, marginalization_indices
    )
    neg_points = explore_half_axis_by_steps(
        model, y, transform, mode_logpdf,
        dim, -1, evaluation_step_z, max_log_drop, interpolation_subdivisions,
        marginalization_method, marginalization_indices
    )

    # Return a dictionary for fast, safe lookups
    point_lookup = Dict(vcat(pos_points, neg_points))

    # And the range of integer steps it found for this dimension
    step_indices = [p[1][dim] for p in vcat(pos_points, neg_points)]
    push!(step_indices, 0) # Include the mode
    step_range = minimum(step_indices):maximum(step_indices)

    return point_lookup, step_range
end

"""
    explore_hyperparameter_posterior(model, y, θ_star, marginalization_method, marginalization_indices; kwargs...)

Explore the hyperparameter posterior around the mode `θ_star` using a robust,
integer-based grid construction method.

# Keyword Arguments
- `integration_step_z::Float64 = 1.0`: The step size in the standardized z-space for the coarse *integration* grid. A step of 1.0 corresponds to one standard deviation.
- `interpolation_subdivisions::Int = 2`: The number of fine-grid steps per coarse integration step.
- `max_log_drop::Float64 = 2.5`: Exploration along any axis stops when the log-density drops by this much from the mode.

# Returns
- `HyperparameterExploration`: A struct containing the complete, normalized results of the exploration.
"""
function explore_hyperparameter_posterior(
        model::INLAModel, y, θ_star, marginalization_method, marginalization_indices;
        integration_step_z::Float64 = 1.0,
        max_log_drop::Float64 = 2.5,
        interpolation_subdivisions::Int = 2
    )
    n_dim = length(θ_star)

    # Step 1: Compute the transformation object
    transform = compute_reparameterization(model, y, θ_star)

    # Step 2: Evaluate the mode point once, authoritatively
    mode_log_density, mode_marginal_result = evaluate_logpdf_and_marginals(
        model, y, θ_star; compute_marginals = true, marginalization_method, marginalization_indices
    )
    mode_point = GridPoint(θ_star, mode_log_density, mode_marginal_result)

    # Step 3: Explore axes and build the lookup table of raw (unnormalized) points
    point_lookup = Dict{NTuple{n_dim, Int}, GridPoint}()
    point_lookup[Tuple(zeros(Int, n_dim))] = mode_point
    step_ranges_per_dim = Vector{UnitRange{Int}}(undef, n_dim)
    evaluation_step_z = integration_step_z / interpolation_subdivisions

    for d in 1:n_dim
        axis_points, axis_range = explore_dimension_and_build_lookup(
            model, y, transform, mode_log_density, d, evaluation_step_z, max_log_drop,
            interpolation_subdivisions, marginalization_method, marginalization_indices
        )
        merge!(point_lookup, axis_points) # Add all on-axis points to the master table
        step_ranges_per_dim[d] = axis_range
    end

    # Step 4: Build the full grid by evaluating off-axis points
    raw_interpolation_points = GridPoint[]
    for key_tuple in Iterators.product(step_ranges_per_dim...)
        if haskey(point_lookup, key_tuple)
            push!(raw_interpolation_points, point_lookup[key_tuple])
            continue
        end

        is_integration_point = all(iszero, collect(key_tuple) .% interpolation_subdivisions)
        θ_off_axis = transform(evaluation_step_z .* collect(key_tuple))

        log_density, marginal_result = evaluate_logpdf_and_marginals(
            model, y, θ_off_axis;
            compute_marginals = is_integration_point,
            marginalization_method = marginalization_method,
            marginalization_indices = marginalization_indices
        )

        if mode_log_density - log_density <= max_log_drop
            push!(raw_interpolation_points, GridPoint(θ_off_axis, log_density, marginal_result))
        end
    end

    # Step 5: Compute the normalization constant using the Jacobian
    integration_indices = findall(p -> p.marginal_result !== nothing, raw_interpolation_points)
    unnormalized_integration_logpdfs = [p.log_density for p in raw_interpolation_points[integration_indices]]

    function logsumexp(x)
        max_x = maximum(x)
        return max_x + log(sum(exp.(x .- max_x)))
    end

    log_z_cell_volume = n_dim * log(integration_step_z)
    log_normalization_constant = logsumexp(unnormalized_integration_logpdfs) + logdet_jacobian(transform) + log_z_cell_volume

    # Step 6: Create the final, clean, normalized GridPoint objects for the user
    final_grid_points = GridPoint[]
    for p in raw_interpolation_points
        normalized_log_density = p.log_density - log_normalization_constant
        push!(final_grid_points, GridPoint(p.θ, normalized_log_density, p.marginal_result))
    end

    return HyperparameterExploration(
        final_grid_points,
        integration_indices,
        transform,
        log_normalization_constant
    )
end
