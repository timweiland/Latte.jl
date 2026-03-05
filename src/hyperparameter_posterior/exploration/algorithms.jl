export explore_hyperparameter_posterior, explore_half_axis_by_steps, explore_dimension_and_build_lookup

using StatsFuns: logsumexp

"""
    explore_half_axis_by_steps(model, y, transform, mode_logpdf, dim, direction, evaluation_step_z, max_log_drop, interpolation_subdivisions, marginalization_method, marginalization_indices, accumulators)

Explores one direction along a dimension using integer step indices.
Returns a list of (key, GridPoint) tuples where key is an integer coordinate tuple.
"""
function explore_half_axis_by_steps(
        model, y, transform::ReparameterizationTransform, mode_logpdf,
        dim::Int, direction::Int, evaluation_step_z::Float64, max_log_drop::Float64, interpolation_subdivisions::Int,
        marginalization_method, marginalization_indices, accumulators::Tuple,
        accumulator_call_keys::Vector
    )
    n_dim = length(transform.θ_star)
    keyed_points = []

    step_count = 1
    while true
        key_vec = zeros(Int, n_dim)
        key_vec[dim] = direction * step_count
        key = Tuple(key_vec)

        z_vec = evaluation_step_z .* collect(key)
        θ_test = transform(z_vec)  # Returns WorkingHyperparameters directly

        is_integration_point = (step_count % interpolation_subdivisions == 0)

        result = evaluate_at_grid_point(
            model, y, θ_test;
            compute_marginals = is_integration_point,
            marginalization_method = marginalization_method,
            marginalization_indices = marginalization_indices
        )

        if mode_logpdf - result.log_density > max_log_drop
            break
        end

        point = GridPoint(θ_test, result.log_density, result.marginal_result)

        # Call accumulators eagerly and record the grid key for ordering
        if is_integration_point && !isempty(accumulators)
            for acc in accumulators
                accumulate!(
                    acc;
                    result...,
                    θ = θ_test,
                    y = y,
                    is_mode = false
                )
            end
            push!(accumulator_call_keys, key)
        end

        push!(keyed_points, (key, point))
        step_count += 1
    end

    return keyed_points
end

"""
    explore_dimension_and_build_lookup(model, y, transform, mode_logpdf, dim, evaluation_step_z, max_log_drop, interpolation_subdivisions, marginalization_method, marginalization_indices, accumulators)

Explore a single dimension and build lookup table for integer-based grid coordinates.
Calls explore_half_axis_by_steps for both directions and collects results.

Returns (point_lookup, step_range) where:
- point_lookup: Dictionary mapping integer tuple keys to GridPoint objects
- step_range: Range of integer steps found for this dimension
"""
function explore_dimension_and_build_lookup(
        model, y, transform::ReparameterizationTransform, mode_logpdf,
        dim::Int, evaluation_step_z::Float64, max_log_drop::Float64, interpolation_subdivisions::Int,
        marginalization_method, marginalization_indices, accumulators::Tuple,
        accumulator_call_keys::Vector
    )
    # Call our helper function for both directions
    pos_points = explore_half_axis_by_steps(
        model, y, transform, mode_logpdf,
        dim, 1, evaluation_step_z, max_log_drop, interpolation_subdivisions,
        marginalization_method, marginalization_indices, accumulators,
        accumulator_call_keys
    )
    neg_points = explore_half_axis_by_steps(
        model, y, transform, mode_logpdf,
        dim, -1, evaluation_step_z, max_log_drop, interpolation_subdivisions,
        marginalization_method, marginalization_indices, accumulators,
        accumulator_call_keys
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

# Arguments
- `θ_star::WorkingHyperparameters`: The posterior mode in working space

# Keyword Arguments
- `integration_step_z::Float64 = 1.0`: The step size in the standardized z-space for the coarse *integration* grid. A step of 1.0 corresponds to one standard deviation.
- `interpolation_subdivisions::Int = 2`: The number of fine-grid steps per coarse integration step.
- `max_log_drop::Float64 = 2.5`: Exploration along any axis stops when the log-density drops by this much from the mode.
- `progress_callback`: Optional function for progress updates with signature `f(; kwargs...)`
- `accumulators::Tuple = ()`: Tuple of PosteriorAccumulator objects to process integration points

# Returns
- `HyperparameterExploration`: A struct containing the complete, normalized results of the exploration.
- `accumulators`: Tuple of finalized accumulators (if provided)
"""
function explore_hyperparameter_posterior(
        model::INLAModel, y, θ_star::WorkingHyperparameters, marginalization_method, marginalization_indices;
        integration_step_z::Float64 = 1.0,
        max_log_drop::Float64 = 2.5,
        interpolation_subdivisions::Int = 2,
        progress_callback = nothing,
        accumulators::Tuple = ()
    )
    # Handle progress callback
    if progress_callback === nothing
        progress_callback = (; kwargs...) -> nothing
    end

    n_dim = length(θ_star)

    # Step 1: Compute the transformation object
    progress_callback(status = "Computing reparameterization", dimensions = n_dim)
    transform = compute_reparameterization(model, y, θ_star)

    # Step 2: Evaluate the mode point once, authoritatively
    progress_callback(status = "Evaluating mode point", mode = θ_star)
    mode_result = evaluate_at_grid_point(
        model, y, θ_star; compute_marginals = true, marginalization_method, marginalization_indices
    )
    mode_point = GridPoint(θ_star, mode_result.log_density, mode_result.marginal_result)
    mode_log_density = mode_result.log_density

    # Track the order in which accumulators are called (grid coordinate keys).
    # This lets us compute the permutation from call order → grid order in finalize!.
    mode_key = ntuple(_ -> 0, n_dim)
    accumulator_call_keys = NTuple{n_dim, Int}[mode_key]

    # Call accumulators for the mode point
    if !isempty(accumulators)
        for acc in accumulators
            accumulate!(
                acc;
                mode_result...,
                θ = θ_star,
                y = y,
                is_mode = true
            )
        end
    end

    # Step 3: Explore axes and build the lookup table of raw (unnormalized) points
    progress_callback(status = "Starting axis exploration", dimensions = n_dim)
    point_lookup = Dict{NTuple{n_dim, Int}, typeof(mode_point)}()
    point_lookup[mode_key] = mode_point
    step_ranges_per_dim = Vector{UnitRange{Int}}(undef, n_dim)
    evaluation_step_z = integration_step_z / interpolation_subdivisions

    for d in 1:n_dim
        progress_callback(status = "Exploring axis", current_dimension = d, total_dimensions = n_dim)
        axis_points, axis_range = explore_dimension_and_build_lookup(
            model, y, transform, mode_log_density, d, evaluation_step_z, max_log_drop,
            interpolation_subdivisions, marginalization_method, marginalization_indices, accumulators,
            accumulator_call_keys
        )
        merge!(point_lookup, axis_points) # Add all on-axis points to the master table
        step_ranges_per_dim[d] = axis_range
    end

    # Step 4: Build the full grid by evaluating off-axis points
    total_grid_points = prod(length.(step_ranges_per_dim))
    progress_callback(status = "Building full grid", estimated_total_points = total_grid_points)

    raw_interpolation_points = typeof(mode_point)[]
    points_evaluated = 0

    for key_tuple in Iterators.product(step_ranges_per_dim...)
        points_evaluated += 1

        if haskey(point_lookup, key_tuple)
            push!(raw_interpolation_points, point_lookup[key_tuple])
            continue
        end

        is_integration_point = all(iszero, collect(key_tuple) .% interpolation_subdivisions)
        θ_off_axis = transform(evaluation_step_z .* collect(key_tuple))  # Returns WorkingHyperparameters

        result = evaluate_at_grid_point(
            model, y, θ_off_axis;
            compute_marginals = is_integration_point,
            marginalization_method = marginalization_method,
            marginalization_indices = marginalization_indices
        )

        if mode_log_density - result.log_density <= max_log_drop
            push!(raw_interpolation_points, GridPoint(θ_off_axis, result.log_density, result.marginal_result))

            # Call accumulators eagerly and record the grid key for ordering
            if is_integration_point && !isempty(accumulators)
                for acc in accumulators
                    accumulate!(
                        acc;
                        result...,
                        θ = θ_off_axis,
                        y = y,
                        is_mode = false
                    )
                end
                push!(accumulator_call_keys, key_tuple)
            end
        end

        # Update progress for every grid point
        progress_callback(
            status = "Evaluating grid points",
            points_evaluated = points_evaluated,
            total_points = total_grid_points,
            current_θ = θ_off_axis,
            log_density = result.log_density
        )
    end

    # Step 5: Compute permutation from grid order → accumulator call order.
    # Accumulators were called eagerly in discovery order (mode, neg axis, pos axis, off-axis),
    # but integration_indices follows grid order (Iterators.product iteration).
    # The permutation lets get_integration_weights reorder weights to match call order.
    accumulator_reorder = Int[]
    if !isempty(accumulators)
        # Build grid-order position index via a single pass over the grid
        call_key_set = Set(accumulator_call_keys)
        grid_position = Dict{NTuple{n_dim, Int}, Int}()
        grid_idx = 0
        for key_tuple in Iterators.product(step_ranges_per_dim...)
            if key_tuple in call_key_set
                grid_idx += 1
                grid_position[key_tuple] = grid_idx
            end
        end
        # accumulator_reorder[k] = grid position of the k-th accumulator call
        # get_integration_weights uses: weights_grid[accumulator_reorder] → call order
        accumulator_reorder = [grid_position[k] for k in accumulator_call_keys]
    end

    # Step 6: Compute the normalization constant using the Jacobian
    progress_callback(status = "Computing normalization", total_explored_points = length(raw_interpolation_points))
    integration_indices = findall(p -> p.marginal_result !== nothing, raw_interpolation_points)
    unnormalized_integration_logpdfs = [p.log_density for p in raw_interpolation_points[integration_indices]]

    log_z_cell_volume = n_dim * log(integration_step_z)
    log_normalization_constant = logsumexp(unnormalized_integration_logpdfs) + logdet_jacobian(transform) + log_z_cell_volume

    # Step 6: Create the final, clean, normalized GridPoint objects for the user
    progress_callback(status = "Finalizing exploration", integration_points = length(integration_indices))

    # Warn if exploration found very few points
    if length(raw_interpolation_points) < 3
        @warn "Exploration found only $(length(raw_interpolation_points)) grid points. Consider relaxing exploration parameters: increase max_log_drop (current: $max_log_drop) or decrease integration_step_z (current: $integration_step_z) for better coverage."
    end

    final_grid_points = typeof(mode_point)[]
    for p in raw_interpolation_points
        normalized_log_density = p.log_density - log_normalization_constant
        push!(final_grid_points, GridPoint(p.θ, normalized_log_density, p.marginal_result))
    end

    # Create exploration object
    exploration = HyperparameterExploration(
        final_grid_points,
        integration_indices,
        transform,
        log_normalization_constant;
        accumulator_reorder = isempty(accumulator_reorder) ? collect(1:length(integration_indices)) : accumulator_reorder
    )

    # Finalize accumulators
    if !isempty(accumulators)
        progress_callback(status = "Finalizing accumulators", n_accumulators = length(accumulators))
        for acc in accumulators
            finalize!(acc, exploration)
        end
    end

    progress_callback(status = "Exploration complete", final_points = length(final_grid_points))

    return exploration, accumulators
end
