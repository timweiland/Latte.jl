"""
Main implementation of grid-based hyperparameter marginalization with adaptive expansion.
"""

using Printf

"""
    _marginalize_impl(method::GridBasedMarginal, exploration, model, y, progress_callback)

Main iteration loop for grid-based hyperparameter marginalization.

# Algorithm
1. Initialize asymmetric log-drop limits (from exploration's max_log_drop)
2. Loop until convergence or max iterations:
   a. Build interpolant from current exploration
   b. Compute 1D marginals by integrating interpolant
   c. Diagnose tail coverage for each marginal
   d. If not converged and auto_adjust: expand region where needed
   e. Extend exploration with new points
3. Return final hyperparameter marginal distributions

# Returns
NamedTuple mapping parameter names to `HyperparameterMarginalDistribution` objects.
"""
function _marginalize_impl(
        method::GridBasedMarginal,
        exploration::HyperparameterExploration,
        model::INLAModel,
        y,
        progress_callback
    )
    # Handle progress callback
    if progress_callback === nothing
        progress_callback = (; kwargs...) -> nothing
    end

    n_dim = length(exploration.transform.θ_star)

    # Get initial max_log_drop from exploration
    # (This is the value used in step 2 for latent marginalization)
    initial_max_log_drop = estimate_initial_max_log_drop(exploration)

    # Initialize asymmetric limits (all directions start equal)
    log_drop_limits = AsymmetricLogDropLimits(n_dim, initial_max_log_drop)

    current_exploration = exploration
    converged = false
    iteration = 0
    previous_summary_stats = nothing

    progress_callback(
        status = "Starting hyperparameter marginalization",
        initial_max_log_drop = initial_max_log_drop
    )

    while !converged && iteration < method.max_iterations
        iteration += 1
        progress_callback(status = "Marginalization iteration", iteration = iteration)

        # Step 1: Build interpolant from current exploration
        progress_callback(status = "Building interpolant")
        posterior_approx = build_posterior_interpolant(current_exploration)

        # Step 2: Compute 1D marginals by integrating interpolant
        # (Currently placeholder - will use existing HyperparameterMarginalDistribution)
        progress_callback(status = "Computing preliminary marginals")

        # Step 3: Diagnose tail coverage
        progress_callback(status = "Diagnosing tail coverage")
        diagnostics = diagnose_marginal_coverage(
            posterior_approx,
            log_drop_limits,
            n_dim,
            method.target_tail_mass
        )

        # Step 4: Check convergence
        all_tails_ok = all(d -> d.left_tail_ok && d.right_tail_ok, diagnostics)

        # Also check stability of summary statistics if we have previous iteration
        stats_stable = true
        if previous_summary_stats !== nothing
            current_stats = compute_summary_statistics(posterior_approx, n_dim)
            stats_stable = check_stability(
                previous_summary_stats,
                current_stats,
                method.stability_tolerance
            )
        end

        converged = all_tails_ok

        if !converged
            if method.auto_adjust
                # Step 5: Update limits where needed
                log_drop_limits = update_log_drop_limits(
                    log_drop_limits,
                    diagnostics,
                    method.log_drop_increment,
                    method.max_log_drop_cap,
                    method.allow_asymmetric
                )

                progress_callback(
                    status = "Expanding exploration",
                    iteration = iteration,
                    max_limits = maximum(log_drop_limits.limits)
                )

                # Step 6: Extend exploration with new limits
                current_exploration = extend_exploration_asymmetric(
                    current_exploration,
                    model,
                    y,
                    log_drop_limits,
                    progress_callback
                )

                # Store current stats for next iteration's stability check
                previous_summary_stats = compute_summary_statistics(posterior_approx, n_dim)
            else
                # Manual mode: emit warnings and stop
                emit_expansion_warnings(diagnostics, log_drop_limits, iteration)
                break
            end
        else
            progress_callback(status = "Convergence achieved", iteration = iteration)
        end
    end

    if !converged && method.auto_adjust
        @warn "Hyperparameter marginalization did not converge after $(method.max_iterations) iterations. " *
            "Consider increasing max_iterations or manually setting a higher max_log_drop."

        # Show final diagnostic summary
        final_posterior = build_posterior_interpolant(current_exploration)
        final_diagnostics = diagnose_marginal_coverage(
            final_posterior,
            log_drop_limits,
            n_dim,
            method.target_tail_mass
        )
        emit_final_diagnostic_summary(final_diagnostics, log_drop_limits)
    end

    # Build final interpolant and create marginal distributions
    progress_callback(status = "Creating final marginal distributions")
    final_posterior_approx = build_posterior_interpolant(current_exploration)

    # Extract parameter names from the hyperparameter spec
    param_names = collect(keys(current_exploration.transform.θ_star.spec.free))

    hyperparameter_marginals = NamedTuple(
        param_names[i] => HyperparameterMarginalDistribution(final_posterior_approx, i)
            for i in 1:n_dim
    )

    progress_callback(
        status = "Hyperparameter marginalization complete",
        total_iterations = iteration,
        final_grid_points = length(current_exploration.grid_points)
    )

    return hyperparameter_marginals
end

"""
    estimate_initial_max_log_drop(exploration::HyperparameterExploration)

Estimate the max_log_drop used in the initial exploration.

This is extracted from the exploration grid by finding the maximum log-density
drop from the mode to the boundary points.
"""
function estimate_initial_max_log_drop(exploration::HyperparameterExploration)
    if isempty(exploration.grid_points)
        return 6.0  # Default fallback
    end

    # Find mode log-density (should be at or near zero after normalization)
    mode_log_density = maximum(p.log_density for p in exploration.grid_points)

    # Find minimum log-density at explored points
    min_log_density = minimum(p.log_density for p in exploration.grid_points)

    # The actual max_log_drop used is approximately this difference
    return mode_log_density - min_log_density
end

"""
    emit_expansion_warnings(diagnostics, limits, iteration)

Emit warnings in manual mode when tail coverage is insufficient.
"""
function emit_expansion_warnings(diagnostics, limits, iteration)
    @warn "Hyperparameter marginalization detected insufficient tail coverage (iteration $iteration). " *
        "Consider increasing max_log_drop parameter."

    for diag in diagnostics
        if !diag.left_tail_ok || !diag.right_tail_ok
            println("  Dimension $(diag.dimension):")
            if !diag.left_tail_ok
                suggested = limits.limits[diag.dimension, 1] + diag.suggested_left_extension
                println(
                    "    Left tail: estimated mass = $(@sprintf("%.2e", diag.left_tail_mass)), " *
                        "suggest max_log_drop ≥ $(@sprintf("%.1f", suggested))"
                )
            end
            if !diag.right_tail_ok
                suggested = limits.limits[diag.dimension, 2] + diag.suggested_right_extension
                println(
                    "    Right tail: estimated mass = $(@sprintf("%.2e", diag.right_tail_mass)), " *
                        "suggest max_log_drop ≥ $(@sprintf("%.1f", suggested))"
                )
            end
        end
    end
    return
end

"""
    emit_final_diagnostic_summary(diagnostics, limits)

Emit a summary of diagnostics when non-convergence occurs.
"""
function emit_final_diagnostic_summary(diagnostics, limits)
    println("\nFinal diagnostic summary:")
    for diag in diagnostics
        status = (diag.left_tail_ok && diag.right_tail_ok) ? "✓" : "✗"
        println(
            "  $status Dimension $(diag.dimension): " *
                "left_mass=$(@sprintf("%.2e", diag.left_tail_mass)), " *
                "right_mass=$(@sprintf("%.2e", diag.right_tail_mass))"
        )
    end
    return println()
end
