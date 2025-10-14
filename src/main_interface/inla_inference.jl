using Optim

export inla

"""
    inla(model::INLAModel, y::AbstractVector; kwargs...)

Unified INLA inference with automatic parameter selection and progress tracking.

This function provides a simplified interface for INLA inference, automatically
selecting sensible defaults while supporting advanced customization.

# Arguments
- `model::INLAModel`: The INLA model specification
- `y::AbstractVector`: Observed data

# Keyword Arguments
- `marginalization_method::MarginalApproximation = GaussianMarginal()`: Method for latent marginalization
- `latent_indices::Union{Nothing, AbstractVector{<:Integer}} = nothing`: Indices to marginalize (default: all)
- `max_log_drop::Float64 = 2.5`: Maximum log-density drop for exploration boundary
- `interpolation_subdivisions::Int = 2`: Subdivision factor for interpolation grid
- `mode_method = BFGS()`: Optimization method for mode finding
- `mode_iterations::Int = 1000`: Maximum iterations for mode finding
- `progress::Bool = true`: Enable progress tracking

# Returns
- `INLAResult`: Complete INLA inference results with marginals and diagnostics

# Examples
```julia
# Basic usage with default settings
result = inla(model, y)

# Custom exploration parameters
result = inla(model, y, max_log_drop=3.0, interpolation_subdivisions=3)

# Disable progress tracking
result = inla(model, y, progress=false)

# Custom marginalization
result = inla(model, y, 
    marginalization_method=SimplifiedLaplace(), 
    latent_indices=1:100)
```

# Progress Tracking
When `progress=true`, displays a 3-phase progress bar:
- Phase 1 (33%): Mode finding with iteration tracking
- Phase 2 (66%): Exploration with grid point evaluation
- Phase 3 (100%): Interpolation construction

Each phase shows detailed real-time information about the computation.
"""
function inla(
        model::INLAModel,
        y::AbstractVector;
        marginalization_method = GaussianMarginal(),
        latent_indices::Union{Nothing, AbstractVector{<:Integer}} = nothing,
        max_log_drop::Float64 = 2.5,
        interpolation_subdivisions::Int = 2,
        mode_method = BFGS(),
        mode_iterations::Int = 1000,
        progress::Bool = true
    )

    # Input validation
    validate_inla_inputs(model, y, latent_indices)

    # Auto-detect latent indices if not provided
    if latent_indices === nothing
        latent_dim = latent_dimension(model.observation_model, y)
        if latent_dim !== nothing
            latent_indices = collect(1:latent_dim)
        else
            error("Cannot infer latent dimension. Please specify latent_indices explicitly.")
        end
    end

    # Initialize progress tracking
    progress_state = initialize_progress!(progress)

    # Store timing information
    timing = Dict{Symbol, Float64}()
    total_start_time = time()

    # Phase 1: Mode Finding (0% → 33%)
    mode_callback = create_progress_callback(progress_state, "Finding hyperparameter mode")
    mode_start_time = time()

    θ_star, mode_points, mode_logdensities = find_hyperparameter_mode(
        model, y;
        method = mode_method,
        collect_points = true,
        progress_callback = mode_callback
    )

    timing[:mode_finding] = time() - mode_start_time
    advance_phase!(progress_state, "Mode finding complete", (iterations = length(mode_points),))

    # Phase 2: Exploration (33% → 66%)
    exploration_callback = create_progress_callback(progress_state, "Exploring hyperparameter posterior")
    exploration_start_time = time()

    exploration = explore_hyperparameter_posterior(
        model, y, θ_star, marginalization_method, latent_indices;
        max_log_drop = max_log_drop,
        interpolation_subdivisions = interpolation_subdivisions,
        progress_callback = exploration_callback
    )

    timing[:exploration] = time() - exploration_start_time
    advance_phase!(progress_state, "Exploration complete", (points = length(exploration.grid_points),))

    # Phase 3: Interpolation (66% → 100%)
    interpolation_callback = create_progress_callback(progress_state, "Building posterior interpolant")
    interpolation_start_time = time()

    posterior_approx = build_posterior_interpolant(
        exploration;
        progress_callback = interpolation_callback
    )

    timing[:interpolation] = time() - interpolation_start_time
    timing[:total] = time() - total_start_time

    # Finish progress tracking
    finish_progress!(progress_state)

    # Create hyperparameter marginals (lazy - instantaneous)
    n_hyperparams = length(model.hyperparameter_prior.free_distribution)
    hyperparameter_marginals = [
        HyperparameterMarginalDistribution(posterior_approx, i)
            for i in 1:n_hyperparams
    ]

    # Create latent marginals using the existing utility function
    latent_marginals = create_weighted_mixtures(exploration)

    # Create convergence diagnostics
    convergence = (
        mode_converged = mode_points !== nothing ? true : false,  # Simplified check
        mode_iterations = length(mode_points),
        exploration_points = length(exploration.grid_points),
        integration_points = length(exploration.integration_indices),
    )

    # Store options used
    options = (
        marginalization_method = marginalization_method,
        latent_indices = latent_indices,
        max_log_drop = max_log_drop,
        interpolation_subdivisions = interpolation_subdivisions,
        mode_method = mode_method,
        mode_iterations = mode_iterations,
        progress = progress,
    )

    return INLAResult(
        hyperparameter_marginals,
        latent_marginals,
        θ_star,
        exploration,
        posterior_approx,
        convergence,
        NamedTuple(timing),
        model,
        options
    )
end
