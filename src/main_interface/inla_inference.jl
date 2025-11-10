using GaussianMarkovRandomFields
using Optim
using StatsModels
using DataFrames

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
- `latent_marginalization_method::MarginalApproximation = GaussianMarginal()`: Method for latent marginalization
- `hyperparameter_marginalization_method::HyperparameterMarginalizationMethod = GridBasedMarginal()`: Method for hyperparameter marginalization (with adaptive expansion)
- `latent_indices::Union{Nothing, AbstractVector{<:Integer}} = nothing`: Indices to marginalize (default: all)
- `max_log_drop::Float64 = 6.0`: Initial maximum log-density drop for exploration (can be adaptively increased)
- `interpolation_subdivisions::Int = 2`: Subdivision factor for interpolation grid
- `mode_method = BFGS()`: Optimization method for mode finding
- `mode_iterations::Int = 1000`: Maximum iterations for mode finding
- `progress::Bool = true`: Enable progress tracking
- `accumulators::Tuple = (DICAccumulator(), MarginalLogLikelihoodAccumulator())`: Tuple of PosteriorAccumulator objects for model comparison metrics

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

# Custom latent marginalization
result = inla(model, y,
    latent_marginalization_method=SimplifiedLaplace(),
    latent_indices=1:100)

# Custom hyperparameter marginalization (manual mode, no auto-expansion)
result = inla(model, y,
    hyperparameter_marginalization_method=GridBasedMarginal(auto_adjust=false))
```

# Progress Tracking
When `progress=true`, displays a 3-phase progress bar:
- Phase 1 (33%): Mode finding with iteration tracking
- Phase 2 (66%): Exploration with grid point evaluation
- Phase 3 (100%): Hyperparameter marginalization (may include adaptive expansion)

Each phase shows detailed real-time information about the computation.
"""
function inla(
        model::INLAModel,
        y::AbstractVector;
        latent_marginalization_method = GaussianMarginal(),
        hyperparameter_marginalization_method = GridBasedMarginal(auto_adjust = false),
        latent_indices::Union{Nothing, AbstractVector{<:Integer}} = nothing,
        max_log_drop::Float64 = 15.0,
        interpolation_subdivisions::Int = 2,
        mode_method = BFGS(),
        mode_iterations::Int = 1000,
        progress::Bool = true,
        accumulators::Tuple = (DICAccumulator(), MarginalLogLikelihoodAccumulator())
    )

    # Input validation
    validate_inla_inputs(model, y, latent_indices)

    # Auto-detect latent indices if not provided
    if latent_indices === nothing
        # TODO: This is not gonna work for function-based latent priors
        #       Maybe just enforce LatentModel type then?
        #       Could add macro to construct LatentModel from function more easily
        latent_indices = collect(1:length(model.latent_prior))
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
    # This creates a coarse grid optimized for latent field marginalization
    exploration_callback = create_progress_callback(progress_state, "Exploring hyperparameter posterior")
    exploration_start_time = time()

    exploration, accumulators_result = explore_hyperparameter_posterior(
        model, y, θ_star, latent_marginalization_method, latent_indices;
        max_log_drop = max_log_drop,
        interpolation_subdivisions = interpolation_subdivisions,
        progress_callback = exploration_callback,
        accumulators = accumulators
    )

    timing[:exploration] = time() - exploration_start_time
    advance_phase!(progress_state, "Exploration complete", (points = length(exploration.grid_points),))

    # Phase 3: Hyperparameter Marginalization (66% → 100%)
    # This step can refine the exploration internally if needed for accurate marginals
    marginalization_callback = create_progress_callback(progress_state, "Computing hyperparameter marginals")
    marginalization_start_time = time()

    hyperparameter_marginals = marginalize_hyperparameters(
        hyperparameter_marginalization_method,
        exploration,
        model,
        y;
        progress_callback = marginalization_callback
    )

    timing[:hyperparameter_marginalization] = time() - marginalization_start_time
    timing[:total] = time() - total_start_time

    # Finish progress tracking
    finish_progress!(progress_state)

    # Create latent marginals using the existing utility function
    latent_marginals = create_weighted_mixtures(exploration)

    # Split marginals if model is augmented (use views to avoid copying)
    linear_predictor_marginals = nothing
    base_latent_marginals = nothing

    if model.augmentation_info !== nothing
        info = model.augmentation_info

        # Use views to avoid copying marginal distributions
        linear_predictor_marginals = @view latent_marginals[info.linear_predictor_indices]
        base_latent_marginals = @view latent_marginals[info.base_latent_indices]
    end

    # Create convergence diagnostics
    convergence = (
        mode_converged = mode_points !== nothing ? true : false,  # Simplified check
        mode_iterations = length(mode_points),
        exploration_points = length(exploration.grid_points),
        integration_points = length(exploration.integration_indices),
    )

    # Store options used
    options = (
        latent_marginalization_method = latent_marginalization_method,
        hyperparameter_marginalization_method = hyperparameter_marginalization_method,
        latent_indices = latent_indices,
        max_log_drop = max_log_drop,
        interpolation_subdivisions = interpolation_subdivisions,
        mode_method = mode_method,
        mode_iterations = mode_iterations,
        progress = progress,
        y = y,
    )

    return INLAResult(
        hyperparameter_marginals,
        latent_marginals,
        θ_star,
        exploration,
        convergence,
        NamedTuple(timing),
        model,
        options,
        accumulators_result;
        linear_predictor_marginals = linear_predictor_marginals,
        base_latent_marginals = base_latent_marginals,
        augmentation_info = model.augmentation_info
    )
end

function inla(
        formula::FormulaTerm,
        hyperparam_spec::HyperparameterSpec,
        df::DataFrame;
        family,
        trials = :n,
        kwargs...
    )
    _, y, obs_model, latent_model = build_formula_components(formula, df; family, trials)
    model = INLAModel(hyperparam_spec, latent_model, obs_model)
    return inla(
        model, y; kwargs...
    )
end
