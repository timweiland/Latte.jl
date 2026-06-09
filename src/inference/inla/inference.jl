using GaussianMarkovRandomFields
using Optim
using StatsModels
using DataFrames

export inla

"""
    inla(model::LatentGaussianModel, y::AbstractVector; kwargs...)

Unified INLA inference with automatic parameter selection and progress tracking.

This function provides a simplified interface for INLA inference, automatically
selecting sensible defaults while supporting advanced customization.

# Arguments
- `model::LatentGaussianModel`: The INLA model specification
- `y::AbstractVector`: Observed data

# Keyword Arguments
- `latent_marginalization_method = nothing`: Method for latent marginalization. `nothing`
  resolves per model via [`default_marginalization`](@ref) — compact LTM models get the
  VBC mean correction (`VBCMarginal`), everything else simplified Laplace (`SimplifiedLaplace`).
- `hyperparameter_marginalization_method::HyperparameterMarginalizationMethod = AutoHyperparameterMarginal()`: Method for hyperparameter marginalization (GridSum for D=1, CCD interpolant for D≥2)
- `latent_indices::Union{Nothing, AbstractVector{<:Integer}} = nothing`: Indices to marginalize (default: all)
- `exploration_strategy::ExplorationStrategy = AutoExplorationStrategy()`: Hyperparameter exploration strategy. `AutoExplorationStrategy()` uses grid for D ≤ 2, CCD for D ≥ 3. Can also pass `GridExplorationStrategy(...)` or `CCDExplorationStrategy(...)` directly.
- `mode_method = BFGS()`: Optimization method for mode finding
- `mode_iterations::Int = 1000`: Maximum iterations for mode finding
- `progress::Bool = true`: Enable progress tracking
- `accumulators::Tuple = (DICStrategy(), MarginalLogLikelihoodStrategy(), WAICStrategy(), CPOStrategy())`: Tuple of `PosteriorStrategy` configs for model comparison metrics. Each strategy is materialised into a fresh accumulator per call, so the tuple can be safely reused across multiple `inla()` runs. Pass e.g. `WAICStrategy(n_nodes=25)` to tune knobs.

# Returns
- `INLAResult`: Complete INLA inference results with marginals and diagnostics

# Examples
```julia
# Basic usage with default settings
result = inla(model, y)

# Custom exploration parameters
result = inla(model, y, exploration_strategy=GridExplorationStrategy(max_log_drop=3.0, interpolation_subdivisions=3))

# Disable progress tracking
result = inla(model, y, progress=false)

# Opt into the heavier adaptive latent marginalization
result = inla(model, y,
    latent_marginalization_method=AdaptiveMarginal(),
    latent_indices=1:100)

# Force CCD exploration on a 2D model
result = inla(model, y,
    exploration_strategy=CCDExplorationStrategy())
```

# Progress Tracking
When `progress=true`, displays a 3-phase progress bar:
- Phase 1 (33%): Mode finding with iteration tracking
- Phase 2 (66%): Exploration with grid point evaluation
- Phase 3 (100%): Hyperparameter marginalization (may include adaptive expansion)

Each phase shows detailed real-time information about the computation.
"""
function inla(
        model::LatentGaussianModel,
        y::AbstractVector;
        latent_marginalization_method = nothing,
        hyperparameter_marginalization_method = AutoHyperparameterMarginal(),
        latent_indices::Union{Nothing, AbstractVector{<:Integer}} = nothing,
        exploration_strategy::ExplorationStrategy = AutoExplorationStrategy(),
        mode_method = BFGS(linesearch = Optim.LineSearches.BackTracking(order = 3, maxstep = 5.0)),
        mode_iterations::Int = 1000,
        mode_init = PriorModeStart(),
        mode_diagnostic::Symbol = :warn,
        mode_diagnostic_tol::Float64 = 1.0,
        progress::Bool = true,
        accumulators::Tuple = (DICStrategy(), MarginalLogLikelihoodStrategy(), WAICStrategy(), CPOStrategy()),
        executor::ParallelExecutor = SequentialExecutor(),
        diff_strategy::DifferentiationStrategy = ADStrategy()
    )

    # Pre-process missing observations for prediction
    y_obs, model_pred, prediction_info = _prepare_for_prediction(model, y)

    # Input validation
    validate_inla_inputs(model_pred, y_obs, latent_indices)

    # Resolve the latent-marginalization method from the model: compact LTM models
    # default to the VBC mean correction, everything else to simplified Laplace.
    if latent_marginalization_method === nothing
        latent_marginalization_method = default_marginalization(model_pred)
    end

    accumulators = map(materialize, accumulators)

    # Auto-detect latent indices if not provided
    if latent_indices === nothing
        latent_indices = collect(1:length(model_pred.latent_prior))
    end

    # Initialize progress tracking
    progress_state = initialize_progress!(progress)

    # Store timing information
    timing = Dict{Symbol, Float64}()
    total_start_time = time()

    # Phase 1: Mode Finding (0% → 33%)
    mode_callback = create_progress_callback(progress_state, "Finding hyperparameter mode")
    mode_start_time = time()

    θ_star, mode_points, mode_logdensities, mode_info = find_hyperparameter_mode(
        model_pred, y_obs;
        method = mode_method,
        iterations = mode_iterations,
        collect_points = true,
        progress_callback = mode_callback,
        diff_strategy = diff_strategy,
        mode_init = mode_init,
        executor = executor,
    )

    timing[:mode_finding] = time() - mode_start_time
    advance_phase!(progress_state, "Mode finding complete", (iterations = length(mode_points),))

    # Phase 2: Exploration (33% → 66%)
    # This creates a coarse grid optimized for latent field marginalization
    exploration_callback = create_progress_callback(progress_state, "Exploring hyperparameter posterior")
    exploration_start_time = time()

    exploration, accumulators_result = explore_hyperparameter_posterior(
        exploration_strategy,
        model_pred, y_obs, θ_star, latent_marginalization_method, latent_indices;
        progress_callback = exploration_callback,
        accumulators = accumulators,
        executor = executor,
        diff_strategy = diff_strategy
    )

    timing[:exploration] = time() - exploration_start_time
    advance_phase!(progress_state, "Exploration complete", (points = length(exploration.grid_points),))

    # Mode-quality diagnostic: if the explored grid found a point with
    # log-density much higher than θ*'s, the mode finder probably stuck
    # at a local maximum. Cheap post-hoc check.
    _diagnose_mode_quality(
        mode_info, exploration, model_pred, mode_diagnostic, mode_diagnostic_tol,
    )

    # Phase 3: Hyperparameter Marginalization (66% → 100%)
    # This step can refine the exploration internally if needed for accurate marginals
    marginalization_callback = create_progress_callback(progress_state, "Computing hyperparameter marginals")
    marginalization_start_time = time()

    hyperparameter_marginals = marginalize_hyperparameters(
        hyperparameter_marginalization_method,
        exploration,
        model_pred,
        y_obs;
        progress_callback = marginalization_callback
    )

    timing[:hyperparameter_marginalization] = time() - marginalization_start_time
    timing[:total] = time() - total_start_time

    # Finish progress tracking
    finish_progress!(progress_state)

    # Create latent marginals and KLD diagnostics using the existing utility function
    mixture_result = create_weighted_mixtures(exploration)
    latent_marginals = mixture_result.marginals
    kld = mixture_result.kld

    # Split marginals if model is augmented (use views to avoid copying)
    linear_predictor_marginals = nothing
    base_latent_marginals = nothing

    if model_pred.augmentation_info !== nothing
        info = model_pred.augmentation_info

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
        exploration_strategy = exploration_strategy,
        latent_indices = latent_indices,
        mode_method = mode_method,
        mode_iterations = mode_iterations,
        progress = progress,
        y = y,
        y_obs = y_obs,
    )

    return INLAResult(
        hyperparameter_marginals,
        latent_marginals,
        θ_star,
        exploration,
        convergence,
        NamedTuple(timing),
        model_pred,
        options,
        accumulators_result;
        linear_predictor_marginals = linear_predictor_marginals,
        base_latent_marginals = base_latent_marginals,
        augmentation_info = model_pred.augmentation_info,
        prediction_info = prediction_info,
        kld = kld
    )
end

# Validate, before any optimization, that the spec supplies every hyperparameter
# the formula-built model requires. The formula interface renames a component's
# hyperparameters to avoid collisions (e.g. a walk term `rw2(year)` exposes its
# precision as `τ_rw2`), so a spec written against the bare name would otherwise
# fail deep inside the mode finder with an opaque "missing parameter" error.
function _validate_formula_hyperparameters(latent_model, observation_model, spec::HyperparameterSpec)
    _names(x::NamedTuple) = collect(keys(x))
    _names(x) = collect(x)
    required = Set{Symbol}(_names(hyperparameters(latent_model)))
    union!(required, _names(hyperparameters(observation_model)))
    provided = Set{Symbol}(keys(spec.free)) ∪ Set{Symbol}(keys(spec.fixed))

    missing_hp = setdiff(required, provided)
    isempty(missing_hp) && return nothing

    unused = collect(setdiff(provided, required))
    hints = String[]
    for m in sort!(collect(missing_hp))
        cand = filter(u -> startswith(String(m), String(u) * "_"), unused)
        isempty(cand) || push!(hints, "`$(first(cand))` → `$m`")
    end
    msg = "Hyperparameter spec is missing required parameter(s) " *
        "$(sort!(collect(missing_hp))); provided $(sort!(collect(provided)))."
    isempty(hints) || (
        msg *= " Did you mean " * join(hints, ", ") *
            "? The formula interface renames component hyperparameters to avoid collisions."
    )
    throw(ArgumentError(msg))
end

function inla(
        formula::FormulaTerm,
        hyperparam_spec::HyperparameterSpec,
        df::DataFrame;
        family,
        trials = :n,
        exposure = nothing,
        kwargs...
    )
    # Parse formula terms upfront — needed both for build_formula_components and
    # for storing the parsed terms so predict() can reuse them with predict_cols.
    sch = StatsModels.schema(formula, df)
    tf = StatsModels.apply_schema(formula, sch)
    rhs_terms = tf.rhs isa StatsModels.MatrixTerm ? tf.rhs.terms : [tf.rhs]
    random_terms = Tuple(t for t in rhs_terms if !_is_fixed_effect_term(t))
    fixed_terms = Tuple(t for t in rhs_terms if _is_fixed_effect_term(t))

    _, y, obs_model, latent_model = build_formula_components(formula, df; family, trials, exposure)
    _validate_formula_hyperparameters(latent_model, obs_model, hyperparam_spec)
    model = LatentGaussianModel(hyperparam_spec, latent_model, obs_model)
    result = inla(model, y; kwargs...)

    # Attach formula metadata for predict()
    return _with_options(
        result, (
            formula = formula,
            formula_random_terms = random_terms,
            formula_fixed_terms = fixed_terms,
        )
    )
end

"""Reconstruct an INLAResult with extra fields merged into options."""
function _with_options(result::INLAResult, extra::NamedTuple)
    return INLAResult(
        result.hyperparameter_marginals,
        result.latent_marginals,
        result.hyperparameter_mode,
        result.exploration,
        result.convergence,
        result.computation_time,
        result.model,
        merge(result.options, extra),
        result.accumulators;
        linear_predictor_marginals = result.linear_predictor_marginals,
        base_latent_marginals = result.base_latent_marginals,
        augmentation_info = result.augmentation_info,
        prediction_info = result.prediction_info,
        kld = result.kld
    )
end

# Helper for partitioning formula terms into random vs fixed effects.
# Defaults to false (= random effect) so that custom GMRF term types
# are automatically treated as random effects.
_is_fixed_effect_term(::Any) = false
_is_fixed_effect_term(::StatsModels.InterceptTerm) = true
_is_fixed_effect_term(::StatsModels.ContinuousTerm) = true
_is_fixed_effect_term(::StatsModels.CategoricalTerm) = true
