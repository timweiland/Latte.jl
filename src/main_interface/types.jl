using Distributions
using Printf

export INLAResult

"""
    INLAResult{HM, LM, Mode, Expl, Conv, Time, Model, Opts, Acc}

Results structure for INLA inference containing all outputs from the inference process.

This structure provides organized access to all results from INLA inference, including
hyperparameter marginals, latent marginals, diagnostic information, and model comparison metrics.

# Type Parameters
All fields are fully typed for type stability and performance.

# Fields
- `hyperparameter_marginals::HM`: NamedTuple mapping parameter names to marginal distributions for each hyperparameter
- `latent_marginals::LM`: Vector of marginal distributions for latent variables (WeightedMixture)
- `hyperparameter_mode::Mode`: Mode of the hyperparameter posterior (WorkingHyperparameters)
- `exploration::Expl`: Results from posterior exploration (HyperparameterExploration)
- `convergence::Conv`: Convergence diagnostics and information (NamedTuple)
- `computation_time::Time`: Timing breakdown by computation phase (NamedTuple)
- `model::Model`: Original INLA model specification (INLAModel)
- `options::Opts`: Options used for inference (NamedTuple)
- `accumulators::Acc`: Tuple of PosteriorAccumulator objects with computed metrics (e.g., DIC, marginal likelihood)
- `linear_predictor_marginals::Union{Nothing, Vector}`: Marginals for linear predictors η (if augmented model)
- `base_latent_marginals::Union{Nothing, Vector}`: Marginals for base latent components (if augmented model)
- `augmentation_info::Union{Nothing, AugmentationInfo}`: Metadata about latent field augmentation

# Usage
```julia
result = inla_inference(model, y)

# Access hyperparameter marginals (by name)
result.hyperparameter_marginals.τ  # Marginal for τ hyperparameter
mean(result.hyperparameter_marginals.τ)  # Mean of τ hyperparameter

# Access latent marginals
result.latent_marginals[1]  # First latent variable marginal (WeightedMixture)

# Access mode (WorkingHyperparameters)
result.hyperparameter_mode       # WorkingHyperparameters
convert(NamedTuple, convert(NaturalHyperparameters, result.hyperparameter_mode))  # Convert to NamedTuple in natural space

# Access diagnostics
result.convergence.mode_converged      # Did mode finding converge?
result.computation_time.total          # Total computation time
result.computation_time.mode_finding   # Time spent finding mode

# Access model comparison metrics
result.accumulators[1]                 # First accumulator (e.g., DICAccumulator)
result.accumulators[1].DIC             # DIC value
result.accumulators[1].p_D             # Effective parameters
```
"""
struct INLAResult{HM, LM, Mode, Expl, Conv, Time, Model, Opts, Acc, LPM, BLM, AugInfo}
    hyperparameter_marginals::HM
    latent_marginals::LM
    hyperparameter_mode::Mode
    exploration::Expl
    convergence::Conv
    computation_time::Time
    model::Model
    options::Opts
    accumulators::Acc
    linear_predictor_marginals::LPM
    base_latent_marginals::BLM
    augmentation_info::AugInfo

    function INLAResult(
            hyperparameter_marginals::HM,
            latent_marginals::LM,
            hyperparameter_mode::Mode,
            exploration::Expl,
            convergence::Conv,
            computation_time::Time,
            model::Model,
            options::Opts,
            accumulators::Acc;
            linear_predictor_marginals::LPM = nothing,
            base_latent_marginals::BLM = nothing,
            augmentation_info::AugInfo = nothing
        ) where {HM, LM, Mode, Expl, Conv, Time, Model, Opts, Acc, LPM, BLM, AugInfo}
        return new{HM, LM, Mode, Expl, Conv, Time, Model, Opts, Acc, LPM, BLM, AugInfo}(
            hyperparameter_marginals,
            latent_marginals,
            hyperparameter_mode,
            exploration,
            convergence,
            computation_time,
            model,
            options,
            accumulators,
            linear_predictor_marginals,
            base_latent_marginals,
            augmentation_info
        )
    end
end

"""
    Base.show(io::IO, result::INLAResult)

Pretty printing for INLAResult objects.
"""
function Base.show(io::IO, result::INLAResult)
    n_hyperparams = length(result.hyperparameter_marginals)
    n_latent = length(result.latent_marginals)

    println(io, "INLAResult:")
    println(io, "  Model: ", typeof(result.model))
    println(io, "  Hyperparameters: ", n_hyperparams)
    println(io, "  Latent variables: ", n_latent)

    # Show augmentation info if present
    if result.augmentation_info !== nothing
        info = result.augmentation_info
        println(io, "  Augmented structure:")
        println(io, "    - Linear predictors (η): ", info.n_linear_predictors, " variables")
        println(io, "    - Base latent components: ", info.n_base_latent, " variables")
    end

    # Show mode as named tuple in natural space
    mode_natural = convert(NaturalHyperparameters, result.hyperparameter_mode)
    mode_nt = convert(NamedTuple, mode_natural)
    mode_str = join(["$k=$(round(v, digits = 4))" for (k, v) in pairs(mode_nt)], ", ")
    println(io, "  Mode: (", mode_str, ")")

    # Show convergence status
    if haskey(result.convergence, :mode_converged)
        converged_symbol = result.convergence.mode_converged ? "✓" : "✗"
        println(io, "  Convergence: ", converged_symbol)
    end

    # Show timing information
    if haskey(result.computation_time, :total)
        println(io, "  Total time: ", @sprintf("%.2f", result.computation_time.total), " seconds")
    end

    # Show exploration summary
    n_exploration_points = length(result.exploration.grid_points)
    n_integration_points = length(result.exploration.integration_indices)
    println(io, "  Exploration: ", n_exploration_points, " points (", n_integration_points, " integration)")

    # Show model comparison metrics if present
    if !isempty(result.accumulators)
        println(io, "\nModel comparison metrics:")
        for acc in result.accumulators
            show(io, MIME("text/plain"), acc)
            println(io)
        end
    end

    return print(io, "Use .hyperparameter_marginals, .latent_marginals, .accumulators for analysis")
end
