using Distributions
using Printf

export INLAResult

"""
    INLAResult{HM, LM, Mode, Expl, Post, Conv, Time, Model, Opts}

Results structure for INLA inference containing all outputs from the inference process.

This structure provides organized access to all results from INLA inference, including
hyperparameter marginals, latent marginals, and diagnostic information.

# Type Parameters
All fields are fully typed for type stability and performance.

# Fields
- `hyperparameter_marginals::HM`: Vector of marginal distributions for each hyperparameter (lazy evaluation)
- `latent_marginals::LM`: Vector of marginal distributions for latent variables (WeightedMixture)
- `hyperparameter_mode::Mode`: Mode of the hyperparameter posterior in natural space (NamedTuple)
- `exploration::Expl`: Results from posterior exploration (HyperparameterExploration)
- `posterior_approximation::Post`: Interpolated posterior approximation (HyperparameterPosteriorApproximation)
- `convergence::Conv`: Convergence diagnostics and information (NamedTuple)
- `computation_time::Time`: Timing breakdown by computation phase (NamedTuple)
- `model::Model`: Original INLA model specification (INLAModel)
- `options::Opts`: Options used for inference (NamedTuple)

# Usage
```julia
result = inla_inference(model, y)

# Access hyperparameter marginals
result.hyperparameter_marginals[1]  # First hyperparameter marginal
mean(result.hyperparameter_marginals[1])  # Mean of first hyperparameter

# Access latent marginals
result.latent_marginals[1]  # First latent variable marginal (WeightedMixture)

# Access mode in natural space (named tuple with both free and fixed parameters)
result.hyperparameter_mode.σ  # Access by name
result.hyperparameter_mode    # (σ = 2.5, ρ = 0.3, μ = 0.0)

# Access diagnostics
result.convergence.mode_converged      # Did mode finding converge?
result.computation_time.total          # Total computation time
result.computation_time.mode_finding   # Time spent finding mode
```
"""
struct INLAResult{HM, LM, Mode, Expl, Post, Conv, Time, Model, Opts}
    hyperparameter_marginals::HM
    latent_marginals::LM
    hyperparameter_mode::Mode
    exploration::Expl
    posterior_approximation::Post
    convergence::Conv
    computation_time::Time
    model::Model
    options::Opts
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

    # Show mode as named tuple
    mode_str = join(["$k=$(round(v, digits = 4))" for (k, v) in pairs(result.hyperparameter_mode)], ", ")
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

    return print(io, "  Use .hyperparameter_marginals, .latent_marginals for analysis")
end
