using Distributions
using Printf

export INLAResult

"""
    INLAResult

Results structure for INLA inference containing all outputs from the inference process.

This structure provides organized access to all results from INLA inference, including
hyperparameter marginals, latent marginals, and diagnostic information.

# Fields
- `hyperparameter_marginals::Vector{HyperparameterMarginalDistribution}`: Marginal distributions for each hyperparameter (lazy evaluation)
- `latent_marginals::Vector{WeightedMixture}`: Marginal distributions for latent variables
- `hyperparameter_mode::Vector{Float64}`: Mode of the hyperparameter posterior
- `exploration::HyperparameterExploration`: Results from posterior exploration
- `posterior_approximation::HyperparameterPosteriorApproximation`: Interpolated posterior approximation
- `convergence::NamedTuple`: Convergence diagnostics and information
- `computation_time::NamedTuple`: Timing breakdown by computation phase
- `model::INLAModel`: Original INLA model specification
- `options::NamedTuple`: Options used for inference

# Usage
```julia
result = inla_inference(model, y)

# Access hyperparameter marginals
result.hyperparameter_marginals[1]  # First hyperparameter marginal
mean(result.hyperparameter_marginals[1])  # Mean of first hyperparameter

# Access latent marginals
result.latent_marginals[1]  # First latent variable marginal (WeightedMixture)

# Access mode and exploration results
result.hyperparameter_mode  # Mode of hyperparameter posterior
result.exploration.mode     # Should be same as hyperparameter_mode

# Access diagnostics
result.convergence.mode_converged      # Did mode finding converge?
result.computation_time.total          # Total computation time
result.computation_time.mode_finding   # Time spent finding mode
```
"""
struct INLAResult
    hyperparameter_marginals::Vector{HyperparameterMarginalDistribution}
    latent_marginals::Vector{WeightedMixture}
    hyperparameter_mode::Vector{Float64}
    exploration::HyperparameterExploration
    posterior_approximation::HyperparameterPosteriorApproximation
    convergence::NamedTuple
    computation_time::NamedTuple
    model::INLAModel
    options::NamedTuple
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
    println(io, "  Mode: [", join([@sprintf("%.4f", x) for x in result.hyperparameter_mode], ", "), "]")

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
