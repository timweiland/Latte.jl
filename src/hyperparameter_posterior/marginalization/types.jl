"""
Abstract interface for hyperparameter marginalization methods.

Hyperparameter marginalization is step 3 in the INLA workflow:
1. Mode finding → θ_mode
2. Exploration → HyperparameterExploration (coarse grid for latent integration)
3. Hyperparameter marginalization → hyperparameter marginals (THIS STEP)
4. Latent marginalization → latent marginals (weighted mixture from exploration)

This step takes the exploration from step 2 and produces hyperparameter marginal
distributions. Different methods may refine the exploration internally as needed.
"""

export HyperparameterMarginalizationMethod, marginalize_hyperparameters

"""
    HyperparameterMarginalizationMethod

Abstract base type for hyperparameter marginalization methods.

Concrete implementations include:
- `GridBasedMarginal`: Grid-based integration with interpolation and adaptive expansion
"""
abstract type HyperparameterMarginalizationMethod end

"""
    marginalize_hyperparameters(
        method::HyperparameterMarginalizationMethod,
        exploration::AbstractHyperparameterExploration,
        model::INLAModel,
        y;
        progress_callback = nothing
    )

Compute hyperparameter marginal distributions from exploration results.

# Arguments
- `method::HyperparameterMarginalizationMethod`: Marginalization method to use
- `exploration::AbstractHyperparameterExploration`: Results from step 2 (exploration around mode)
- `model::INLAModel`: INLA model specification
- `y`: Observed data
- `progress_callback`: Optional callback for progress tracking

# Returns
- `Vector{<:ContinuousUnivariateDistribution}`: Hyperparameter marginal distributions (in natural space)

# Details
The exploration from step 2 is optimized for latent field marginalization (step 4)
and may be coarse. This function can internally refine the exploration as needed
to produce accurate hyperparameter marginals.

Different methods have different strategies:
- Grid-based: Build interpolant, diagnose tail coverage, adaptively expand if needed
- Importance sampling: Draw samples, compute weights, create empirical distributions
- Adaptive quadrature: Use adaptive integration schemes

# Example
```julia
# After mode finding and exploration
θ_mode, _, _ = find_hyperparameter_mode(model, y)
exploration = explore_hyperparameter_posterior(
    GridExplorationStrategy(), model, y, θ_mode,
    latent_method, latent_indices
)

# Create hyperparameter marginals
method = GridBasedMarginal()  # Uses adaptive expansion
hp_marginals = marginalize_hyperparameters(method, exploration, model, y)

# Access results
mean(hp_marginals[1])  # Mean of first hyperparameter (natural space)
quantile(hp_marginals[1], [0.025, 0.975])  # 95% credible interval
```
"""
function marginalize_hyperparameters(
        method::HyperparameterMarginalizationMethod,
        exploration::AbstractHyperparameterExploration,
        model::INLAModel,
        y;
        progress_callback = nothing
    )
    # Dispatch to method-specific implementation
    return _marginalize_impl(method, exploration, model, y, progress_callback)
end

"""
    _marginalize_impl(method, exploration, model, y, progress_callback)

Internal implementation function for hyperparameter marginalization.
Must be implemented by each concrete marginalization method.
"""
function _marginalize_impl(
        method::HyperparameterMarginalizationMethod,
        exploration::AbstractHyperparameterExploration,
        model::INLAModel,
        y,
        progress_callback
    )
    throw(ArgumentError("Method $(typeof(method)) has not implemented _marginalize_impl"))
end
