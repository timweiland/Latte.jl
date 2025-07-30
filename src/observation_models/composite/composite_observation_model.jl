"""
    CompositeObservationModel{T<:Tuple} <: ObservationModel

An observation model that combines multiple component observation models.

This type follows the factory pattern - it stores component observation models and 
creates `CompositeLikelihood` instances when called with observation data and hyperparameters.

# Fields
- `components::T`: Tuple of component observation models for type stability

# Example
```julia
gaussian_model = ExponentialFamily(Normal)
poisson_model = ExponentialFamily(Poisson)
composite_model = CompositeObservationModel((gaussian_model, poisson_model))

# Materialize with data and hyperparameters
y_composite = CompositeObservations(([1.0, 2.0], [3, 4]))
composite_lik = composite_model(y_composite; σ=1.5)
```
"""
struct CompositeObservationModel{T <: Tuple} <: ObservationModel
    components::T

    function CompositeObservationModel(components::T) where {T <: Tuple}
        if isempty(components)
            throw(ArgumentError("CompositeObservationModel cannot be empty"))
        end
        return new{T}(components)
    end
end

"""
    CompositeLikelihood{T<:Tuple} <: ObservationLikelihood

A materialized composite likelihood that combines multiple component likelihoods.

Created by calling a `CompositeObservationModel` with observation data and hyperparameters.
Provides efficient evaluation of log-likelihood, gradient, and Hessian by summing
contributions from all component likelihoods.

# Fields
- `components::T`: Tuple of materialized component likelihoods
"""
struct CompositeLikelihood{T <: Tuple} <: ObservationLikelihood
    components::T

    function CompositeLikelihood(components::T) where {T <: Tuple}
        return new{T}(components)
    end
end

# Factory pattern implementation - make CompositeObservationModel callable
function (composite_model::CompositeObservationModel)(y::CompositeObservations; kwargs...)
    # Validate that number of components matches
    if length(composite_model.components) != length(y.components)
        throw(ArgumentError("Number of model components ($(length(composite_model.components))) must match number of observation components ($(length(y.components)))"))
    end

    # Materialize each component likelihood
    component_likelihoods = map(composite_model.components, y.components) do model, y_comp
        # Pass all hyperparameters to each component - they'll take what they need
        model(y_comp; kwargs...)
    end

    return CompositeLikelihood(component_likelihoods)
end

# Show methods for nice display
function Base.show(io::IO, model::CompositeObservationModel)
    n_components = length(model.components)
    print(io, "CompositeObservationModel with $n_components component$(n_components == 1 ? "" : "s"):")
    for (i, component) in enumerate(model.components)
        print(io, "\n  [$i] ")
        show(io, component)
    end
    return
end

function Base.show(io::IO, lik::CompositeLikelihood)
    n_components = length(lik.components)
    print(io, "CompositeLikelihood with $n_components component$(n_components == 1 ? "" : "s"):")
    for (i, component) in enumerate(lik.components)
        print(io, "\n  [$i] ")
        show(io, component)
    end
    return
end
