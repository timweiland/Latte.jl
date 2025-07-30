"""
    CompositeObservations{T<:Tuple} <: AbstractVector{Float64}

A composite observation vector that stores observation data as a tuple of component vectors.

This type implements the `AbstractVector` interface and allows combining different observation
datasets while maintaining their structure. The composite vector presents a unified view
where indexing delegates to the appropriate component vector.

# Fields
- `components::T`: Tuple of observation vectors, one per likelihood component

# Example
```julia
y1 = [1.0, 2.0, 3.0]  # Gaussian observations
y2 = [4.0, 5.0]       # More Gaussian observations
y_composite = CompositeObservations((y1, y2))

length(y_composite)    # 5
y_composite[1]         # 1.0
y_composite[4]         # 4.0
collect(y_composite)   # [1.0, 2.0, 3.0, 4.0, 5.0]
```
"""
struct CompositeObservations{T <: Tuple} <: AbstractVector{Float64}
    components::T

    function CompositeObservations(components::T) where {T <: Tuple}
        # Validation
        if isempty(components)
            throw(ArgumentError("CompositeObservations cannot be empty"))
        end

        # Convert all components to Vector{Float64} for type stability
        converted_components = map(c -> Vector{Float64}(c), components)
        return new{typeof(converted_components)}(converted_components)
    end
end

# AbstractVector interface implementation
Base.size(co::CompositeObservations) = (sum(length, co.components),)

function Base.getindex(co::CompositeObservations, i::Int)
    @boundscheck checkbounds(co, i)

    # Find which component contains index i
    cumulative_idx = 0
    for component in co.components
        if i <= cumulative_idx + length(component)
            return component[i - cumulative_idx]
        end
        cumulative_idx += length(component)
    end

    # This should never be reached due to bounds check
    throw(BoundsError(co, i))
end

function Base.iterate(co::CompositeObservations, state = (1, 1))
    comp_idx, elem_idx = state

    # Check if we've gone past all components
    if comp_idx > length(co.components)
        return nothing
    end

    current_component = co.components[comp_idx]

    # Check if we've exhausted current component
    if elem_idx > length(current_component)
        # Move to next component
        return iterate(co, (comp_idx + 1, 1))
    end

    # Return current element and advance state
    value = current_component[elem_idx]
    next_state = if elem_idx < length(current_component)
        (comp_idx, elem_idx + 1)  # Next element in same component
    else
        (comp_idx + 1, 1)         # First element of next component
    end

    return (value, next_state)
end

# Show methods for nice display
function Base.show(io::IO, co::CompositeObservations)
    n_components = length(co.components)
    total_length = length(co)
    print(io, "CompositeObservations with $n_components component$(n_components == 1 ? "" : "s") ($total_length total observations):")

    for (i, component) in enumerate(co.components)
        n_obs = length(component)
        print(io, "\n  [$i] $n_obs observations: ")

        # Show a preview of the data
        if n_obs <= 3
            print(io, "[", join(component, ", "), "]")
        elseif n_obs <= 6
            print(io, "[", join(component[1:3], ", "), ", ..., ", join(component[(end - 1):end], ", "), "]")
        else
            print(io, "[", join(component[1:2], ", "), ", ..., ", component[end], "] (", n_obs, " elements)")
        end
    end
    return
end

Base.show(io::IO, ::MIME"text/plain", co::CompositeObservations) = show(io, co)
