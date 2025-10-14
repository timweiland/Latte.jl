export BinomialObservations, successes, trials

"""
    BinomialObservations <: AbstractVector{Tuple{Int, Int}}

Combined observation type for binomial data containing both successes and trials.

This type packages binomial observation data (number of successes and trials)
into a single vector-like object where each element is a (successes, trials) tuple.

# Fields
- `successes::Vector{Int}`: Number of successes for each observation
- `trials::Vector{Int}`: Number of trials for each observation

# Example
```julia
# Create binomial observations
y = BinomialObservations([3, 1, 4], [5, 8, 6])

# Access as tuples
y[1]  # (3, 5)
y[2]  # (1, 8)

# Use in INLA
result = inla(@formula(r ~ Independent(hospital)), data; family=Binomial)
```
"""
struct BinomialObservations <: AbstractVector{Tuple{Int, Int}}
    successes::Vector{Int}
    trials::Vector{Int}

    function BinomialObservations(successes::AbstractVector{<:Integer}, trials::AbstractVector{<:Integer})
        successes_int = Int.(successes)
        trials_int = Int.(trials)

        if length(successes_int) != length(trials_int)
            error("Length of successes ($(length(successes_int))) must match length of trials ($(length(trials_int)))")
        end

        # Validate that successes ≤ trials
        for i in eachindex(successes_int)
            if successes_int[i] > trials_int[i]
                error("Number of successes ($(successes_int[i])) cannot exceed number of trials ($(trials_int[i])) at index $i")
            end
            if successes_int[i] < 0 || trials_int[i] < 0
                error("Successes and trials must be non-negative at index $i")
            end
        end

        return new(successes_int, trials_int)
    end
end

# AbstractVector interface implementation
Base.size(y::BinomialObservations) = size(y.successes)
Base.getindex(y::BinomialObservations, i::Int) = (y.successes[i], y.trials[i])
Base.getindex(y::BinomialObservations, I) = [y[i] for i in I]
Base.IndexStyle(::Type{BinomialObservations}) = IndexLinear()

# Iteration interface
Base.iterate(y::BinomialObservations, i::Int = 1) = i > length(y) ? nothing : (y[i], i + 1)

# Convenience accessors
"""
    successes(y::BinomialObservations) -> Vector{Int}

Extract the successes vector from binomial observations.
"""
successes(y::BinomialObservations) = y.successes

"""
    trials(y::BinomialObservations) -> Vector{Int}

Extract the trials vector from binomial observations.
"""
trials(y::BinomialObservations) = y.trials

# Display
function Base.show(io::IO, y::BinomialObservations)
    return print(io, "BinomialObservations($(length(y)) observations)")
end

function Base.show(io::IO, ::MIME"text/plain", y::BinomialObservations)
    println(io, "$(length(y))-element BinomialObservations:")
    for i in eachindex(y)
        println(io, "  [$i]: $(y.successes[i])/$(y.trials[i])")
    end
    return
end
