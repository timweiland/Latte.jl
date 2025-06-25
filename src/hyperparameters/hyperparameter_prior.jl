using Distributions

export HyperparameterPrior
export get_hyperparameter, set_hyperparameter!, to_named, to_vector, extract_hyperparameters

"""
    HyperparameterPrior{FreeNames, AllNames, D, F}

A struct that wraps a hyperparameter prior with both free and fixed parameters.
The parameter names are encoded in the type for complete type safety.

# Type Parameters
- `FreeNames`: Tuple of symbols for free parameters (e.g., (:ρ, :τ))
- `AllNames`: Tuple of symbols for all parameters (e.g., (:μ, :ρ, :σ, :τ))
- `D`: The distribution type over free parameters only
- `F`: The NamedTuple type for fixed parameters

# Fields
- `free_distribution::D`: Distribution only over free parameters (for optimization)
- `fixed_values::F`: NamedTuple of fixed parameter values
- `name_to_index::Dict{Symbol, Int}`: Maps free parameter names to indices

# Examples
```julia
# All parameters free (existing behavior)
hp_prior = HyperparameterPrior((σ = Gamma(2, 3), ρ = Beta(1, 1)))

# Some parameters fixed (new functionality)
hp_prior = HyperparameterPrior(
    (ρ = Beta(1, 1),),           # Free parameters
    fixed = (σ = 0.5,)           # Fixed parameters  
)

# Foundational constructor with fixed parameters
hp_prior = HyperparameterPrior{(:ρ, :τ)}(
    MvNormal([0.0, 0.0], I),     # Joint distribution for free parameters
    fixed = (σ = 0.5, μ = 0.0)   # Fixed parameters
)
```
"""
struct HyperparameterPrior{FreeNames, AllNames, D, F <: NamedTuple}
    free_distribution::D              # Distribution only over free parameters  
    fixed_values::F                   # Type-stable NamedTuple of fixed values
    name_to_index::Dict{Symbol, Int}  # Maps free parameter names to indices
    
    function HyperparameterPrior{FreeNames}(dist::D; fixed::F = NamedTuple()) where {FreeNames, D, F <: NamedTuple}
        # Validate that dimension matches
        n_free = length(FreeNames)
        if length(dist) != n_free
            error("Distribution dimension ($(length(dist))) must match number of free parameters ($n_free)")
        end
        
        # Check for duplicate names in FreeNames
        if length(unique(FreeNames)) != length(FreeNames)
            error("Duplicate free parameter names found: $FreeNames")
        end
        
        # MUST have at least one free parameter
        if isempty(FreeNames)
            error("INLA requires at least one free hyperparameter. All-fixed hyperparameter priors are not supported.")
        end
        
        # Validate no overlap between free and fixed
        free_names = Set(FreeNames)
        fixed_names = Set(keys(fixed))
        overlap = intersect(free_names, fixed_names)
        
        if !isempty(overlap)
            error("Parameters cannot be both free and fixed: $(collect(overlap))")
        end
        
        # Combine and sort all parameter names
        all_names = Tuple(sort(collect(union(free_names, fixed_names))))
        
        # Create efficient mapping for free parameters
        name_to_index = Dict(name => i for (i, name) in enumerate(FreeNames))
        
        new{FreeNames, all_names, D, F}(dist, fixed, name_to_index)
    end
end

# Constructor from NamedTuple of distributions + fixed parameters
function HyperparameterPrior(free_params::NamedTuple{FreeNames}; 
                            fixed::F = NamedTuple()) where {FreeNames, F <: NamedTuple}
    
    # MUST have at least one free parameter
    if isempty(FreeNames)
        error("INLA requires at least one free hyperparameter. All-fixed hyperparameter priors are not supported.")
    end
    
    # Validate no overlap between free and fixed
    free_names = Set(keys(free_params))
    fixed_names = Set(keys(fixed))
    overlap = intersect(free_names, fixed_names)
    
    if !isempty(overlap)
        error("Parameters cannot be both free and fixed: $(collect(overlap))")
    end
    
    # Create distribution only over free parameters
    free_dists = collect(values(free_params))
    joint_dist = product_distribution(free_dists)
    
    return HyperparameterPrior{FreeNames}(joint_dist; fixed = fixed)
end

"""
    get_hyperparameter(θ::Vector{Float64}, hp_prior::HyperparameterPrior, name::Symbol)

Extract a single hyperparameter value by name from the parameter vector.
"""
function get_hyperparameter(θ_free::Vector{Float64}, hp_prior::HyperparameterPrior, name::Symbol)
    # Check if parameter is fixed
    if name in keys(hp_prior.fixed_values)
        return hp_prior.fixed_values[name]
    end
    
    # Parameter is free - look up in θ_free vector
    if name in keys(hp_prior.name_to_index)
        idx = hp_prior.name_to_index[name]
        return θ_free[idx]
    end
    
    throw(KeyError(name))
end

"""
    set_hyperparameter!(θ::Vector{Float64}, hp_prior::HyperparameterPrior, name::Symbol, value::Float64)

Set a single hyperparameter value by name in the parameter vector.
"""
function set_hyperparameter!(θ_free::Vector{Float64}, hp_prior::HyperparameterPrior, name::Symbol, value::Float64)
    # Check if parameter is fixed
    if name in keys(hp_prior.fixed_values)
        error("Cannot set fixed parameter $name. Fixed parameters are immutable.")
    end
    
    # Parameter must be free
    if name in keys(hp_prior.name_to_index)
        idx = hp_prior.name_to_index[name]
        θ_free[idx] = value
        return θ_free
    end
    
    throw(KeyError(name))
end

"""
    to_named(θ::Vector{Float64}, hp_prior::HyperparameterPrior{Names})

Convert a hyperparameter vector to a NamedTuple with parameter names.
"""
function to_named(θ_free::Vector{Float64}, hp_prior::HyperparameterPrior{FreeNames, AllNames}) where {FreeNames, AllNames}
    # Build complete NamedTuple with both free and fixed parameters
    values = map(AllNames) do name
        if name in keys(hp_prior.fixed_values)
            hp_prior.fixed_values[name]  # Fixed value
        else
            # Free parameter - look up in θ_free vector
            idx = hp_prior.name_to_index[name]
            θ_free[idx]
        end
    end
    
    return NamedTuple{AllNames}(values)
end

"""
    to_vector(named_params::NamedTuple, hp_prior::HyperparameterPrior{Names})

Convert a NamedTuple of hyperparameters to a vector in the correct order.
All parameters in hp_prior must be provided in named_params.
"""
function to_vector(named_params::NamedTuple, hp_prior::HyperparameterPrior{FreeNames, AllNames}) where {FreeNames, AllNames}
    # Check that all required FREE parameters are provided
    provided_names = Set(keys(named_params))
    required_free_names = Set(FreeNames)
    
    missing_names = setdiff(required_free_names, provided_names)
    if !isempty(missing_names)
        throw(KeyError("Missing required free hyperparameters: $(collect(missing_names))"))
    end
    
    # Extract only FREE parameters for the vector
    θ_free = Vector{Float64}(undef, length(FreeNames))
    for (name, value) in pairs(named_params)
        if name in required_free_names
            θ_free[hp_prior.name_to_index[name]] = value
        end
        # Ignore fixed parameters and extra parameters
    end
    return θ_free
end

"""
    extract_hyperparameters(θ::Vector{Float64}, hp_prior::HyperparameterPrior, names::Tuple)

Extract a subset of hyperparameters as a NamedTuple.
"""
function extract_hyperparameters(θ::Vector{Float64}, hp_prior::HyperparameterPrior, ::Val{Names}) where {Names}
    values = ntuple(i -> get_hyperparameter(θ, hp_prior, Names[i]), Val{length(Names)}())
    return NamedTuple{Names}(values)
end

# Convenience method that takes the names tuple directly
function extract_hyperparameters(θ::Vector{Float64}, hp_prior::HyperparameterPrior, names::NTuple{N, Symbol}) where {N}
    return extract_hyperparameters(θ, hp_prior, Val{names}())
end

"""
    Base.show(io::IO, hp_prior::HyperparameterPrior{Names})

Enhanced display for HyperparameterPrior showing parameter names and distributions.
"""
function Base.show(io::IO, hp_prior::HyperparameterPrior{FreeNames, AllNames}) where {FreeNames, AllNames}
    println(io, "HyperparameterPrior with $(length(AllNames)) parameters:")
    for name in AllNames
        if name in keys(hp_prior.fixed_values)
            println(io, "  $name = $(hp_prior.fixed_values[name]) (fixed)")
        else
            # Find the distribution for this free parameter
            idx = hp_prior.name_to_index[name]
            if isa(hp_prior.free_distribution, Product)
                dist = hp_prior.free_distribution.v[idx]
                println(io, "  $name ~ $(repr(dist)) (free)")
            else
                println(io, "  $name (free, joint distribution)")
            end
        end
    end
    print(io, "Free parameters: $(length(FreeNames)), Fixed parameters: $(length(hp_prior.fixed_values))")
end
