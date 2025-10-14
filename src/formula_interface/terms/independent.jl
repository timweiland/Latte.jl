using StatsModels
using LinearAlgebra
using SparseArrays

export IndependentTerm
export Independent

"""
    IndependentTerm <: RandomEffectTerm

Random effect term for independent (IID) random effects.

Represents a collection of independent random variables, typically used for
group-specific intercepts or slopes. The precision matrix is diagonal.

# Fields
- `variable::Symbol`: The grouping variable (e.g., `:hospital`, `:subject_id`)

# Example
```julia
# Create term for independent group effects
iid_term = IndependentTerm(:hospital)

# Use in formula (via constructor function)
@formula(y ~ x + Independent(hospital))
```

# Mathematical Model
For a grouping variable with G groups, creates G independent random effects:
- u_g ~ N(0, 1/τ) for g = 1, ..., G  
- Prior precision matrix: Q = τ * I_G (diagonal)
- Design matrix: indicator matrix mapping observations to groups
"""
struct IndependentTerm <: RandomEffectTerm
    variable::Symbol
end

# Constructor function for formula syntax
"""
    Independent(variable)

Constructor function for independent random effects in formula syntax.

Creates a `FunctionTerm` that gets transformed to `IndependentTerm` during schema application.

# Arguments
- `variable`: The grouping variable (symbol or term)

# Example
```julia
@formula(y ~ x + Independent(hospital))
@formula(y ~ x + Independent(subject_id))
```
"""
Independent(var) = var

# StatsModels integration

"""
    StatsModels.apply_schema(t::FunctionTerm{typeof(Independent)}, schema, Mod)

Transform an `Independent(variable)` FunctionTerm into an `IndependentTerm`.
"""
function StatsModels.apply_schema(
        t::StatsModels.FunctionTerm{typeof(Independent)},
        schema::StatsModels.Schema,
        Mod::Type
    )
    # Extract the variable from the function term
    var_term = only(t.args)  # Should be one variable
    return IndependentTerm(var_term.sym)
end

"""
    StatsModels.termvars(term::IndependentTerm)

Return the variables used by an Independent term.
"""
StatsModels.termvars(term::IndependentTerm) = [term.variable]

"""
    Base.show(io::IO, term::IndependentTerm)

Display representation for Independent terms.
"""
Base.show(io::IO, term::IndependentTerm) =
    print(io, "Independent($(term.variable))")

"""
    StatsModels.modelcols(iid::IndependentTerm, data)

Create design matrix columns for an Independent term.

Returns an indicator matrix that maps observations to groups.
For a dataset with `n` observations and `g` unique groups, returns an
`n × g` matrix where `A[i,j] = 1` if observation `i` belongs to group `j`.

# Arguments
- `iid::IndependentTerm`: The independent effects term
- `data`: DataFrame containing the data

# Returns  
- `Matrix`: Design matrix mapping observations to group-specific random effects
"""
function StatsModels.modelcols(iid::IndependentTerm, data)
    # Handle both DataFrame and NamedTuple/dict-like interfaces
    if isa(data, DataFrame)
        if hasproperty(data, iid.variable)
            grouping_var = data[!, iid.variable]
        else
            error("Variable $(iid.variable) not found in DataFrame columns: $(names(data))")
        end
    elseif haskey(data, iid.variable)
        grouping_var = data[iid.variable]
    elseif hasproperty(data, iid.variable)
        grouping_var = getproperty(data, iid.variable)
    else
        error("Variable $(iid.variable) not found in data")
    end

    unique_groups = sort(unique(grouping_var))
    n_obs = length(grouping_var)
    n_groups = length(unique_groups)

    # Create indicator matrix
    indicator_matrix = zeros(n_obs, n_groups)
    for (obs_idx, group_val) in enumerate(grouping_var)
        group_idx = findfirst(==(group_val), unique_groups)
        if group_idx === nothing
            error("Group value $group_val not found in unique groups")
        end
        indicator_matrix[obs_idx, group_idx] = 1.0
    end

    return indicator_matrix
end

# GMRF construction

"""
    gmrf_block(term::IndependentTerm, data, θ_named)

Create precision matrix block for independent random effects.

The IID precision matrix is τ times the identity matrix:
Q = τ * I_G where G is the number of groups.

# Arguments
- `term::IndependentTerm`: The independent effects term
- `data`: DataFrame containing the grouping variable
- `θ_named::Dict`: Named hyperparameters

# Returns
- `Matrix`: Diagonal precision matrix for independent effects
"""
function gmrf_block(term::IndependentTerm, data, θ_named)
    # Count unique groups
    if isa(data, DataFrame)
        grouping_var = data[!, term.variable]
    else
        grouping_var = data[term.variable]
    end

    n_groups = length(unique(grouping_var))
    τ = get(θ_named, :τ_iid, 1.0)  # Default precision

    # IID precision: τ * I (diagonal matrix)
    return τ * I(n_groups)
end

"""
    hyperparameters(term::IndependentTerm)

Return the hyperparameter names required by an Independent term.

Returns `[:τ_iid]` - the precision parameter for the independent effects.
"""
hyperparameters(term::IndependentTerm) = [:τ_iid]
