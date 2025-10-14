using StatsModels
using LinearAlgebra
using SparseArrays

export RandomWalkTerm
export RandomWalk

"""
    RandomWalkTerm{Order} <: RandomEffectTerm

Random effect term for random walk processes of specified order.

Represents a temporal or spatial random walk process. The precision matrix 
has a structured sparse form based on the finite difference operator.

# Type Parameters
- `Order`: The order of the random walk (1 for RW1, 2 for RW2, etc.)

# Fields  
- `variable::Symbol`: The indexing variable (e.g., `:time`, `:spatial_idx`)

# Example
```julia
# Create first-order random walk
rw1_term = RandomWalkTerm{1}(:time)

# Use in formula (via constructor function)  
@formula(y ~ x + RandomWalk(1, time))
```
"""
struct RandomWalkTerm{Order} <: RandomEffectTerm
    variable::Symbol
end

# Constructor function for formula syntax
"""
    RandomWalk(order, variable)

Constructor function for random walk effects in formula syntax.

Creates a `FunctionTerm` that gets transformed to `RandomWalkTerm{Order}` during schema application.

# Arguments  
- `order`: Order of the random walk (integer)
- `variable`: The indexing variable (symbol or term)

# Example
```julia
@formula(y ~ x + RandomWalk(1, time))
```
"""
RandomWalk(order, var) = (order, var)

# StatsModels integration

"""
    StatsModels.apply_schema(t::FunctionTerm{typeof(RandomWalk)}, schema, Mod)

Transform a `RandomWalk(order, variable)` FunctionTerm into a `RandomWalkTerm{Order}`.
"""
function StatsModels.apply_schema(
        t::StatsModels.FunctionTerm{typeof(RandomWalk)},
        schema::StatsModels.Schema,
        Mod::Type
    )
    order_term, var_term = t.args

    if isa(order_term, StatsModels.ConstantTerm)
        order = order_term.n
    else
        error("RandomWalk order must be a constant integer, got $(typeof(order_term))")
    end

    return RandomWalkTerm{order}(var_term.sym)
end

"""
    StatsModels.termvars(term::RandomWalkTerm)

Return the variables used by a RandomWalk term.
"""
StatsModels.termvars(term::RandomWalkTerm) = [term.variable]

"""
    Base.show(io::IO, term::RandomWalkTerm{Order})

Display representation for RandomWalk terms.
"""
Base.show(io::IO, term::RandomWalkTerm{Order}) where {Order} =
    print(io, "RandomWalk{$Order}($(term.variable))")

"""
    StatsModels.modelcols(rw::RandomWalkTerm, data)

Create design matrix columns for a RandomWalk term.

Returns a matrix that maps observations to time/spatial points.
"""
function StatsModels.modelcols(rw::RandomWalkTerm, data)
    # Handle both DataFrame and NamedTuple/dict-like interfaces
    if isa(data, DataFrame)
        if hasproperty(data, rw.variable)
            time_var = data[!, rw.variable]
        else
            error("Variable $(rw.variable) not found in DataFrame columns: $(names(data))")
        end
    elseif haskey(data, rw.variable)
        time_var = data[rw.variable]
    elseif hasproperty(data, rw.variable)
        time_var = getproperty(data, rw.variable)
    else
        error("Variable $(rw.variable) not found in data")
    end

    unique_times = sort(unique(time_var))
    n_obs = length(time_var)
    n_times = length(unique_times)

    # Create mapping matrix: observations → time points
    mapping_matrix = zeros(n_obs, n_times)
    for (obs_idx, time_val) in enumerate(time_var)
        time_idx = findfirst(==(time_val), unique_times)
        mapping_matrix[obs_idx, time_idx] = 1.0
    end

    return mapping_matrix
end

# GMRF construction

"""
    gmrf_block(term::RandomWalkTerm{1}, data, θ_named)

Create precision matrix block for first-order random walk.

The RW1 precision matrix is τ times the tridiagonal difference operator:
Q = τ * (2I - L - L'), where L is the lower bidiagonal shift matrix.
"""
function gmrf_block(term::RandomWalkTerm{1}, data, θ_named)
    n_times = length(unique(data[!, term.variable]))
    τ = get(θ_named, :τ_rw, 1.0)  # Default precision

    # RW1 precision: τ * tridiagonal difference matrix
    # Structure: [2 -1 0 ...; -1 2 -1 ...; 0 -1 2 ...; ...]
    D = 2 * I(n_times) - diagm(1 => ones(n_times - 1)) - diagm(-1 => ones(n_times - 1))
    return τ * D
end

"""
    gmrf_block(term::RandomWalkTerm{2}, data, θ_named)

Create precision matrix block for second-order random walk.

The RW2 precision matrix is τ times the pentadiagonal second difference operator.
"""
function gmrf_block(term::RandomWalkTerm{2}, data, θ_named)
    n_times = length(unique(data[!, term.variable]))
    τ = get(θ_named, :τ_rw, 1.0)

    if n_times < 3
        error("RW2 requires at least 3 time points, got $n_times")
    end

    # RW2 precision: τ * second difference operator
    # Structure: [1 -2 1 0 ...; -2 5 -4 1 ...; 1 -4 6 -4 1 ...; ...]
    D = zeros(n_times, n_times)

    # First row
    D[1, 1:3] = [1, -2, 1]
    # Second row
    D[2, 1:4] = [-2, 5, -4, 1]
    # Interior rows
    for i in 3:(n_times - 2)
        D[i, (i - 2):(i + 2)] = [1, -4, 6, -4, 1]
    end
    # Second-to-last row
    D[n_times - 1, (n_times - 3):n_times] = [1, -4, 5, -2]
    # Last row
    D[n_times, (n_times - 2):n_times] = [1, -2, 1]

    return τ * D
end

"""
    hyperparameters(term::RandomWalkTerm)

Return the hyperparameter names required by a RandomWalk term.
"""
hyperparameters(term::RandomWalkTerm) = [:τ_rw]
