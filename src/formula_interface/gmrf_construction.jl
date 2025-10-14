using LinearAlgebra
using SparseArrays
using DataFrames

export construct_gmrf_precision, collect_hyperparameters, validate_hyperparameters

"""
    construct_gmrf_precision(random_terms, fixed_terms, data, θ_named)

Construct block diagonal prior precision matrix from formula terms.

This function builds the prior precision matrix Q₀ in block diagonal form:
- Fixed effects get weak precision (flat priors)
- Each random effect term contributes one precision block via `gmrf_block`

The block diagonal structure ensures computational efficiency and maintains
the structured sparsity needed for fast factorization.

# Arguments
- `random_terms::Vector{RandomEffectTerm}`: Random effect terms from formula
- `fixed_terms::Vector`: Fixed effect terms from formula  
- `data::DataFrame`: Data containing all referenced variables
- `θ_named::Dict`: Named hyperparameters (e.g., Dict(:τ_rw => 1.5))

# Returns
- `SparseMatrixCSC`: Block diagonal prior precision matrix Q₀

# Example
```julia
f = @formula(y ~ x + RandomWalk(1, time))
A, terms, resp = construct_design_matrix(f, data)
random_terms, fixed_terms = terms

θ = Dict(:τ_rw => 2.0)
Q = construct_gmrf_precision(random_terms, fixed_terms, data, θ)
```
"""
function construct_gmrf_precision(random_terms, fixed_terms, data, θ_named)
    # Count dimensions for each block
    n_fixed = length(fixed_terms)
    if n_fixed == 0
        n_fixed = 1  # Always have at least intercept
    end

    # Get random effect block sizes
    random_block_sizes = Int[]
    for term in random_terms
        # Get the size by creating a test design matrix
        test_cols = StatsModels.modelcols(term, data)
        push!(random_block_sizes, size(test_cols, 2))
    end

    n_random = sum(random_block_sizes)
    n_total = n_fixed + n_random

    # Build precision blocks
    precision_blocks = []

    # 1. Random effects blocks first (arrowhead optimization)
    for term in random_terms
        block = gmrf_block(term, data, θ_named)
        push!(precision_blocks, sparse(block))
    end

    # 2. Fixed effects block last (weak precision for flat priors)
    weak_precision = 1.0e-6  # R-INLA style weak precision
    Q_fixed = weak_precision * I(n_fixed)
    push!(precision_blocks, sparse(Q_fixed))

    # 3. Combine into block diagonal matrix
    Q_full = blockdiag(precision_blocks...)

    return Q_full
end

"""
    collect_hyperparameters(random_terms, fixed_terms)

Collect all required hyperparameter names from formula terms.

Returns a vector of symbols representing all hyperparameters needed
by the random and fixed effect terms in the formula.
"""
function collect_hyperparameters(random_terms, fixed_terms)
    params = Symbol[]

    # Collect from random effects
    for term in random_terms
        term_params = hyperparameters(term)
        append!(params, term_params)
    end

    # Fixed effects typically need mean and precision parameters
    # But we'll use weak priors for now, so no additional parameters needed

    return unique(params)
end

"""
    validate_hyperparameters(θ_named, required_params)

Validate that all required hyperparameters are provided.

Throws an error if any required parameters are missing from θ_named.
"""
function validate_hyperparameters(θ_named, required_params)
    missing_params = Symbol[]

    for param in required_params
        if !haskey(θ_named, param)
            push!(missing_params, param)
        end
    end

    if !isempty(missing_params)
        error("Missing required hyperparameters: $missing_params")
    end

    return true
end
