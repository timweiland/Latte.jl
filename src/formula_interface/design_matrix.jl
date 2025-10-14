using StatsModels
using DataFrames
using LinearAlgebra
using SparseArrays

export construct_design_matrix, design_matrix_sparsity

"""
    construct_design_matrix(formula::FormulaTerm, data::DataFrame)

Construct the full sparse design matrix from a formula with arrowhead optimization.

This function implements the key design principle: **random effects first, fixed effects last**.
"""
function construct_design_matrix(formula::FormulaTerm, data::DataFrame)
    # 1. Apply schema transformation once on the entire formula
    schema = StatsModels.schema(formula, data)
    transformed_formula = StatsModels.apply_schema(formula, schema)

    # 2. Get response and predictor matrices
    resp, pred = StatsModels.modelcols(transformed_formula, data)

    # 3. Convert to sparse matrix for efficiency
    A_full = sparse(pred)

    # 4. Separate terms for later use in GMRF construction
    fixed_terms = []
    random_terms = RandomEffectTerm[]

    # Extract terms from MatrixTerm
    if isa(transformed_formula.rhs, StatsModels.MatrixTerm)
        for term in transformed_formula.rhs.terms
            if isa(term, RandomEffectTerm)
                push!(random_terms, term)
            else
                push!(fixed_terms, term)
            end
        end
    else
        # Single term case
        if isa(transformed_formula.rhs, RandomEffectTerm)
            push!(random_terms, transformed_formula.rhs)
        else
            push!(fixed_terms, transformed_formula.rhs)
        end
    end

    return A_full, (random_terms, fixed_terms), resp
end

"""
    design_matrix_sparsity(A::SparseMatrixCSC)

Compute sparsity statistics for a sparse design matrix.
"""
function design_matrix_sparsity(A::SparseMatrixCSC)
    total_elements = length(A)
    nonzero_elements = nnz(A)
    zero_elements = total_elements - nonzero_elements
    sparsity_percent = (zero_elements / total_elements) * 100

    return (
        sparsity_percent = sparsity_percent,
        nonzero_elements = nonzero_elements,
        zero_elements = zero_elements,
        total_elements = total_elements,
        size = size(A),
    )
end
