using SparseArrays
using StatsModels
using GaussianMarkovRandomFields: predict_cols

import StatsBase: predict
export predict

"""
    predict(result::INLAResult, new_df) -> Vector{WeightedMixture}

Predict at new locations using a formula-fitted INLA model.

Takes a DataFrame (or Tables.jl-compatible object) with the same covariate columns
as the training data and returns posterior marginal distributions at each new location.

Internally builds a prediction design matrix using `predict_cols` for each formula
term (reusing the trained model's discretization, e.g. SPDE mesh), pads for the
augmented latent field, and delegates to `linear_combinations`.

# Arguments
- `result::INLAResult`: Results from `inla(formula, hp, df; family=...)`
- `new_df`: New data with covariate columns (response column not needed)

# Returns
- `Vector{WeightedMixture}`: Posterior marginal distribution at each prediction location

# Example
```julia
result = inla(@formula(y ~ 1 + Matern()(x, y_coord)), hp, df; family=Normal)

pred_df = DataFrame(x = grid_x, y_coord = grid_y)
pred_marginals = predict(result, pred_df)
mean.(pred_marginals)  # posterior means
std.(pred_marginals)   # posterior std devs
```
"""
function StatsBase.predict(result::INLAResult, new_df)
    if !haskey(result.options, :formula_random_terms)
        throw(
            ArgumentError(
                "predict(result, df) requires a formula-based model. " *
                    "Use linear_combinations(result, A) for non-formula models."
            )
        )
    end
    if result.augmentation_info === nothing
        throw(
            ArgumentError(
                "predict requires an augmented model (from formula interface). " *
                    "Use linear_combinations(result, A) instead."
            )
        )
    end

    random_terms = result.options.formula_random_terms
    fixed_terms = result.options.formula_fixed_terms

    # Get the base CombinedModel from the augmented latent prior
    base_model = result.model.latent_prior.base_model
    components = base_model.components

    # Build prediction projection blocks for each random term
    A_blocks = SparseMatrixCSC{Float64, Int}[]
    for (i, term) in enumerate(random_terms)
        A_i = predict_cols(term, components[i], new_df)
        A_i isa SparseMatrixCSC || (A_i = sparse(A_i))
        push!(A_blocks, A_i)
    end

    # Build fixed effects columns
    if !isempty(fixed_terms)
        fixed_cols = AbstractMatrix[]
        for ft in fixed_terms
            X = StatsModels.modelcols(ft, new_df)
            X isa AbstractMatrix || (X = reshape(X, :, 1))
            push!(fixed_cols, X)
        end
        push!(A_blocks, sparse(hcat(fixed_cols...)))
    end

    # Combine into prediction design matrix for base latent field
    A_pred = hcat(A_blocks...)

    # Pad with zeros for the η (linear predictor) columns in the augmented field
    # Augmented field structure: [η₁...η_n_obs; x_base₁...x_base_n_base]
    n_obs = result.augmentation_info.n_linear_predictors
    n_pred = size(A_pred, 1)
    A_full = hcat(spzeros(n_pred, n_obs), A_pred)

    return linear_combinations(result, A_full)
end
