"""
    AugmentationInfo

Metadata about latent field augmentation, stored in LatentGaussianModel for models that use
AugmentedLatentModel with LinearlyTransformedObservationModel.

# Fields
- `n_linear_predictors::Int`: Number of linear predictor components
- `n_base_latent::Int`: Number of base latent components
- `linear_predictor_indices::UnitRange{Int}`: Indices for η in augmented field (1:n_obs)
- `base_latent_indices::UnitRange{Int}`: Indices for x_base in augmented field

# Example
```julia
info = AugmentationInfo(200, 100)  # 200 linear predictors, 100 base components
info.linear_predictor_indices  # 1:200
info.base_latent_indices       # 201:300
```
"""
struct AugmentationInfo
    n_linear_predictors::Int
    n_base_latent::Int
    linear_predictor_indices::UnitRange{Int}
    base_latent_indices::UnitRange{Int}

    function AugmentationInfo(n_linear_predictors::Int, n_base_latent::Int)
        n_linear_predictors > 0 || throw(ArgumentError("n_linear_predictors must be positive"))
        n_base_latent > 0 || throw(ArgumentError("n_base_latent must be positive"))

        linear_predictor_indices = 1:n_linear_predictors
        base_latent_indices = (n_linear_predictors + 1):(n_linear_predictors + n_base_latent)

        return new(
            n_linear_predictors, n_base_latent,
            linear_predictor_indices, base_latent_indices
        )
    end
end

function Base.show(io::IO, info::AugmentationInfo)
    return print(io, "AugmentationInfo($(info.n_linear_predictors) linear predictors, $(info.n_base_latent) base components)")
end
