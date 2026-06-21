"""
    validate_inla_inputs(model::LatentGaussianModel, y::AbstractVector, latent_indices)

Validate inputs for INLA inference.

# Arguments
- `model::LatentGaussianModel`: The INLA model
- `y::AbstractVector`: Observed data (already filtered if prediction)
- `latent_indices::Union{Nothing, AbstractVector{<:Integer}}`: Indices of latent variables to marginalize

# Throws
- `ArgumentError`: If inputs are invalid
"""
function validate_inla_inputs(model::LatentGaussianModel, y::AbstractVector, latent_indices::Union{Nothing, AbstractVector{<:Integer}})
    # Check that y is not empty
    if length(y) == 0
        throw(ArgumentError("Observed data y cannot be empty"))
    end

    # Check latent_indices if provided
    return if latent_indices !== nothing
        if length(latent_indices) == 0
            throw(ArgumentError("latent_indices cannot be empty"))
        end

        n_latent = length(model.latent_prior)
        if any(i -> i < 1 || i > n_latent, latent_indices)
            throw(ArgumentError("latent_indices must be between 1 and $(n_latent)"))
        end

        # Check for duplicates
        if length(unique(latent_indices)) != length(latent_indices)
            throw(ArgumentError("latent_indices cannot contain duplicates"))
        end
    end
end
