using LinearAlgebra
using FiniteDiff

export ReparameterizationTransform, logdet_jacobian, compute_reparameterization

"""
    ReparameterizationTransform

A struct that encapsulates the transformation from the standardized space (z) to
the original hyperparameter space (θ).

The transformation is defined as: θ(z) = θ_star + V * Λ_inv_sqrt * z.
It is callable, so you can apply it like a function: `θ = transform(z)`.
"""
struct ReparameterizationTransform
    θ_star::Vector{Float64}
    V::Matrix{Float64}
    Λ_inv_sqrt::Diagonal{Float64, Vector{Float64}}
    H::Matrix{Float64}  # Storing the positive-definite negative Hessian
end

"""
    (t::ReparameterizationTransform)(z::AbstractVector{<:Real})

Applies the forward transformation z -> θ, making the struct callable.
"""
function (t::ReparameterizationTransform)(z::AbstractVector{<:Real})
    return t.θ_star + t.V * t.Λ_inv_sqrt * z
end

"""
    logdet_jacobian(t::ReparameterizationTransform)

Calculates the log-determinant of the Jacobian for the transformation z -> θ.
This is a crucial component for correctly scaling the integration volume.
"""
function logdet_jacobian(t::ReparameterizationTransform)
    return logdet(t.Λ_inv_sqrt)
end

"""
    compute_reparameterization(model::INLAModel, y, θ_star)

Computes the reparameterization around the mode and returns it as a
`ReparameterizationTransform` object.
"""
function compute_reparameterization(model::INLAModel, y, θ_star)
    # Compute the positive-definite negative Hessian of the log-posterior at the mode
    H = -FiniteDiff.finite_difference_hessian(θ -> hyperparameter_logpdf(model, θ, y), θ_star)

    eigen_result = eigen(H)

    # Regularize for numerical stability if Hessian is not perfectly positive-definite
    if any(eigen_result.values .<= 0)
        @warn "Hessian at the mode is not positive definite. Regularizing eigenvalues."
        Λ = max.(eigen_result.values, 1.0e-6)
    else
        Λ = eigen_result.values
    end

    V = eigen_result.vectors
    Λ_inv_sqrt = Diagonal(1.0 ./ sqrt.(Λ))

    return ReparameterizationTransform(θ_star, V, Λ_inv_sqrt, H)
end
