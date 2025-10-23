using LinearAlgebra
using FiniteDiff
using Printf

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

# Arguments
- `model::INLAModel`: The INLA model
- `y`: Observed data
- `θ_star::AbstractVector`: Hyperparameter mode in natural space (as vector)

# Returns
- `ReparameterizationTransform`: Transform object containing eigendecomposition
"""
function compute_reparameterization(model::INLAModel, y, θ_star::AbstractVector)
    # Compute the positive-definite negative Hessian of the log-posterior at the mode
    H = -FiniteDiff.finite_difference_hessian(θ -> hyperparameter_logpdf(model, to_named_tuple(θ, model.hyperparameter_spec), y), θ_star)

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

# Custom show method for better user experience
function Base.show(io::IO, t::ReparameterizationTransform)
    n_dim = length(t.θ_star)
    println(io, "ReparameterizationTransform:")

    print(io, "  Mode: ")
    if n_dim <= 3
        print(io, "[", join([@sprintf("%.4f", x) for x in t.θ_star], ", "), "]")
    else
        print(io, "[", @sprintf("%.4f", t.θ_star[1]), ", ", @sprintf("%.4f", t.θ_star[2]), ", ..., ", @sprintf("%.4f", t.θ_star[end]), "]")
    end
    println(io)

    println(io, "  Dimensions: ", n_dim)

    # Show eigenvalues of the Hessian (curvature information)
    eigenvalues = 1.0 ./ (diag(t.Λ_inv_sqrt) .^ 2)
    print(io, "  Hessian eigenvalues: ")
    if n_dim <= 3
        print(io, "[", join([@sprintf("%.2e", x) for x in eigenvalues], ", "), "]")
    else
        print(io, "[", @sprintf("%.2e", eigenvalues[1]), ", ", @sprintf("%.2e", eigenvalues[2]), ", ..., ", @sprintf("%.2e", eigenvalues[end]), "]")
    end
    println(io)

    return print(io, "  Log-det Jacobian: ", @sprintf("%.4f", logdet_jacobian(t)))
end
