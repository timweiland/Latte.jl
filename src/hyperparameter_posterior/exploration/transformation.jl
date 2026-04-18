using LinearAlgebra
using Printf
using GaussianMarkovRandomFields: with_workspace

export ReparameterizationTransform, logdet_jacobian, compute_reparameterization

"""
    ReparameterizationTransform{W}

A struct that encapsulates the transformation from the standardized space (z) to
the original hyperparameter space (θ).

The transformation is defined as: θ(z) = θ_star + V * Λ_inv_sqrt * z.
It is callable, so you can apply it like a function: `θ = transform(z)` which returns `WorkingHyperparameters`.

# Type Parameters
- `W <: WorkingHyperparameters`: Type of working hyperparameters
"""
struct ReparameterizationTransform{W <: WorkingHyperparameters, HT <: AbstractMatrix}
    θ_star::W
    V::Matrix{Float64}
    Λ_inv_sqrt::Diagonal{Float64, Vector{Float64}}
    H::HT  # Storing the positive-definite negative Hessian
end

"""
    (t::ReparameterizationTransform)(z::AbstractVector{<:Real})

Applies the forward transformation z -> θ, making the struct callable.
Returns `WorkingHyperparameters` via broadcasting.
"""
function (t::ReparameterizationTransform)(z::AbstractVector{<:Real})
    return t.θ_star .+ t.V * t.Λ_inv_sqrt * z
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
- `θ_star::WorkingHyperparameters`: Hyperparameter mode in working space

# Returns
- `ReparameterizationTransform`: Transform object containing eigendecomposition
"""
function compute_reparameterization(
        model::INLAModel, y, θ_star::WorkingHyperparameters;
        pool,
        executor::ParallelExecutor = SequentialExecutor(),
        diff_strategy::DifferentiationStrategy = ADStrategy(),
    )
    # Each Hessian evaluation checks out a workspace from the pool. For
    # SequentialExecutor the pool has size 1 and the single workspace is
    # checked out + returned on every call. Under ThreadedExecutor the pool
    # is sized to `nworkers`, so concurrent tasks each get their own
    # workspace without racing.
    logpdf_fn = θ_vec -> with_workspace(pool) do ws
        try
            hyperparameter_logpdf(model, WorkingHyperparameters(θ_vec, θ_star.spec), y; ws = ws)
        catch
            -Inf
        end
    end

    H = _compute_negative_hessian(diff_strategy, logpdf_fn, θ_star.θ; executor = executor)
    eigen_result = eigen(H)

    # Clamp non-positive eigenvalues as a fallback
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
    θ_vec = t.θ_star.θ
    if n_dim <= 3
        print(io, "[", join([@sprintf("%.4f", x) for x in θ_vec], ", "), "]")
    else
        print(io, "[", @sprintf("%.4f", θ_vec[1]), ", ", @sprintf("%.4f", θ_vec[2]), ", ..., ", @sprintf("%.4f", θ_vec[end]), "]")
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
