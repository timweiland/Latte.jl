using ForwardDiff
using LinearAlgebra
using GaussianMarkovRandomFields
using Distributions

export loghessian_directional_derivative

"""
    loghessian_directional_derivative(x0, v, obs_lik::ObservationLikelihood)

Compute the directional derivative of the Hessian matrix of the observation
negative log-likelihood.

Given a Hessian matrix H(x) = -∇²[log p(y|x)], this function computes:
    d/dt[H(x₀ + tv)]|_{t=0}

This quantity is needed for computing third-order corrections in the Simplified
Laplace Approximation for INLA.

# Arguments
- `x0::Vector`: Point at which to evaluate the derivative
- `v::Vector`: Direction vector for the directional derivative
- `obs_lik::ObservationLikelihood`: Materialized observation likelihood (contains data and hyperparameters)

# Returns
- Matrix (often `Diagonal` for exponential families): The directional derivative of the Hessian

# Mathematical Details
For exponential family observation models with diagonal Hessians, the result is:
    diag(h'''(x₀₁)·v₁, h'''(x₀₂)·v₂, ..., h'''(x₀ₙ)·vₙ)

where h'''(x₀ᵢ) is the third derivative of the negative log-likelihood -log p(yᵢ|xᵢ).

# Examples
```julia
# Poisson observation model
obs_model = ExponentialFamily(Poisson)
y = [3, 5, 2, 4]
θ = NamedTuple()
obs_lik = obs_model(y; θ...)

x0 = [1.0, 1.5, 0.8, 1.2]
v = ones(4)
dH = loghessian_directional_derivative(x0, v, obs_lik)
```

# Implementation
The default fallback uses automatic differentiation via ForwardDiff. Specialized
implementations are provided for common exponential family distributions to
improve performance and numerical stability.
"""
function loghessian_directional_derivative(x0::AbstractVector, v::AbstractVector, obs_lik::ObservationLikelihood)
    # Fallback implementation using automatic differentiation
    # This creates a function that maps t → H(x₀ + tv)
    loghessian_path = t -> loghessian(x0 + t * v, obs_lik)

    # Compute the derivative at t=0
    return ForwardDiff.derivative(loghessian_path, 0.0)
end

# ============================================================================
# Specialized implementations for exponential family likelihoods
# ============================================================================

"""
    loghessian_directional_derivative(x0, v, obs_lik::PoissonLikelihood)

Specialized implementation for Poisson observation model with log link.

For Poisson: -log p(y|x) = -y·x + exp(x) + const
- Hessian entry: h''(x) = exp(x)
- Third derivative: h'''(x) = exp(x)
"""
function loghessian_directional_derivative(
        x0::AbstractVector, v::AbstractVector,
        obs_lik::PoissonLikelihood{LogLink}
    )
    # For Poisson: log p(y|x) = y*x - exp(x) - log(y!)
    # Hessian: ∇²[log p(y|x)] = -exp(x)
    # Directional derivative: d/dt[-exp(x + tv)]|_{t=0} = -exp(x) * v
    third_deriv_values = -exp.(x0) .* v
    return Diagonal(third_deriv_values)
end

"""
    loghessian_directional_derivative(x0, v, obs_lik::BernoulliLikelihood)

Specialized implementation for Bernoulli observation model with logit link.

For Bernoulli: -log p(y|x) = -y·x + log(1 + exp(x))
- Let p = sigmoid(x) = 1/(1 + exp(-x))
- Hessian entry: h''(x) = p(1-p)
- Third derivative: h'''(x) = p(1-p)(1-2p)
"""
function loghessian_directional_derivative(
        x0::AbstractVector, v::AbstractVector,
        obs_lik::BernoulliLikelihood{LogitLink}
    )
    # Compute third derivative values element-wise
    third_deriv_values = similar(x0)

    for i in eachindex(x0)
        # Compute sigmoid in a numerically stable way
        x = x0[i]
        if x >= 0
            exp_neg_x = exp(-x)
            p = 1 / (1 + exp_neg_x)
        else
            exp_x = exp(x)
            p = exp_x / (1 + exp_x)
        end

        # Hessian of log p(y|x) has third derivative: -(1-2p)*p*(1-p)
        # So directional derivative is: -(1-2p)*p*(1-p) * v
        third_deriv_values[i] = -(1 - 2p) * p * (1 - p) * v[i]
    end

    return Diagonal(third_deriv_values)
end

"""
    loghessian_directional_derivative(x0, v, obs_lik::BinomialLikelihood)

Specialized implementation for Binomial observation model with logit link.

For Binomial with n trials:
- Third derivative: h'''(x) = n · p(1-p)(1-2p)
where p = sigmoid(x)
"""
function loghessian_directional_derivative(
        x0::AbstractVector, v::AbstractVector,
        obs_lik::BinomialLikelihood{LogitLink}
    )
    # Compute third derivative values element-wise
    third_deriv_values = similar(x0)

    for i in eachindex(x0)
        # Compute sigmoid in a numerically stable way
        x = x0[i]
        if x >= 0
            exp_neg_x = exp(-x)
            p = 1 / (1 + exp_neg_x)
        else
            exp_x = exp(x)
            p = exp_x / (1 + exp_x)
        end

        # Get number of trials for this observation
        n_trials = obs_lik.n[i]

        # Directional derivative: -n * (1-2p) * p * (1-p) * v
        third_deriv_values[i] = -n_trials * (1 - 2p) * p * (1 - p) * v[i]
    end

    return Diagonal(third_deriv_values)
end

"""
    loghessian_directional_derivative(x0, v, obs_lik::NormalLikelihood)

Specialized implementation for Normal (Gaussian) observation model with identity link.

For Normal: -log p(y|x) = (y-x)²/(2σ²) + const
- Hessian entry: h''(x) = 1/σ²
- Third derivative: h'''(x) = 0

The directional derivative is zero since the Hessian is constant.
"""
function loghessian_directional_derivative(
        x0::AbstractVector, v::AbstractVector,
        obs_lik::NormalLikelihood{IdentityLink}
    )
    # For Normal with identity link, the Hessian is constant
    # So its derivative is zero
    n = length(x0)
    return Diagonal(zeros(n))
end

"""
    loghessian_directional_derivative(x0, v, obs_lik::LinearlyTransformedLikelihood)

Specialized implementation for linearly transformed observation models.

If the base model has Hessian H_base, and the transformation is η = Ax,
then by the chain rule:
    H(x) = A^T H_base(Ax) A

The directional derivative is:
    dH/dt|_{t=0} = A^T (dH_base/dt|_{t=0}) A
where dH_base/dt is evaluated at Ax₀ in direction Av.

This recursively calls loghessian_directional_derivative on the base likelihood.
"""
function loghessian_directional_derivative(
        x0::AbstractVector, v::AbstractVector,
        obs_lik::LinearlyTransformedLikelihood
    )
    # Extract the design matrix A and base likelihood
    A = obs_lik.design_matrix
    base_lik = obs_lik.base_likelihood

    # Transform the point and direction to the base space
    x0_base = A * x0
    v_base = A * v

    # Compute the directional derivative in the base space
    dH_base = loghessian_directional_derivative(x0_base, v_base, base_lik)

    # Transform back: dH = A^T * dH_base * A
    return A' * dH_base * A
end
