"""
Evaluation methods for CompositeLikelihood.

These methods implement the core mathematical operations by summing contributions
from all component likelihoods.
"""

"""
    loglik(composite_lik::CompositeLikelihood, x) -> Float64

Compute the log-likelihood of a composite likelihood by summing component contributions.

Each component likelihood receives the full latent field `x` and contributes to the total
log-likelihood. This handles cases where components may have overlapping dependencies
on the latent field.
"""
function loglik(composite_lik::CompositeLikelihood, x)
    return sum(comp -> loglik(comp, x), composite_lik.components)
end

"""
    loggrad(composite_lik::CompositeLikelihood, x) -> Vector{Float64}

Compute the gradient of the log-likelihood by summing component gradients.

Each component contributes its gradient with respect to the full latent field `x`.
For overlapping dependencies, gradients are automatically summed at each latent field element.
"""
function loggrad(composite_lik::CompositeLikelihood, x)
    # Start with zero gradient
    grad = zeros(eltype(x), length(x))

    # Sum contributions from each component
    for component in composite_lik.components
        grad .+= loggrad(component, x)
    end

    return grad
end

"""
    loghessian(composite_lik::CompositeLikelihood, x) -> AbstractMatrix{Float64}

Compute the Hessian of the log-likelihood by summing component Hessians.

Each component contributes its Hessian with respect to the full latent field `x`.
For overlapping dependencies, Hessians are automatically summed element-wise.
"""
function loghessian(composite_lik::CompositeLikelihood, x)
    # Start with zero Hessian - let first component determine type/structure
    first_hess = loghessian(composite_lik.components[1], x)
    total_hess = copy(first_hess)

    # Sum contributions from remaining components
    for i in 2:length(composite_lik.components)
        total_hess .+= loghessian(composite_lik.components[i], x)
    end

    return total_hess
end
