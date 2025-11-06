export MarginalLogLikelihoodAccumulator

"""
    MarginalLogLikelihoodAccumulator()

Extract marginal log-likelihood log p(y) from INLA exploration.

The marginal likelihood (model evidence) is computed during INLA as the
normalization constant of the hyperparameter posterior. Used for model
comparison via Bayes factors.

# Fields (after finalize!)
- `log_marginal_likelihood::Float64`: log p(y)

# Usage
Compare models via Bayes factor: BF = p(y|M₁) / p(y|M₂) = exp(log p(y|M₁) - log p(y|M₂))
- BF > 10: Strong evidence for M₁
- BF > 100: Decisive evidence for M₁
"""
mutable struct MarginalLogLikelihoodAccumulator <: PosteriorAccumulator
    log_marginal_likelihood::Float64

    MarginalLogLikelihoodAccumulator() = new(0.0)
end

# No per-point accumulation needed
function accumulate!(acc::MarginalLogLikelihoodAccumulator; kwargs...)
    return nothing
end

# Extract from exploration
function finalize!(acc::MarginalLogLikelihoodAccumulator, exploration::HyperparameterExploration)
    acc.log_marginal_likelihood = exploration.log_normalization_constant
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", acc::MarginalLogLikelihoodAccumulator)
    println(io, "Marginal Log-Likelihood:")
    return println(io, "  log p(y): ", round(acc.log_marginal_likelihood, digits = 2))
end
