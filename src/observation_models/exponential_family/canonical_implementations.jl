using LinearAlgebra
using StatsFuns
using Distributions
using Distributions: product_distribution

# =======================================================================================
# EXPONENTIAL FAMILY IMPLEMENTATIONS: Generic loglik + specialized gradients/hessians
# =======================================================================================

# ----------------------------- Generic loglik using product_distribution --------------------------

"""
    loglik(lik::ExponentialFamilyLikelihood, x) -> Float64

Generic loglik implementation for all exponential family likelihoods using product_distribution.
"""
function loglik(lik::ExponentialFamilyLikelihood, x)
    y = lik.y
    η = x
    μ = apply_invlink.(Ref(lik.link), η)
    dist = _construct_distribution(lik, μ)
    return logpdf(dist, y)
end

"""
    loglik(lik::NormalLikelihood, x) -> Float64

Specialized fast implementation for Normal likelihood that avoids product_distribution overhead.

Computes: ∑ᵢ logpdf(Normal(μᵢ, σ), yᵢ) = -n/2 * log(2π) - n * log(σ) - 1/(2σ²) * ∑ᵢ(yᵢ - μᵢ)²
"""
function loglik(lik::NormalLikelihood, x)
    y = lik.y
    η = x
    μ = apply_invlink.(Ref(lik.link), η)

    # Fast computation avoiding product_distribution
    n = length(y)
    residuals = y .- μ
    sum_sq_residuals = sum(abs2, residuals)

    # -n/2 * log(2π) - n * log(σ) - 1/(2σ²) * ∑(yᵢ - μᵢ)²
    return -0.5 * n * log(2π) - n * lik.log_σ - 0.5 * lik.inv_σ² * sum_sq_residuals
end

# Family-specific distribution construction
function _construct_distribution(lik::NormalLikelihood, μ)
    return product_distribution(Normal.(μ, lik.σ))
end

function _construct_distribution(lik::PoissonLikelihood, μ)
    return product_distribution(Poisson.(μ))
end

function _construct_distribution(lik::BernoulliLikelihood, μ)
    return product_distribution(Bernoulli.(μ))
end

function _construct_distribution(lik::BinomialLikelihood, μ)
    return product_distribution(Binomial.(lik.n, μ))
end

# ----------------------------- loggrad methods for canonical links --------------------------

"""
    loggrad(lik::NormalLikelihood{IdentityLink}, x) -> Vector{Float64}

Compute gradient of Normal likelihood with canonical identity link w.r.t. latent field x.
"""
function loggrad(lik::NormalLikelihood{IdentityLink}, x)
    y = lik.y
    μ = x  # Canonical identity link: μ = x
    return (y .- μ) .* lik.inv_σ²
end

"""
    loggrad(lik::PoissonLikelihood{LogLink}, x) -> Vector{Float64}

Compute gradient of Poisson likelihood with canonical log link w.r.t. latent field x.
"""
function loggrad(lik::PoissonLikelihood{LogLink}, x)
    y = lik.y
    η = x
    μ = exp.(η)  # Canonical log link: μ = exp(η)
    return y .- μ
end

"""
    loggrad(lik::BernoulliLikelihood{LogitLink}, x) -> Vector{Float64}

Compute gradient of Bernoulli likelihood with canonical logit link w.r.t. latent field x.
"""
function loggrad(lik::BernoulliLikelihood{LogitLink}, x)
    y = lik.y
    η = x
    μ = logistic.(η)  # Canonical logit link: μ = logistic(η)
    return y .- μ
end

"""
    loggrad(lik::BinomialLikelihood{LogitLink}, x) -> Vector{Float64}

Compute gradient of Binomial likelihood with canonical logit link w.r.t. latent field x.
"""
function loggrad(lik::BinomialLikelihood{LogitLink}, x)
    y = lik.y
    n = lik.n
    η = x
    μ = logistic.(η)  # Canonical logit link: μ = logistic(η)
    return y .- n .* μ
end

# ----------------------------- loghessian methods for canonical links --------------------------

"""
    loghessian(lik::NormalLikelihood{IdentityLink}, x) -> Diagonal{Float64}

Compute Hessian of Normal likelihood with canonical identity link w.r.t. latent field x.
"""
function loghessian(lik::NormalLikelihood{IdentityLink}, x)
    return Diagonal(-ones(length(x)) .* lik.inv_σ²)
end

"""
    loghessian(lik::PoissonLikelihood{LogLink}, x) -> Diagonal{Float64}

Compute Hessian of Poisson likelihood with canonical log link w.r.t. latent field x.
"""
function loghessian(lik::PoissonLikelihood{LogLink}, x)
    η = x
    μ = exp.(η)  # Canonical log link: μ = exp(η)
    return Diagonal(-μ)
end

"""
    loghessian(lik::BernoulliLikelihood{LogitLink}, x) -> Diagonal{Float64}

Compute Hessian of Bernoulli likelihood with canonical logit link w.r.t. latent field x.
"""
function loghessian(lik::BernoulliLikelihood{LogitLink}, x)
    η = x
    μ = logistic.(η)  # Canonical logit link: μ = logistic(η)
    return Diagonal(-μ .* (1 .- μ))
end

"""
    loghessian(lik::BinomialLikelihood{LogitLink}, x) -> Diagonal{Float64}

Compute Hessian of Binomial likelihood with canonical logit link w.r.t. latent field x.
"""
function loghessian(lik::BinomialLikelihood{LogitLink}, x)
    n = lik.n
    η = x
    μ = logistic.(η)  # Canonical logit link: μ = logistic(η)
    return Diagonal(-n .* μ .* (1 .- μ))
end
