using StatsFuns

export LinkFunction, IdentityLink, LogLink, LogitLink
export apply_link, apply_invlink

"""
    LinkFunction

Abstract base type for link functions used in exponential family models.

A link function g(μ) connects the mean parameter μ of a distribution to the linear 
predictor η through the relationship g(μ) = η, or equivalently μ = g⁻¹(η).

# Implemented Link Functions
- [`IdentityLink`](@ref): g(μ) = μ (for Normal distributions)
- [`LogLink`](@ref): g(μ) = log(μ) (for Poisson distributions)  
- [`LogitLink`](@ref): g(μ) = logit(μ) (for Bernoulli/Binomial distributions)

# Interface
Concrete link functions must implement:
- `apply_link(link, μ)`: Apply the link function g(μ)
- `apply_invlink(link, η)`: Apply the inverse link function g⁻¹(η)

For performance in INLA, they should also implement:
- `derivative_invlink(link, η)`: First derivative of g⁻¹(η)
- `second_derivative_invlink(link, η)`: Second derivative of g⁻¹(η)

See also: [`ExponentialFamily`](@ref), [`apply_link`](@ref), [`apply_invlink`](@ref)
"""
abstract type LinkFunction end

"""
    IdentityLink <: LinkFunction

Identity link function: g(μ) = μ.

This is the canonical link for Normal distributions. The mean parameter μ is 
directly equal to the linear predictor η.

# Mathematical Definition
- Link: g(μ) = μ
- Inverse link: g⁻¹(η) = η  
- First derivative: d/dη g⁻¹(η) = 1
- Second derivative: d²/dη² g⁻¹(η) = 0

# Example
```julia
link = IdentityLink()
μ = apply_invlink(link, 1.5)  # μ = 1.5
η = apply_link(link, μ)       # η = 1.5
```

See also: [`LogLink`](@ref), [`LogitLink`](@ref)
"""
struct IdentityLink <: LinkFunction end

"""
    LogLink <: LinkFunction

Logarithmic link function: g(μ) = log(μ).

This is the canonical link for Poisson and Gamma distributions. It ensures the 
mean parameter μ remains positive by mapping the real-valued linear predictor η 
to μ = exp(η).

# Mathematical Definition
- Link: g(μ) = log(μ) 
- Inverse link: g⁻¹(η) = exp(η)
- First derivative: d/dη g⁻¹(η) = exp(η)
- Second derivative: d²/dη² g⁻¹(η) = exp(η)

# Example
```julia
link = LogLink()
μ = apply_invlink(link, 1.0)  # μ = exp(1.0) ≈ 2.718
η = apply_link(link, μ)       # η = log(μ) = 1.0
```

See also: [`IdentityLink`](@ref), [`LogitLink`](@ref)
"""
struct LogLink <: LinkFunction end

"""
    LogitLink <: LinkFunction

Logit link function: g(μ) = logit(μ) = log(μ/(1-μ)).

This is the canonical link for Bernoulli and Binomial distributions. It maps 
probabilities μ ∈ (0,1) to the real line via the logistic transformation, 
ensuring μ = logistic(η) = 1/(1+exp(-η)) remains a valid probability.

# Mathematical Definition
- Link: g(μ) = logit(μ) = log(μ/(1-μ))
- Inverse link: g⁻¹(η) = logistic(η) = 1/(1+exp(-η))
- First derivative: d/dη g⁻¹(η) = μ(1-μ) where μ = logistic(η)
- Second derivative: d²/dη² g⁻¹(η) = μ(1-μ)(1-2μ)

# Example
```julia
link = LogitLink()
μ = apply_invlink(link, 0.0)  # μ = logistic(0.0) = 0.5
η = apply_link(link, μ)       # η = logit(0.5) = 0.0
```

See also: [`IdentityLink`](@ref), [`LogLink`](@ref)
"""
struct LogitLink <: LinkFunction end

"""
    apply_link(link::LinkFunction, μ) -> Real

Apply the link function g(μ) to transform mean parameters to linear predictor scale.

This function computes η = g(μ), where g is the link function. This transformation
is typically used to ensure the mean parameter satisfies appropriate constraints
(e.g., positivity for Poisson, probability bounds for Bernoulli).

# Arguments
- `link`: A link function (IdentityLink, LogLink, or LogitLink)
- `μ`: Mean parameter value(s) in the natural parameter space

# Returns
The transformed value(s) η on the linear predictor scale

# Examples
```julia
apply_link(LogLink(), 2.718)      # ≈ 1.0
apply_link(LogitLink(), 0.5)      # = 0.0  
apply_link(IdentityLink(), 1.5)   # = 1.5
```

See also: [`apply_invlink`](@ref), [`LinkFunction`](@ref)
"""
apply_link(::IdentityLink, x) = x
apply_link(::LogLink, x) = log(x)
apply_link(::LogitLink, x) = logit(x)

"""
    apply_invlink(link::LinkFunction, η) -> Real

Apply the inverse link function g⁻¹(η) to transform linear predictor to mean parameters.

This function computes μ = g⁻¹(η), where g⁻¹ is the inverse link function. This is
the primary transformation used in INLA to convert the latent field values to the
natural parameter space of the observation distribution.

# Arguments
- `link`: A link function (IdentityLink, LogLink, or LogitLink)
- `η`: Linear predictor value(s)

# Returns
The transformed value(s) μ in the natural parameter space

# Examples
```julia
apply_invlink(LogLink(), 1.0)      # ≈ 2.718 (= exp(1))
apply_invlink(LogitLink(), 0.0)    # = 0.5   (= logistic(0))
apply_invlink(IdentityLink(), 1.5) # = 1.5
```

See also: [`apply_link`](@ref), [`LinkFunction`](@ref)
"""
apply_invlink(::IdentityLink, x) = x
apply_invlink(::LogLink, x) = exp(x)
apply_invlink(::LogitLink, x) = logistic(x)

derivative_invlink(::IdentityLink, x) = one(x)
derivative_invlink(::LogLink, x) = exp(x)
function derivative_invlink(::LogitLink, x)
    p = logistic(x)
    return p * (one(p) - p)
end

second_derivative_invlink(::IdentityLink, x) = zero(x)
second_derivative_invlink(::LogLink, x) = exp(x)
function second_derivative_invlink(::LogitLink, x)
    p = logistic(x)
    return p * (one(p) - p) * (one(p) - 2 * p)
end
