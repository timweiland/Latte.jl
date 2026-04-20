using Bijectors
using Distributions
using GaussianMarkovRandomFields: LinkFunction, LogLink, LogitLink, IdentityLink

export get_bijector

"""
    get_bijector(link::LinkFunction)

Maps a `LinkFunction` from GaussianMarkovRandomFields.jl to the corresponding
`Bijector` representing the **link function** (the transformation from observation
space to linear predictor space).

# Mathematical Relationship
For a link function g:
- Link: η = g(μ)  (maps observations to linear predictors)
- Inverse link: μ = g⁻¹(η)  (maps linear predictors to observations)

This function returns a bijector `b` such that `b(μ) = g(μ)`.
To get the inverse link transformation, use `inverse(b)`.

# Supported Link Functions
- `LogLink()`: Returns `elementwise(log)` for log transformation
- `LogitLink()`: Returns `Bijectors.Logit(0.0, 1.0)` for logit transformation
- `IdentityLink()`: Returns `identity` for identity transformation

# Arguments
- `link::LinkFunction`: The link function from an observation model

# Returns
A bijector object that can be used with `Bijectors.jl` interface:
- `b(μ)`: Apply link to transform μ → η
- `inverse(b)(η)`: Apply inverse link to transform η → μ
- `logabsdetjac(b, μ)`: Log absolute determinant of Jacobian

# Examples
```julia
# Log link (for Poisson, etc.)
link = LogLink()
bij = get_bijector(link)
μ = 7.389
η = bij(μ)  # log(7.389) ≈ 2.0
μ_back = inverse(bij)(η)  # exp(2.0) ≈ 7.389

# Logit link (for Binomial, etc.)
link = LogitLink()
bij = get_bijector(link)
μ = 0.5
η = bij(μ)  # logit(0.5) = 0.0
μ_back = inverse(bij)(η)  # logistic(0.0) = 0.5

# Identity link (for Gaussian, etc.)
link = IdentityLink()
bij = get_bijector(link)
η = bij(2.0)  # 2.0
```
"""
function get_bijector(link::LogLink)
    # Link is log: η = log(μ)
    return elementwise(log)
end

function get_bijector(link::LogitLink)
    # Link is logit: η = logit(μ)
    return Bijectors.Logit(0.0, 1.0)
end

function get_bijector(link::IdentityLink)
    # Identity transformation
    return identity
end

# Fallback for unsupported link functions
function get_bijector(link::LinkFunction)
    error(
        "Bijector mapping not implemented for link function type $(typeof(link)). " *
            "Supported types: LogLink, LogitLink, IdentityLink."
    )
end
