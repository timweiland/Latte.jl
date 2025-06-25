using Distributions
using Distributions: product_distribution
using StatsFuns
using LinearAlgebra

export ExponentialFamily, LinkFunction, IdentityLink, LogLink, LogitLink
export apply_link, apply_invlink, likelihood

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
    return p * (one(p) - p) * (one(p) - 2*p)
end

"""
    ExponentialFamily{F<:Distribution, L<:LinkFunction} <: ObservationModel

Observation model for exponential family distributions with link functions.

This struct represents observation models where the observations come from an exponential 
family distribution (Normal, Poisson, Bernoulli, Binomial) and the mean parameter is 
related to the latent field through a link function.

# Mathematical Model
For observations yᵢ and latent field values xᵢ:
- Linear predictor: ηᵢ = xᵢ
- Mean parameter: μᵢ = g⁻¹(ηᵢ) where g is the link function
- Observations: yᵢ ~ F(μᵢ, θ) where F is the distribution family

# Fields
- `family::Type{F}`: The distribution family (e.g., `Poisson`, `Bernoulli`)
- `link::L`: The link function connecting mean parameters to linear predictors

# Type Parameters
- `F`: A subtype of `Distribution` from Distributions.jl
- `L`: A subtype of `LinkFunction`

# Constructors
```julia
# Use canonical link (recommended)
ExponentialFamily(Poisson)        # Uses LogLink()
ExponentialFamily(Bernoulli)      # Uses LogitLink()
ExponentialFamily(Normal)         # Uses IdentityLink()

# Specify custom link function
ExponentialFamily(Poisson, IdentityLink())  # Non-canonical
```

# Supported Combinations
- `Normal` with `IdentityLink` (canonical) or `LogLink`
- `Poisson` with `LogLink` (canonical) or `IdentityLink`  
- `Bernoulli` with `LogitLink` (canonical) or `LogLink`
- `Binomial` with `LogitLink` (canonical) or `IdentityLink`

# Hyperparameters (θ)
Different families require different hyperparameters:
- `Normal`: `θ = [σ]` (standard deviation)
- `Poisson`: `θ = []` (no hyperparameters)
- `Bernoulli`: `θ = []` (no hyperparameters)
- `Binomial`: `θ = [n]` (number of trials)

# Examples
```julia
# Poisson model for count data
model = ExponentialFamily(Poisson)
x = [1.0, 2.0]        # Latent field (log scale due to LogLink)
θ = Float64[]         # No hyperparameters  
y = [2, 7]           # Count observations

ll = loglik(model, x, θ, y)
dist = likelihood(model, x, θ)  # Returns Product distribution

# Bernoulli model for binary data
model = ExponentialFamily(Bernoulli)
x = [0.0, 1.0]       # Latent field (logit scale due to LogitLink)
y = [0, 1]           # Binary observations
```

# Performance Notes
Canonical link functions have optimized implementations that avoid redundant computations.
Non-canonical links use general chain rule formulations which may be slower.

See also: [`LinkFunction`](@ref), [`loglik`](@ref), [`likelihood`](@ref)
"""
struct ExponentialFamily{F <: Distribution, L <: LinkFunction} <: ObservationModel
    family::Type{F}
    link::L
end

"""
    ExponentialFamily(family::Type{<:Distribution}) -> ExponentialFamily

Create an exponential family observation model with the canonical link function.

This constructor automatically selects the appropriate canonical link function for 
the given distribution family:
- `Normal` → `IdentityLink()`
- `Poisson` → `LogLink()`  
- `Bernoulli` → `LogitLink()`
- `Binomial` → `LogitLink()`

# Arguments
- `family`: A distribution type from Distributions.jl

# Returns
An `ExponentialFamily` instance with the canonical link function

# Examples
```julia
poisson_model = ExponentialFamily(Poisson)    # Uses LogLink
normal_model = ExponentialFamily(Normal)      # Uses IdentityLink
bernoulli_model = ExponentialFamily(Bernoulli) # Uses LogitLink
```

See also: [`ExponentialFamily`](@ref)
"""
ExponentialFamily(family::Type{<:Distribution}) = ExponentialFamily(family, _default_link(family))

_default_link(::Type{<:Normal}) = IdentityLink()
_default_link(::Type{<:Poisson}) = LogLink()
_default_link(::Type{<:Bernoulli}) = LogitLink()
_default_link(::Type{<:Binomial}) = LogitLink()

"""
    likelihood(obs_model::ExponentialFamily, x, θ_named) -> Distribution

Construct the likelihood distribution for given latent field values and hyperparameters.

This function returns a `Distribution` object (typically a product distribution) that 
represents the likelihood p(y | x, θ) for the exponential family observation model. 
The returned distribution can be used with all standard Distributions.jl functions
such as `logpdf`, `rand`, `mean`, `var`, etc.

# Arguments
- `obs_model`: An `ExponentialFamily` observation model
- `x`: Latent field values (vector of length n)
- `θ_named`: Hyperparameters as a NamedTuple (e.g., `(σ = 0.5,)`)

# Returns
A `Distribution` object representing the likelihood. For independent observations,
this is typically a product distribution from Distributions.jl.

# Examples
```julia
# Poisson model
model = ExponentialFamily(Poisson)
x = [1.0, 2.0]              # Latent field (log scale)
θ_named = NamedTuple()      # No hyperparameters

dist = likelihood(model, x, θ_named)  # Product of Poisson distributions
y_sample = rand(dist)                 # Generate random observations
ll = logpdf(dist, [2, 7])            # Compute log-likelihood

# Normal model  
model = ExponentialFamily(Normal)
x = [0.0, 1.0]              # Latent field (identity scale)
θ_named = (σ = 0.5,)        # Standard deviation

dist = likelihood(model, x, θ_named)  # Product of Normal distributions
```

# Relationship to loglik
The `likelihood` function provides the underlying distribution, while `loglik` 
evaluates it:
```julia
dist = likelihood(model, x, θ_named)
ll1 = logpdf(dist, y)                  # Using likelihood
ll2 = loglik(model, x, θ_named, y)     # Direct evaluation
# ll1 ≈ ll2
```

See also: [`ExponentialFamily`](@ref), [`loglik`](@ref)
"""
function likelihood(obs_model::ExponentialFamily, x, θ_named)
    η = x  # Linear predictor
    μ = apply_invlink.(Ref(obs_model.link), η)
    
    return _likelihood_family(obs_model.family, μ, θ_named)
end

function _likelihood_family(::Type{<:Normal}, μ, θ_named)
    σ = θ_named.σ  # Extract standard deviation
    return product_distribution(Normal.(μ, σ))
end

function _likelihood_family(::Type{<:Poisson}, μ, θ_named)
    return product_distribution(Poisson.(μ))
end

function _likelihood_family(::Type{<:Bernoulli}, μ, θ_named)
    return product_distribution(Bernoulli.(μ))
end

function _likelihood_family(::Type{<:Binomial}, μ, θ_named)
    n = θ_named.n  # Extract number of trials
    return product_distribution(Binomial.(n, μ))
end

# Refactor loglik to use the likelihood function
function loglik(obs_model::ExponentialFamily, x, θ_named, y)
    dist = likelihood(obs_model, x, θ_named)
    return logpdf(dist, y)
end


# Specialized fast paths for canonical links
function loggrad(::ExponentialFamily{<:Poisson, LogLink}, x, θ_named, y)
    η = x
    μ = exp.(η)
    return y .- μ
end

function loggrad(::ExponentialFamily{<:Bernoulli, LogitLink}, x, θ_named, y)
    η = x
    μ = logistic.(η)
    return y .- μ
end

function loggrad(::ExponentialFamily{<:Binomial, LogitLink}, x, θ_named, y)
    η = x
    μ = logistic.(η)
    n = θ_named.n
    return y .- n .* μ
end

function loggrad(::ExponentialFamily{<:Normal, IdentityLink}, x, θ_named, y)
    η = x
    μ = η  # Identity link
    σ = θ_named.σ
    return (y .- μ) ./ σ^2
end

# General fallback using chain rule
function loggrad(obs_model::ExponentialFamily, x, θ_named, y)
    η = x
    μ = apply_invlink.(Ref(obs_model.link), η)
    dμ_dη = derivative_invlink.(Ref(obs_model.link), η)
    
    return _loggrad_family(obs_model.family, μ, dμ_dη, θ_named, y)
end

function _loggrad_family(::Type{<:Normal}, μ, dμ_dη, θ_named, y)
    σ = θ_named.σ
    return ((y .- μ) ./ σ^2) .* dμ_dη
end

function _loggrad_family(::Type{<:Poisson}, μ, dμ_dη, θ_named, y)
    return ((y .- μ) ./ μ) .* dμ_dη
end

function _loggrad_family(::Type{<:Bernoulli}, μ, dμ_dη, θ_named, y)
    return ((y .- μ) ./ (μ .* (1 .- μ))) .* dμ_dη
end

function _loggrad_family(::Type{<:Binomial}, μ, dμ_dη, θ_named, y)
    n = θ_named.n
    return ((y .- n .* μ) ./ (μ .* (1 .- μ))) .* dμ_dη
end

# Specialized fast paths for canonical links
function loghessian(obs_model::ExponentialFamily{<:Poisson, LogLink}, x, θ_named, y)
    η = x
    μ = exp.(η)
    return Diagonal(-μ)
end

function loghessian(obs_model::ExponentialFamily{<:Bernoulli, LogitLink}, x, θ_named, y)
    η = x
    μ = logistic.(η)
    return Diagonal(-μ .* (1 .- μ))
end

function loghessian(obs_model::ExponentialFamily{<:Binomial, LogitLink}, x, θ_named, y)
    η = x
    μ = logistic.(η)
    n = θ_named.n
    return Diagonal(-n .* μ .* (1 .- μ))
end

function loghessian(obs_model::ExponentialFamily{<:Normal, IdentityLink}, x, θ_named, y)
    η = x
    σ = θ_named.σ
    return Diagonal(-ones(length(η)) ./ σ^2)
end

# General fallback using chain rule
function loghessian(obs_model::ExponentialFamily, x, θ_named, y)
    η = x
    μ = apply_invlink.(Ref(obs_model.link), η)
    dμ_dη = derivative_invlink.(Ref(obs_model.link), η)
    d2μ_dη2 = second_derivative_invlink.(Ref(obs_model.link), η)
    
    diagonal_terms = _loghessian_diagonal_family(obs_model.family, μ, dμ_dη, d2μ_dη2, θ_named, y)
    return Diagonal(diagonal_terms)
end

function _loghessian_diagonal_family(::Type{<:Normal}, μ, dμ_dη, d2μ_dη2, θ_named, y)
    σ = θ_named.σ
    return -(dμ_dη.^2) ./ σ^2 .+ (y .- μ) ./ σ^2 .* d2μ_dη2
end

function _loghessian_diagonal_family(::Type{<:Poisson}, μ, dμ_dη, d2μ_dη2, θ_named, y)
    # ∂²ℓ/∂η² = (∂²ℓ/∂μ²) × (∂μ/∂η)² + (∂ℓ/∂μ) × (∂²μ/∂η²)
    # For Poisson: ∂²ℓ/∂μ² = -y/μ², ∂ℓ/∂μ = y/μ - 1
    d2l_dmu2 = -y ./ (μ.^2)
    dl_dmu = (y ./ μ) .- 1
    return d2l_dmu2 .* (dμ_dη.^2) .+ dl_dmu .* d2μ_dη2
end

function _loghessian_diagonal_family(::Type{<:Bernoulli}, μ, dμ_dη, d2μ_dη2, θ_named, y)
    # ∂²ℓ/∂η² = (∂²ℓ/∂μ²) × (∂μ/∂η)² + (∂ℓ/∂μ) × (∂²μ/∂η²)
    # For Bernoulli: ∂²ℓ/∂μ² = -y/μ² - (1-y)/(1-μ)², ∂ℓ/∂μ = y/μ - (1-y)/(1-μ)
    d2l_dmu2 = -(y ./ (μ.^2)) .- ((1 .- y) ./ ((1 .- μ).^2))
    dl_dmu = (y ./ μ) .- ((1 .- y) ./ (1 .- μ))
    return d2l_dmu2 .* (dμ_dη.^2) .+ dl_dmu .* d2μ_dη2
end

function _loghessian_diagonal_family(::Type{<:Binomial}, μ, dμ_dη, d2μ_dη2, θ_named, y)
    n = θ_named.n
    # ∂²ℓ/∂η² = (∂²ℓ/∂μ²) × (∂μ/∂η)² + (∂ℓ/∂μ) × (∂²μ/∂η²)
    # For Binomial: ∂²ℓ/∂μ² = -y/μ² - (n-y)/(1-μ)², ∂ℓ/∂μ = y/μ - (n-y)/(1-μ)
    d2l_dmu2 = -(y ./ (μ.^2)) .- ((n .- y) ./ ((1 .- μ).^2))
    dl_dmu = (y ./ μ) .- ((n .- y) ./ (1 .- μ))
    return d2l_dmu2 .* (dμ_dη.^2) .+ dl_dmu .* d2μ_dη2
end

# Hyperparameter interface implementations
hyperparameters(::ExponentialFamily{<:Normal}) = (:σ,)
hyperparameters(::ExponentialFamily{<:Bernoulli}) = ()
hyperparameters(::ExponentialFamily{<:Binomial}) = (:n,)
hyperparameters(::ExponentialFamily{<:Poisson}) = ()
