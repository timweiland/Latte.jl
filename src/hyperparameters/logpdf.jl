using Distributions
using Bijectors

export logpdf_prior

"""
    logpdf_prior(θ::WorkingHyperparameters) -> Float64

Evaluate the log prior density in working space.

# Arguments
- `θ::WorkingHyperparameters`: Hyperparameters in working space

# Returns
- `Float64`: Log prior density in working space

# Details
Since priors are stored in working space internally, this directly evaluates the prior
at the working-space values.

# Example
```julia
spec = HyperparameterSpec(
    free = (σ = Hyperparameter(Exponential(1.0), transform=elementwise(log), prior_space=:natural),)
)
θ_w = WorkingHyperparameters([0.5], spec)
log_p = logpdf_prior(θ_w)  # Evaluates log p(η) in working space
```
"""
function logpdf_prior(θ::WorkingHyperparameters)
    return mapreduce(+, enumerate(keys(θ.spec.free)); init = 0.0) do (i, name)
        hp = θ.spec.free[name]
        working_value = θ.θ[i]
        # Prior is stored in working space, evaluate directly
        logpdf(hp.prior, working_value)::Float64
    end
end

"""
    logpdf_prior(θ::NaturalHyperparameters) -> Float64

Evaluate the log prior density in natural space.

# Arguments
- `θ::NaturalHyperparameters`: Hyperparameters in natural space

# Returns
- `Float64`: Log prior density in natural space

# Details
Converts to working space, evaluates the working-space prior, then adds the Jacobian
correction to obtain the natural-space density.

# Example
```julia
spec = HyperparameterSpec(
    free = (σ = Hyperparameter(Exponential(1.0), transform=elementwise(log), prior_space=:natural),)
)
θ_n = NaturalHyperparameters([2.0], spec)
log_p = logpdf_prior(θ_n)  # Evaluates log π(θ) in natural space
```
"""
function logpdf_prior(θ::NaturalHyperparameters)
    # Convert to working space
    θ_w = convert(WorkingHyperparameters, θ)
    # Evaluate working-space prior and add Jacobian correction
    return logpdf_prior(θ_w) + logdetjac(θ)
end

"""
    logpdf_prior(θ_natural::NamedTuple, spec::HyperparameterSpec) -> Float64

Evaluate the log prior density in natural space (legacy interface).

# Arguments
- `θ_natural::NamedTuple`: Free parameters in natural space
- `spec::HyperparameterSpec`: Hyperparameter specification

# Returns
- `Float64`: Log prior density in natural space

# Details
Legacy interface that constructs NaturalHyperparameters internally and dispatches
to the modern implementation.

# Example
```julia
spec = HyperparameterSpec(
    free = (σ = Hyperparameter(Exponential(1.0), transform=elementwise(log), prior_space=:natural),)
)

θ_natural = (σ = 2.0,)  # σ in natural space
log_p = logpdf_prior(θ_natural, spec)  # Evaluates log π(σ)
```
"""
function logpdf_prior(θ_natural::NamedTuple, spec::HyperparameterSpec)
    # Extract free parameters as vector
    θ_vec = [θ_natural[name] for name in keys(spec.free)]
    θ_n = NaturalHyperparameters(θ_vec, spec)
    return logpdf_prior(θ_n)
end
