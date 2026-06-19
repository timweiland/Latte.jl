# Lower-level construction

The [`@latte` macro](latte.md) is the primary way to define a model: you write
the latent field and likelihood once, and Latte assembles the
[`LatentGaussianModel`](@ref) and its hyperparameter prior for you. This page
documents the secondary path — building an `LatentGaussianModel` by hand from
its three components. Reach for it when `@latte` cannot express your model, for
example when you want full control over the latent precision matrix and assemble
it yourself.

!!! tip "Prefer `@latte` for most models"
    Most models are written with the `@latte` macro, which builds a
    `LatentGaussianModel` for you (see [Defining models: the `@latte` macro](latte.md)
    and the tutorials). Construct one directly, as shown below, when you need
    full control over the latent prior.

A hand-built model has three parts:

1. a hyperparameter prior, declared with [`@hyperparams`](@ref);
2. a latent field prior — a function of the hyperparameters returning a mean and
   a precision matrix, wrapped in `FunctionLatentModel`;
3. an observation model linking observations to the latent field (see
   [Observation Models](@ref observation-models)).

## Hyperparameters: `@hyperparams`

Hyperparameters control the latent prior (e.g. marginal variance, correlation)
and the observation model (e.g. noise level). The [`@hyperparams`](@ref) macro
declares them, each with a prior distribution, a transformation, and the space
in which the prior is written.

```julia
using Latte, Distributions

spec = @hyperparams begin
    (σ ~ Exponential(1.0), transform = log, space = natural)
    (ρ ~ Beta(2, 2),       transform = logit, space = natural)
end
```

Each `~` line introduces a free parameter; a plain `name = value` line fixes a
parameter to a constant. Only free parameters appear in the parameter vector.

### Natural and working space

Latte distinguishes two spaces:

- **Natural space** is where parameters live for the user and for the model
  functions: a standard deviation `σ > 0`, a correlation `ρ ∈ (0, 1)`.
- **Working space** is the unconstrained space the inference engines optimise
  and explore in. The `transform` field is the bijector mapping natural →
  working (`log` for positive parameters, `logit` for the unit interval,
  `identity` for unbounded ones).

The two parameter wrappers, [`WorkingHyperparameters`](@ref) and
[`NaturalHyperparameters`](@ref), carry a vector together with the spec, and
`convert` moves between them and to a `NamedTuple`:

```julia
θ_w  = WorkingHyperparameters([0.5, -1.2], spec)  # working space (log σ, logit ρ)
θ_n  = convert(NaturalHyperparameters, θ_w)        # natural space (σ, ρ)
θ_nt = convert(NamedTuple, θ_n)                    # (ρ = logistic(-1.2), σ = exp(0.5))
```

Model functions receive parameters in natural space as keyword arguments, even
though optimisation happens in working space.

### Prior space and the Jacobian correction

The `space` field records where the prior is written. With `space = natural`,
the prior is a density over the natural parameter, and Latte applies the change
of variables to evaluate it in working space:

```math
\log p_\text{working}(\eta) = \log p_\text{natural}(\theta) + \log\left| \frac{d\theta}{d\eta} \right|,
```

where ``\eta`` is the working value and ``\theta = g(\eta)`` the corresponding
natural value. The log-Jacobian term is supplied by [`logdetjac`](@ref). With
`space = working`, the prior is already a density over the working parameter and
no correction is added. [`logpdf_prior`](@ref) evaluates the prior in working
space — including this correction when the prior was specified in natural space.

```julia
spec = @hyperparams begin
    (σ ~ Exponential(1.0), transform = log, space = natural)
end
# Equivalent to a prior on log(σ) in working space:
# p(log σ) = Exponential(1.0)(σ) · |dσ/d(log σ)| = Exponential(1.0)(σ) · σ
```

```@docs
@hyperparams
HyperparameterSpec
Hyperparameter
WorkingHyperparameters
NaturalHyperparameters
logpdf_prior
logdetjac
```

## Building a model directly: `LatentGaussianModel`

[`LatentGaussianModel`](@ref) combines the three components. It factors the
joint as ``p(\theta) \, p(x \mid \theta) \, p(y \mid x, \theta)`` with
``p(x \mid \theta)`` Gaussian.

The latent field prior is a function of the (keyword) hyperparameters that
returns a `(mean, precision)` tuple. Wrap it with `FunctionLatentModel(f, n)`,
passing the latent dimension `n`; a raw function is rejected with an error
asking you to wrap it. The function receives every hyperparameter — free and
fixed — by name in natural space, so accept `kwargs...` to ignore the ones it
does not use.

```julia
using Latte, GaussianMarkovRandomFields, Distributions, SparseArrays

spec = @hyperparams begin
    (σ ~ Exponential(1.0), transform = log, space = natural)
    (ρ ~ Beta(2, 2),       transform = logit, space = natural)
end

# Latent GMRF: an AR(1) precision built by hand
function ar1_latent(; σ, ρ, kwargs...)
    n = 100
    Q = spdiagm(
        -1 => -ρ * ones(n - 1),
         0 => (1 + ρ^2) * ones(n),
         1 => -ρ * ones(n - 1),
    )
    return (zeros(n), Q / σ^2)  # FunctionLatentModel expects (mean, precision)
end

obs_model = ExponentialFamily(Normal)

model = LatentGaussianModel(spec, FunctionLatentModel(ar1_latent, 100), obs_model)
```

### Validation

The constructor checks that every hyperparameter required by the observation
model is present in the spec, as a free or fixed parameter. A missing one is an
error:

```julia
# Errors: Normal requires a σ hyperparameter, but the spec only declares τ
spec = @hyperparams begin
    (τ ~ Gamma(2, 1), transform = log, space = natural)
end
obs_model = ExponentialFamily(Normal)
LatentGaussianModel(spec, FunctionLatentModel(latent_fn, 50), obs_model)  # ERROR: missing σ
```

Extra hyperparameters beyond those the observation model needs are allowed —
they are typically consumed by the latent field prior function.

Once built, the model feeds the inference engines directly; pass it to `inla`
(see the [INLA engine](../engines/inla.md)) and work with the result using the
[result accessors](results.md).

```@docs
LatentGaussianModel
latent_gmrf
log_joint_density
```
