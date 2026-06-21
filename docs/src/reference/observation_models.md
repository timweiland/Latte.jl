# [Observation Models](@id observation-models)

!!! note "Provided by GaussianMarkovRandomFields.jl"
    Observation models are implemented in [GaussianMarkovRandomFields.jl](https://github.com/timweiland/GaussianMarkovRandomFields.jl) v0.4+ and re-exported by Latte.jl for user convenience. **For detailed API documentation**, see the [GaussianMarkovRandomFields.jl documentation](https://timweiland.github.io/GaussianMarkovRandomFields.jl/).

This guide shows how to use observation models with INLA. Observation models define the relationship between observations `y` and the latent field `x` through probability distributions and link functions.

## Quick Start

```julia
using Latte
using Distributions

# 1. Create observation model (Poisson with canonical log link)
obs_model = ExponentialFamily(Poisson)

# 2. Use in INLA model
spec = @hyperparams begin
    (σ ~ Gamma(2, 1), transform = log, space = natural)
end

model = LatentGaussianModel(spec, FunctionLatentModel(my_latent_function, n), obs_model)

# 3. Run inference
result = inla(model, y_observed)
```

## Overview

Observation models use a **factory pattern** for efficiency:
1. **Create template**: `obs_model = ExponentialFamily(Distribution)`
2. **Use in INLA**: Pass to `LatentGaussianModel(spec, FunctionLatentModel(latent_fn, n), obs_model)`
3. **INLA handles the rest**: Automatic materialization with data and hyperparameters

The `ExponentialFamily` struct supports most common statistical distributions with their canonical link functions.

## Common Observation Models

| Distribution | Canonical Link | Hyperparameters | Use Case |
|:-------------|:---------------|:----------------|:---------|
| `Normal` | Identity | `σ` (std dev) | Continuous data |
| `Poisson` | Log | None | Count data |
| `Bernoulli` | Logit | None | Binary data |
| `Binomial` | Logit | `n` (trials) | Binomial trials |

### Usage Examples

#### Poisson Model (Count Data)
```julia
using Latte, GaussianMarkovRandomFields, Distributions

# Poisson uses log link: η = log(λ) where λ is the rate
obs_model = ExponentialFamily(Poisson)

spec = @hyperparams begin
    (ρ ~ Beta(2, 2), transform = logit, space = natural)
end

# Latent field defines log-rates — returns (mean, precision)
latent_fn(; ρ, kwargs...) = (zeros(100), ar1_precision(100, ρ))

model = LatentGaussianModel(spec, FunctionLatentModel(latent_fn, 100), obs_model)
```

#### Bernoulli Model (Binary Data)
```julia
# Bernoulli uses logit link: η = logit(p) where p is the probability
obs_model = ExponentialFamily(Bernoulli)

# Use in INLA model (no observation hyperparameters needed)
model = LatentGaussianModel(spec, FunctionLatentModel(latent_fn, 100), obs_model)
```

#### Normal Model (Continuous Data)
```julia
# Normal uses identity link: η = μ directly
obs_model = ExponentialFamily(Normal)

# Normal requires observation noise σ as hyperparameter
spec = @hyperparams begin
    (σ_latent ~ Gamma(2, 1), transform = log, space = natural)
    (σ_obs ~ Gamma(2, 1), transform = log, space = natural)
end

# The σ_obs will be passed to the observation model automatically
model = LatentGaussianModel(spec, FunctionLatentModel(latent_fn, 100), obs_model)
```

#### Binomial Model
```julia
# Binomial uses logit link for success probability
obs_model = ExponentialFamily(Binomial)

# Binomial requires n (number of trials) as hyperparameter
spec = @hyperparams begin
    (ρ ~ Beta(2, 2), transform = logit, space = natural)
    n = 10.0  # Fixed number of trials
end

model = LatentGaussianModel(spec, FunctionLatentModel(latent_fn, 100), obs_model)
```

### Non-Canonical Links

You can specify custom link functions when needed:

```julia
# Poisson with identity link (latent field = rate directly, must be positive)
obs_model = ExponentialFamily(Poisson, IdentityLink())

# Bernoulli with log link (latent field = log-probability)
obs_model = ExponentialFamily(Bernoulli, LogLink())
```

For most applications, the canonical links (default) are recommended.

## Composite Observation Models

Composite observation models combine multiple observation types (e.g., continuous and count data) in a single INLA model:

```julia
using Latte

# Create models with index ranges
normal_model = ExponentialFamily(Normal, indices = 1:3)    # First 3 latent values
poisson_model = ExponentialFamily(Poisson, indices = 4:6)  # Next 3 latent values

composite_model = CompositeObservationModel((normal_model, poisson_model))

# Prepare heterogeneous observations
y_normal = [1.0, 2.0, 1.5]
y_poisson = [2, 3, 1]
y_composite = CompositeObservations((y_normal, y_poisson))

# Use with INLA
spec = @hyperparams begin
    (ρ ~ Beta(2, 2), transform = logit, space = natural)
    (σ ~ Gamma(2, 1), transform = log, space = natural)  # For Normal component
end

model = LatentGaussianModel(spec, FunctionLatentModel(latent_fn, 6), composite_model)
result = inla(model, y_composite)
```

Each component uses its specified latent field indices and extracts its required hyperparameters automatically.

### Auto-detection via `@latte`

For new code, the easiest path is the `@latte` macro: it parses your
model body at macro time, classifies every `~` block, and produces a
`LatentGaussianModel` directly. Composite observation grouping is
detected automatically from each block's hyperparameter dependencies —
no manual `obs_groups` argument needed in the typical case.

```julia
using Latte, Distributions, LinearAlgebra

@latte function pde_inverse(y_phys, y_sensor, A_phys, A_sensor)
    σ_phys ~ Gamma(2, 1)
    σ_data ~ Gamma(2, 1)
    β ~ MvNormal(zeros(size(A_phys, 2)), 100.0 * I)
    for i in eachindex(y_phys)
        y_phys[i] ~ Normal(dot(A_phys[i, :], β), σ_phys)
    end
    for i in eachindex(y_sensor)
        y_sensor[i] ~ Normal(dot(A_sensor[i, :], β), σ_data)
    end
end

lgm = pde_inverse(y_phys, y_sensor, A_phys, A_sensor)   # auto-detected: two obs groups
```

The macro classifies each `~` block as one of:
- **observation** — LHS is a positional argument of the function;
- **random effect** — RHS is a known random-effect-shaped constructor
  (`MvNormal`, `IIDModel`, `RWModel`, `BesagModel`, `MaternModel`,
  `BYM2Model`, `SeparableModel`, `GMRF`, `ConstrainedGMRF`, …);
- **hyperparameter / fixed effect** — anything else.

`@random` and `@fixed` markers override the default per `~` block:

```julia
@latte function tmb_style(y, X)
    @random α ~ Normal(0, 1)         # scalar but marginalised (TMB-style)
    @fixed Σ ~ InverseWishart(...)    # multivariate but treated as hyperparameter
    σ ~ Gamma(2, 1)                   # default: scalar → fixed
    β ~ MvNormal(...)                 # default: multivariate Gaussian → random
    ...
end
```

The same body is also exposed as a Turing-compatible DPPL model:

```julia
turing_model = Latte.dppl_model(pde_inverse)(y_phys, y_sensor, A_phys, A_sensor)
sample(turing_model, NUTS(), 1000)    # Turing handoff with the same definition
```

### Manual control through `latte_from_dppl`

When the model is written as a DPPL `@model`, you can split observation `~`
blocks into named groups via the `obs_groups` keyword. The adapter builds
one component per group, each with its own kwargs routing — letting two
otherwise identical likelihoods (e.g. two `MvNormal` blocks) carry
distinct hyperparameter names.

The motivating case is a PDE-inverse problem with two Gaussian channels —
a physics residual with `σ_phys` and sensor observations with `σ_data`:

```julia
using Latte
using DynamicPPL: @model
using Distributions, LinearAlgebra

@model function pde_inverse(y_phys, y_sensor, A_phys, A_sensor)
    σ_phys ~ Gamma(2, 1)
    σ_data ~ Gamma(2, 1)
    β ~ MvNormal(zeros(size(A_phys, 2)), 100.0 * I)
    for i in eachindex(y_phys)
        y_phys[i] ~ Normal(dot(A_phys[i, :], β), σ_phys)
    end
    for i in eachindex(y_sensor)
        y_sensor[i] ~ Normal(dot(A_sensor[i, :], β), σ_data)
    end
end

lgm = latte_from_dppl(
    pde_inverse(y_phys, y_sensor, A_phys, A_sensor);
    random = (:β,),
    obs_groups = [
        :physics => (:y_phys,),
        :data    => (:y_sensor,),
    ],
)
```

Either form is accepted:

```julia
obs_groups = [:physics => (:y_phys,), :data => (:y_sensor,)]
obs_groups = (physics = (:y_phys,), data = (:y_sensor,))   # NamedTuple
```

Constraints (validated at adapter time):
- every observation `~` symbol must appear in exactly one group;
- each declared symbol must actually be observed by the model (not a
  hyperparameter or a latent random variable).

#### v1 limitations

The first cut focuses on hyperparameter routing — distinct `σ_phys` /
`σ_data` for two Gaussian channels. A few things are deliberately *not*
yet supported, and will surface as separate work:

- **WAIC / CPO** accumulators on composite-obs adapters. Component AD
  likelihoods don't expose `pointwise_loglik_func` because the upstream
  diagonal-Hessian shortcut would silently return a wrong (diagonal)
  Hessian for any block whose linear predictor mixes latent components.
- **Outer hp-gradient INLA strategies** (mode finder + grid expansion).
  The single-AD path leans on an IFT dispatch that isn't wired up for
  `CompositeLikelihood` upstream, so composite-obs LGMs hit nested-AD
  tag stacking. Fixed-grid strategies and `log_joint_density` calls work
  fine; full `inla()` will land once the upstream IFT path is extended.
- **Per-group Hessian-pattern overrides.** All components share the
  global pattern (`likelihood_hessian_pattern` kwarg). Splitting an
  opaque PDE solver from a tracer-friendly sensor block into separate
  patterns is a future feature.
- **Posterior-predictive utilities** that depend on the observation
  model's `conditional_distribution` (`rand(model)`, posterior-predictive
  draws, missing-value prediction) aren't wired up for composite-obs
  adapters. This matches the existing AD adapter — composite just
  inherits the same gap. Use the underlying DPPL model directly for
  prior / generative workflows.

## Custom Observation Models

For specialized applications beyond the built-in exponential family models, you can implement custom observation models. This requires:

1. Defining a struct that subtypes `ObservationModel`
2. Implementing the factory pattern with `(model::YourModel)(y; kwargs...)`
3. Implementing `loglik(obs_lik, x)` for the materialized likelihood

See the [GaussianMarkovRandomFields.jl documentation](https://timweiland.github.io/GaussianMarkovRandomFields.jl/) for detailed implementation guides and advanced features like:
- Analytical gradient and Hessian implementations for performance
- Automatic differentiation with sparsity detection
- Custom link function definitions

## Further Reading

For complete API documentation, advanced features, and implementation details, refer to:
- **[GaussianMarkovRandomFields.jl Documentation](https://timweiland.github.io/GaussianMarkovRandomFields.jl/)** - Complete observation model API reference
- The [tutorials](../tutorials/index.md) for practical INLA usage patterns
