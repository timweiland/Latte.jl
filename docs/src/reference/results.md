# Working with results

```@meta
CurrentModule = Latte
```

Every engine — [`inla`](../engines/inla.md), `tmb`, `hmc_laplace` — returns a result that
implements one accessor protocol. This page collects those accessors in one
place. The examples use INLA, but the named-block and marginal accessors below
work the same way on any engine's result.

## Use the accessor functions, not the fields

Reach for the accessor *functions* — `latent_marginals(result)`,
`linear_predictor_marginals(result)`, `base_latent_marginals(result)` — rather
than the matching `result.…` fields.

Under the default (compact) latent parameterization the model does not
materialize the augmented blocks, so `result.linear_predictor_marginals` and
`result.base_latent_marginals` are `nothing`. The functions handle this: they
return the stored block when it exists and derive it from the latent posterior
otherwise. A compact result derives the linear predictor `η = A·x` from the
latent posterior through the design map; an augmented result slices the stored
`η`-block. Either way you get the same vector of marginals back, so code written
against the functions keeps working if the parameterization changes.

```julia
result = inla(model, y)

ηs = linear_predictor_marginals(result)   # works in both compact and augmented mode
xs = base_latent_marginals(result)        # the original model's latent components
```

## Latent marginals by name

When the model is written with [`@latte`](latte.md) (or converted with
`latte_from_dppl`), each `~`-bound latent term gets a name. Rather than indexing
the flat marginal vector with hand-computed offsets, ask for a named block with
`latent_marginals(result, name::Symbol)`. It returns the slice of marginals for
that term — a `Vector` of `Distribution`s, length 1 for a scalar term:

```julia
# Model with `β ~ MvNormal(...)` (regression coefficients) and a
# `u ~ BesagModel(W)(τ = τ)` spatial effect.
β_marginals = latent_marginals(result, :β)   # one marginal per coefficient
u_marginals = latent_marginals(result, :u)   # one marginal per spatial unit

β_means = [mean(m) for m in β_marginals]
β_stds  = [std(m)  for m in β_marginals]
```

This is the recommended way to pull out a specific term: it stays correct when
the latent layout changes (extra covariates, a different number of spatial
units) and reads more clearly than positional indexing. The available names are
those of the `~`-bound latent terms in the model; `latent_groups(result)`
returns the full `name => index-range` map. The same name-keyed form works for
hyperparameters via `hyperparameter_marginals(result, name)`, with
`hyperparameter_groups(result)` exposing their layout.

The returned marginals are `Distributions.jl` distributions, so `mean`, `var`,
`std`, `quantile`, `pdf`, and `rand` work on them directly.

## Hyperparameter marginals are in natural space

`hyperparameter_marginals(result)` returns the hyperparameter posteriors in
*natural* space — the scale the prior was declared on — not the internal working
(transformed) space the optimizer uses. So no back-transformation is needed
before computing summaries:

```julia
τ_marginal = hyperparameter_marginals(result, :τ)[1]   # precision, natural space

precision_mean = mean(τ_marginal)
precision_ci   = quantile(τ_marginal, [0.025, 0.975])
```

`hyperparameter_mode(result)` likewise reports the posterior mode in natural
space.

## Derived quantities

A declared hyperparameter's transform is inferred from its prior's support, so
**declaring the prior on the parameter you care about** is usually all you need.
A positive parameter is cleanest as a positive prior — write `α ~ LogNormal(0, 1)`
rather than `log_α ~ Normal(0, 1)` with `α = exp(log_α)` (an *identical* prior):
then `α` is a declared hyperparameter and `mean(hyperparameter_marginals(result, :α)[1])`
is the natural-space `E[α]`, directly.

For a quantity that genuinely can't be written as a single declared prior — a
function of two or more hyperparameters, say — use [`pushforward`](#pushforward)
to map a marginal through a transform. Its `mean`, `quantile`, etc. are computed
by integration, so `mean(pushforward(m, exp))` is the true `E[exp X]`, not the
Jensen-biased `exp(E[X])`:

```julia
m = hyperparameter_marginals(result, :log_β)[1]
derived = pushforward(m, exp)
mean(derived)                          # true E[exp(log_β)], integrated
quantile(derived, [0.025, 0.975])
```

## Fitted values on the response scale

`observation_marginals(result)` transforms the linear-predictor marginals
through the inverse link to give marginals for the expected observation
`μ = g⁻¹(η)`, one per observation. For a Poisson log-link these are the rates
`λ`; for a logit link they are the success probabilities `p`. The returned
distributions support the full `Distributions.jl` interface:

```julia
fitted = observation_marginals(result)
μ_mean = mean(fitted[1])
μ_ci   = (quantile(fitted[1], 0.025), quantile(fitted[1], 0.975))
```

This requires an `ExponentialFamily` observation model (possibly wrapped) whose
link function can be extracted.

## Convergence and timing

```julia
converged(result)       # did the optimization / exploration converge?
time_elapsed(result)    # total wall-clock time, in seconds
```

`log_marginal_likelihood(result)` returns the engine's approximation to
`log p(y)`, or `nothing` when the method has no natural estimate.

## Approximation quality

`diagnose(result)` runs a PSIS-k̂ check on the inner Laplace approximation
`q(x | θ) ≈ p(x | y, θ)` at the hyperparameter mode and returns a `NamedTuple`
with a relative effective sample size, the GPD shape `pareto_k`, and a
qualitative `interpretation` (`:excellent` / `:acceptable` / `:unreliable`):

```julia
d = diagnose(result)
d.rel_ess          # relative effective sample size, in (0, 1]
d.interpretation   # :excellent / :acceptable / :unreliable
```

An `:unreliable` verdict means the Gaussian inner approximation is a poor fit at
the mode, so the marginals downstream of it should be treated with caution.

## Reference

```@docs
INLAResult
latent_marginals
base_latent_marginals
linear_predictor_marginals
hyperparameter_marginals
observation_marginals
latent_groups
hyperparameter_groups
hyperparameter_mode
log_marginal_likelihood
converged
time_elapsed
pushforward
diagnose
```
