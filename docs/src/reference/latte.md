# [Defining models: the `@latte` macro](@id latte-macro)

## Overview

`@latte` is the primary way to define a model in Latte. You write a function whose
body is a sequence of `~` statements; the macro walks that body at macro-expansion
time, classifies each site (observation, latent random effect, or hyperparameter),
and rewrites the function so that *calling* it returns a [`LatentGaussianModel`](lower_level.md).
That model object is what every engine consumes — pass it to [`inla`](../engines/inla.md),
`tmb`, or `hmc_laplace`.

A Besag disease-mapping model reads as a direct transcription of the generative story:

```julia
using Latte, GaussianMarkovRandomFields, Distributions, LinearAlgebra

@latte function besag_model(cases, expected, n, W)
    τ ~ Gamma(1.0, 1.0)                       # hyperparameter (scalar prior)
    β ~ MvNormal(zeros(1), 100.0 * I(1))      # latent (MvNormal ⇒ random effect)
    spatial ~ BesagModel(W)(τ = τ)            # latent (recognized constructor)
    for i in eachindex(cases)
        cases[i] ~ Poisson(expected[i] * exp(β[1] + spatial[i]))   # observation
    end
end

lgm = besag_model(cases, expected, n, W)      # returns a LatentGaussianModel
result = inla(lgm, cases)                      # run the engine
```

`lgm` carries the three pieces an engine needs — the hyperparameter priors, the
latent GMRF, and the observation model — assembled from the `~` statements. See
[Working with results](results.md) for the `result` object.

## What the macro classifies

Each `~` block is classified by static analysis of the function body. The rules:

- **observation** — the left-hand side is a positional argument of the function
  (`cases[i] ~ Poisson(...)`, where `cases` is in the signature);
- **random / latent** — the right-hand side is a recognized random-effect-shaped
  constructor: `MvNormal`, `IIDModel`, `RWModel`, `BesagModel`, `MaternModel`,
  `BYM2Model`, `SeparableModel`, `GMRF`, `ConstrainedGMRF`, and similar latent-model
  constructors;
- **hyperparameter / fixed effect** — anything else (a scalar univariate prior such
  as `τ ~ Gamma(1.0, 1.0)` falls here).

Composite observation grouping is detected automatically from each observation
block's hyperparameter dependencies, so two Gaussian channels with distinct noise
parameters become two observation groups without any manual argument.

### Overriding the default with `@random` / `@fixed`

Two markers placed in front of a `~` block override the per-block default:

```julia
@latte function marked(y, X)
    @random α ~ Normal(0, 1)          # scalar, but marginalised as a latent (TMB-style)
    @fixed   Σ ~ InverseWishart(...)  # multivariate, but treated as a hyperparameter
    σ ~ Gamma(2, 1)                   # default: scalar prior → fixed/hyperparameter
    β ~ MvNormal(...)                 # default: multivariate Gaussian → random
    # ...
end
```

`@random` promotes a site into the latent Gaussian field; `@fixed` keeps a site in
the hyperparameter block regardless of its shape.

## How `@latte` compiles your model

When you fit a non-Gaussian model with `@latte`, Latte reads your `~` statements as a
*factor graph* and builds the sparse gradient and Hessian the Laplace approximation
needs by differentiating each small factor on its own, instead of differentiating the
whole-model log-density at once. That keeps the work per Newton step proportional to
the model's sparsity, and it keeps the per-model compilation small.

A hierarchical model's log-density is a sum of *local* terms. Each `~` statement
contributes one factor that depends on only a handful of latent variables:

```math
\log p(x, y \mid \theta) \;=\;
\underbrace{\textstyle\sum_i \log p\!\left(x_i \mid x_{\mathrm{pa}(i)}, \theta\right)}_{\text{prior factors}}
\;+\;
\underbrace{\textstyle\sum_k \log p\!\left(y_k \mid x, \theta\right)}_{\text{observation factors}} .
```

Take a small nonlinear state-space model — a latent series with a nonlinear drift,
observed with Gaussian noise:

```julia
@latte function ssm(y, n)
    log_σ ~ Normal(0.0, 1.0)
    σ = exp(log_σ)
    x = Vector{Real}(undef, n)
    x[1] ~ Normal(0.0, 1.0)                   # initial-state factor
    for t in 2:n
        d = x[t - 1] - 0.5 * x[t - 1]^2       # nonlinear drift ⇒ non-Gaussian prior
        x[t] ~ Normal(d, σ)                   # transition factor: couples x[t], x[t-1]
    end
    for t in 1:n
        y[t] ~ Normal(x[t], 0.1)              # observation factor: ties x[t] to y[t]
    end
end
```

Every `~` line is a factor. The transitions all share one form,
`Normal(x[t-1] - 0.5 x[t-1]², σ)`, so they form a *group*: one small two-input
log-density applied at each `t`. The observations form another group. Drawing the
latent variables as circles and the factors as squares, for `n = 4`:

```@raw html
<div style="margin: 1.5rem 0; text-align: center;">
<svg viewBox="0 0 680 300" style="max-width: 640px; width: 100%; height: auto; font-family: system-ui, -apple-system, sans-serif;" role="img" aria-label="Factor graph of the state-space model">
  <rect x="14" y="20" width="652" height="262" rx="10" fill="#f7f8fa" stroke="#e1e4e8" stroke-width="1"/>
  <!-- edges -->
  <g stroke="#8a93a0" stroke-width="1.6">
    <line x1="69" y1="95" x2="120" y2="95"/>
    <line x1="160" y1="95" x2="206" y2="95"/><line x1="224" y1="95" x2="270" y2="95"/>
    <line x1="310" y1="95" x2="356" y2="95"/><line x1="374" y1="95" x2="420" y2="95"/>
    <line x1="460" y1="95" x2="506" y2="95"/><line x1="524" y1="95" x2="570" y2="95"/>
    <line x1="140" y1="115" x2="140" y2="172"/><line x1="290" y1="115" x2="290" y2="172"/>
    <line x1="440" y1="115" x2="440" y2="172"/><line x1="590" y1="115" x2="590" y2="172"/>
    <line x1="140" y1="188" x2="140" y2="235"/><line x1="290" y1="188" x2="290" y2="235"/>
    <line x1="440" y1="188" x2="440" y2="235"/><line x1="590" y1="188" x2="590" y2="235"/>
  </g>
  <!-- prior factors (blue squares): initial + transitions -->
  <g fill="#5b8ff9">
    <rect x="51" y="86" width="18" height="18" rx="2"/>
    <rect x="206" y="86" width="18" height="18" rx="2"/>
    <rect x="356" y="86" width="18" height="18" rx="2"/>
    <rect x="506" y="86" width="18" height="18" rx="2"/>
  </g>
  <!-- observation factors (amber squares) -->
  <g fill="#f2a93b">
    <rect x="132" y="172" width="16" height="16" rx="2"/>
    <rect x="282" y="172" width="16" height="16" rx="2"/>
    <rect x="432" y="172" width="16" height="16" rx="2"/>
    <rect x="582" y="172" width="16" height="16" rx="2"/>
  </g>
  <!-- latent variable nodes (circles) -->
  <g fill="#ffffff" stroke="#3b4252" stroke-width="1.8">
    <circle cx="140" cy="95" r="20"/><circle cx="290" cy="95" r="20"/>
    <circle cx="440" cy="95" r="20"/><circle cx="590" cy="95" r="20"/>
  </g>
  <!-- data nodes (gray squares) -->
  <g fill="#b0b7c0">
    <rect x="130" y="235" width="20" height="20" rx="2"/><rect x="280" y="235" width="20" height="20" rx="2"/>
    <rect x="430" y="235" width="20" height="20" rx="2"/><rect x="580" y="235" width="20" height="20" rx="2"/>
  </g>
  <!-- labels -->
  <g fill="#2e3440" font-size="15" text-anchor="middle">
    <text x="140" y="100">x₁</text><text x="290" y="100">x₂</text>
    <text x="440" y="100">x₃</text><text x="590" y="100">x₄</text>
  </g>
  <g fill="#ffffff" font-size="12" text-anchor="middle">
    <text x="140" y="250">y₁</text><text x="290" y="250">y₂</text>
    <text x="440" y="250">y₃</text><text x="590" y="250">y₄</text>
  </g>
</svg>
</div>
```

The blue squares are the prior factors (the initial-state factor and the transitions,
which share the hyperparameter `σ`); the amber squares are the observation factors.
Each colour is one *group* — a single reusable per-factor log-density replicated over
its index set. Latte differentiates those tiny functions and scatters the results into
the global sparse Hessian, so the work per Newton step is proportional to the number of
nonzeros, and automatic differentiation only ever specialises on the small factor
functions, not on the whole `n`-dimensional density. The generic assembly, Newton loop,
and hyperparameter-gradient machinery are compiled once and reused across models, which
is what keeps the first `inla` call fast.

Latte builds your model along one of two tiers:

- the **monolithic** tier differentiates an automatic-differentiation version of the
  whole density — always correct, but slower to compile the first time;
- the **structured** tier uses the factor-graph decomposition above, compiling only the
  small per-factor functions.

!!! note "Correctness never depends on it"
    The structured factor-graph prior is a performance refinement, not a separate model.
    On the first `inla` call Latte also builds a monolithic, automatic-differentiation
    version of your density, and accepts the factor-graph version only after checking it
    reproduces that version's local quadratic at independent probe points. Any pattern
    Latte can't structure silently falls back to the monolithic path — same answer,
    slower first compile.

## Writing models that take the fast path

A model takes the structured (fast-compile) path when each latent and observation site
is an *indexed scalar* `~` statement.

Allocate each latent field as an array whose shape Latte can read, then assign
element-wise:

```julia
x = Vector{Real}(undef, n)
x[1] ~ Normal(0.0, 1.0)
for t in 2:n
    x[t] ~ Normal(f(x[t - 1]), σ)
end
```

Write observations indexed and as a function of the latents they touch,
`y[t] ~ Dist(g(x[t], θ))`. Loops and nested loops are fine; the loop structure is
preserved. Hyperparameters and derived constants can be used inside a factor
(`σ = exp(log_σ)`, then `Normal(·, σ)`) — Latte carries the minimal set each factor
needs into its closure.

Beyond scalar sites, two further shapes take the structured path: a
multivariate-block site `x[:, t] ~ MvNormal(...)` (each column becomes one block
factor), and an element-wise broadcast prior `u .~ Dist.(...)` (lowered to the
equivalent scalar loop).

You can check which path a built model took:

```julia
lgm = ssm(y, n)
lgm.latent_prior isa StructuredLatentPrior   # true ⇒ structured path
```

If a model takes the monolithic path, Latte warns at build time (*"… uses the general
non-Gaussian path, whose first `inla` call is slow to compile …"*) and points at the
indexed-factor form to aim for. Rewriting the offending site to a scalar indexed `~`
usually moves it onto the structured path.

## Nonlinear Gaussian observations

A Gaussian observation whose mean is nonlinear in the latent field,
`y[i] ~ Normal(f(x), σ)` with `f` nonlinear in `x`, is recognized automatically and
dispatched to a Gauss–Newton nonlinear-least-squares observation model. This is the
default for such models; it replaces the full-Hessian automatic-differentiation path.

The forward map `f` may depend on hyperparameters, the noise scale `σ` may be a constant
(shared or per-observation) or an inferred hyperparameter, and the observation may be one
component of a composite model alongside other blocks. Each is carried through without
further annotation.

```julia
@latte function m(y, n)
    α ~ truncated(Normal(1, 0.5); lower = 0.1)   # enters the forward map
    τ ~ Gamma(2, 1)
    x ~ IIDModel(n)(τ = τ)
    for i in eachindex(y)
        y[i] ~ Normal(exp(α * x[i]), 0.1)         # nonlinear in x
    end
end
```

### The Gauss–Newton approximation

The Gauss–Newton model uses `JᵀWJ` for the observation Hessian, dropping the
residual-curvature term. The consequences, stated plainly:

- The mode and the hyperparameter-marginal gradients are exact (the score is unchanged),
  so posterior means and the hyperparameter posterior match the full-Hessian path.
- The latent marginal variances are approximate, and differ most where the forward map is
  strongly curved or the fit is poor.

`diagnose(result).obs_hessian` reports `:gauss_newton` for these models and `:exact`
otherwise, so the approximation is never silent. To take the exact full-Hessian path
instead, pass `nls = false` to the model constructor:

```julia
lgm = m(y, n; nls = false)   # exact observation Hessian
```

### When it falls back to automatic differentiation

The Gauss–Newton path requires Gaussian noise and a noise scale that does not depend on
the latent field. It returns to the exact AD path when the observation is non-Gaussian,
when `σ` depends on the latent vector, or when `σ` is a nonlinear transform of a
hyperparameter rather than a hyperparameter used directly. One naming constraint: in a
single-observation model an inferred `σ` must be named `σ`; composite-observation models
can route a differently-named noise hyperparameter. Every fallback is correct; the only
cost is a slower first compile.

## Limitations

Foregrounding where `@latte` falls back or restricts you. In every fallback case the
result is the always-correct monolithic path; the only cost is a slower first compile.

### Factor-graph fallbacks

The structured path covers the common case — scalar latent sites in (nested) loops with
a single observed array. It does not currently cover the following, falling back to the
monolithic path for each:

- Gaussian / linear-Gaussian models. These already assemble quickly through the GMRF
  path, so structuring isn't attempted — a different, and already fast, route rather
  than a limitation.
- Self-referential broadcast priors, such as `x[2:n] .~ Normal.(x[1:n-1], σ)` — ill-posed
  as a broadcast (it reads not-yet-sampled latents). Latte rejects these at definition
  time with an error pointing at the sequential loop form; write
  `for t in 2:n; x[t] ~ Normal(x[t-1], σ); end` instead. Element-wise broadcast priors
  (`u .~ Normal.(0.0, τ)`) are supported and structure like the loop form.
- Multiple observed symbols. The structured observation path handles a single observed
  array; models with several observed symbols keep the monolithic observation.
- Latents without a static array allocation. Each latent symbol needs a top-level
  allocation (e.g. `Vector{Real}(undef, n)`) whose shape Latte can infer.

### Recognition constraints

- `@latte` requires a plain function definition. It errors on anything that is not
  `function name(args...) ... end`.
- Keyword arguments in the signature disable the prelude-lift optimisation. The macro
  routes only positional arguments into the lifted prelude, so a signature with kwargs
  is built without that fast path (to avoid silently dropping the kwargs).
- Concrete-`LatentModel` recognition (a curried prior such as `BesagModel(W)(τ = τ)`) is
  shape-only. The macro recognizes the *form* at macro time; if the right-hand side does
  not actually instantiate to a `LatentModel` at runtime, Latte silently falls back to
  the DAG / sparse-AD extraction path.

### Composite-observation v1 limits

When a model has more than one observation group, the first cut focuses on
hyperparameter routing (distinct noise parameters per channel). A few things are not yet
supported for composite-observation models:

- WAIC / CPO accumulators are not exposed on composite-obs adapters.
- Outer hyperparameter-gradient INLA strategies (mode finder + grid expansion) are not
  wired up; fixed-grid strategies and `log_joint_density` calls work.
- Per-group Hessian-pattern overrides — all components share the global pattern.
- Posterior-predictive utilities that depend on the observation model's
  `conditional_distribution` (posterior-predictive draws, missing-value prediction) are
  not wired up for composite-obs adapters; use the underlying DPPL model directly for
  generative workflows.

## Turing handoff

The same function body is also forwarded to `DynamicPPL.@model` and exposed via
`Latte.dppl_model(name)`, so the identical definition can be sampled with NUTS:

```julia
turing_model = Latte.dppl_model(besag_model)(cases, expected, n, W)
sample(turing_model, NUTS(), 1000)
```

This lets you cross-check the INLA result against MCMC without rewriting the model. See
the [Turing handoff tutorial](../tutorials/turing_handoff.md).

## Worked examples

- [Tweedie likelihood](../tutorials/tweedie_insurance.md) — a custom non-Gaussian
  observation model.
- [Age-structured state-space model](../tutorials/age_structured_sam.md) — a structured
  factor-graph latent prior.

## Reference

```@docs
@latte
latte_from_dppl
```

`latte_from_dppl` is the manual escape hatch: when a model is written as a full
DynamicPPL `@model` whose body the macro's static analysis can't read (dynamic control
flow, programmatic site construction), convert the instantiated DPPL model into a
`LatentGaussianModel` directly by listing its latent (`random`) symbols.
