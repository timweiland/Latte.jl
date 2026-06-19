# [Factor-graph models](@id factor-graph-models)

When you fit a non-Gaussian model with `@latte`, Latte reads your `~` statements as a *factor
graph* and builds the sparse gradient and Hessian the Laplace approximation needs by
differentiating each small factor on its own — the same sparse-Hessian-of-the-joint idea TMB
uses — instead of differentiating the whole-model log-density at once. That keeps the work per
Newton step proportional to the model's sparsity, and it keeps the per-model compilation small.

This page explains the idea, how to write a model so it takes that path, and where it currently
falls back.

## The idea

A hierarchical model's log-density is a sum of *local* terms. Each `~` statement contributes one
factor that depends on only a handful of latent variables:

```math
\log p(x, y \mid \theta) \;=\;
\underbrace{\textstyle\sum_i \log p\!\left(x_i \mid x_{\mathrm{pa}(i)}, \theta\right)}_{\text{prior factors}}
\;+\;
\underbrace{\textstyle\sum_k \log p\!\left(y_k \mid x, \theta\right)}_{\text{observation factors}} .
```

Take a small nonlinear state-space model — a latent series with a nonlinear drift, observed with
Gaussian noise:

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

Every `~` line is a factor. The transitions all share one form, `Normal(x[t-1] - 0.5 x[t-1]², σ)`,
so they form a *group*: one small two-input log-density applied at each `t`. The observations form
another group. Drawing the latent variables as circles and the factors as squares, for `n = 4`:

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

The blue squares are the prior factors (the initial-state factor and the transitions, which share
the hyperparameter `σ`); the amber squares are the observation factors. Each colour is one *group*
— a single reusable per-factor log-density replicated over its index set. Latte differentiates
those tiny functions and scatters the results into the global sparse Hessian, so the work per
Newton step is proportional to the number of nonzeros, and automatic differentiation only ever
specialises on the small factor functions, not on the whole `n`-dimensional density. The generic
assembly, Newton loop, and hyperparameter-gradient machinery are compiled once and reused across
models, which is what keeps the first `inla` call fast.

!!! note "Correctness never depends on it"
    The structured factor-graph prior is a performance refinement, not a separate model. On the
    first `inla` call Latte also builds a monolithic, automatic-differentiation version of your
    density, and accepts the factor-graph version only after checking it reproduces that version's
    local quadratic at independent probe points. Any pattern Latte can't structure silently falls
    back to the monolithic path — same answer, slower first compile.

## Writing models that take the structured path

A model takes the structured path when each latent and observation site is an *indexed scalar*
`~` statement.

Allocate each latent field as an array whose shape Latte can read, then assign element-wise:

```julia
x = Vector{Real}(undef, n)
x[1] ~ Normal(0.0, 1.0)
for t in 2:n
    x[t] ~ Normal(f(x[t - 1]), σ)
end
```

Write observations indexed and as a function of the latents they touch, `y[t] ~ Dist(g(x[t], θ))`.
Loops and nested loops are fine; the loop structure is preserved. Hyperparameters and derived
constants can be used inside a factor (`σ = exp(log_σ)`, then `Normal(·, σ)`) — Latte carries the
minimal set each factor needs into its closure.

You can check which path a built model took:

```julia
lgm = ssm(y, n)
lgm.latent_prior isa StructuredLatentPrior   # true ⇒ structured path
```

If a model takes the monolithic path, Latte warns at build time (*"… uses the general non-Gaussian
path, whose first `inla` call is slow to compile …"*) and points at the indexed-factor form to aim
for. Rewriting the offending site to a scalar indexed `~` usually moves it onto the structured
path.

## Limits and fallback

The structured path covers the common case — scalar latent sites in (nested) loops with a single
observed array. It does not currently cover the following, and falls back to the monolithic path
for each:

- Gaussian / linear-Gaussian models. These already assemble quickly through the GMRF path, so
  structuring isn't attempted — a different, and already fast, route rather than a limitation.
- Multivariate-block latents written with a slice, such as `x[:, t] ~ MvNormal(...)`. Write the
  block element-wise as scalar sites (`x[a, t] ~ …`); this is currently required, since the
  monolithic path does not handle this shape yet either (issue #22).
- Broadcast (dotted) priors, such as `u .~ Normal.(...)`. Write the explicit loop form
  (`for i; u[i] ~ Dist(...); end`) instead (issue #23).
- Multiple observed symbols. The structured observation path handles a single observed array;
  models with several observed symbols keep the monolithic observation.
- Latents without a static array allocation. Each latent symbol needs a top-level allocation
  (e.g. `Vector{Real}(undef, n)`) whose shape Latte can infer.

In every case the fallback is the always-correct monolithic path; the only cost is a slower first
compile. Clearer up-front errors for the unsupported shapes are tracked in issue #24.

## See also

- [INLA engine](engines/inla.md) — the inference the factor graph feeds.
- [Main interface](main_interface.md) — `inla` and the model objects.
- [Custom likelihoods: Tweedie](tutorials/tweedie_insurance.md) and
  [state-space assessment](tutorials/fisheries_state_space.md) — worked non-Gaussian models.
