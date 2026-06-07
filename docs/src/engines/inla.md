# [INLA](@id engine-inla)

Latte's flagship engine — and the method the package is named for. Given a
[latent Gaussian model](@ref main-interface), [`inla`](@ref) returns posterior
marginals for the latent field and the hyperparameters, without MCMC, by *nesting*
two approximations.

## How it works

INLA never forms the full joint posterior — it targets each marginal directly.

**Inner — the latent field given the hyperparameters.** For a fixed ``\theta``,
the conditional ``p(x \mid y, \theta)`` is approximated by a Gaussian at its mode.
For Gaussian observations this is *exact*; otherwise it is a Laplace
approximation, and how faithfully you take it is the main accuracy knob.

**Outer — the hyperparameters.** The marginal posterior
``p(\theta \mid y) \propto p(\theta)\,p(y \mid \theta)`` — with the marginal
likelihood ``p(y \mid \theta)`` coming from that inner step — is reconstructed by
evaluating it at a set of ``\theta`` points and fitting a smooth surface through
them. The latent marginals are then a weighted mixture of the inner
approximations over those points.

The craft is in two places — *how accurately the inner marginals are taken* and
*where the outer ``\theta`` points are placed* — and both are exposed.

```@raw html
<InlaNested />
```

## Tuning

The defaults are accurate-and-fast for typical models; reach for these when a
model is unusual.

### Latent marginal accuracy — `latent_marginalization_method`

The inner Gaussian is symmetric, but real latent marginals are often skewed. This
controls how much of that skew you recover:

- `GaussianMarginal()` — the inner Gaussian itself. Fastest, symmetric.
- `SimplifiedLaplace()` — **default**: a cheap skewness correction to the Gaussian.
  The INLA sweet spot for most models.
- `LaplaceMarginal()` — a full Laplace approximation per component. Most accurate,
  most expensive; use it when the simplified correction isn't enough (strongly
  non-Gaussian likelihoods, few observations informing each latent).

```julia
inla(model, y; latent_marginalization_method = LaplaceMarginal())
```

### Hyperparameter exploration — `exploration_strategy`

Where the outer ``\theta`` points go:

- `AutoExplorationStrategy()` — **default**: a dense grid in low dimension,
  switching to a central-composite design (CCD) as the number of hyperparameters
  grows.
- `GridExplorationStrategy(integration_step_z = …, max_log_drop = …)` — an explicit
  grid. **Decrease `integration_step_z` to densify it** — the default is
  deliberately coarse, and a finer grid visibly sharpens *skewed* hyperparameter
  tails (we measure this on the [Validation](../validation/index.md) page). The cost
  is roughly one extra inner solve per added point.
- `CCDExplorationStrategy()` — force the CCD design: the practical choice with
  several hyperparameters, where a full grid is too expensive.

```julia
inla(model, y; exploration_strategy = GridExplorationStrategy(integration_step_z = 0.5))
```

### Mode finding — `mode_init`, `mode_diagnostic`

The outer points are placed relative to the mode of ``p(\theta \mid y)``. On
awkward (multimodal, skewed, near-degenerate) hyperparameter posteriors the
optimiser can settle on a poor mode, so Latte runs a post-hoc check and by default
**warns** if it finds a better grid point (`mode_diagnostic = :warn`). When that
fires, seed the search with a better guess — or a multi-start — via `mode_init`.

### Other knobs

- `latent_indices` — compute marginals only for the components you care about; a
  large speed-up on big latent fields.
- Model-selection criteria (DIC, WAIC, CPO, the marginal likelihood) are computed
  alongside by default through `accumulators`.

## Reference

```@docs
inla
```

## Limits

Measured, not asserted — see the [Validation](../validation/index.md) page:

- **Skewed hyperparameter tails.** The coarse default grid under-resolves them;
  densify with `GridExplorationStrategy(integration_step_z = …)`. A small residual
  remains for non-Gaussian likelihoods — that part is the inner Laplace marginal
  likelihood, which a finer grid can't fix.
- **Non-identified / ridged hyperparameter posteriors.** When the data identify
  only a *combination* of hyperparameters (e.g. ``\sigma^2 + 1/\tau`` in a
  Gaussian–Gaussian IID model), the outer surface can't follow the degenerate ridge
  and those marginals drift — even though the inner Laplace and the *exact*
  posterior agree. A faithful sampler (HMC-Laplace) recovers it.
- **Many hyperparameters.** The full grid grows exponentially in the number of
  hyperparameters; the `Auto`/CCD path mitigates this, and HMC-Laplace scales
  better when ``\theta`` is large.

## References

```@raw html
<div class="ref-grid-2">
<PaperCite
  tag="INLA"
  title="Approximate Bayesian Inference for Latent Gaussian Models by Using Integrated Nested Laplace Approximations"
  authors="H. Rue, S. Martino & N. Chopin"
  venue="J. R. Statist. Soc. B" year="2009"
  doi="10.1111/j.1467-9868.2008.00700.x"
  url="https://doi.org/10.1111/j.1467-9868.2008.00700.x"
  abstract="The original INLA paper: deterministic approximate Bayesian inference for latent Gaussian models via nested Laplace approximations and numerical integration over the hyperparameters." />
<PaperCite
  tag="Review"
  title="Bayesian Computing with INLA: A Review"
  authors="H. Rue, A. Riebler, S. H. Sørbye, J. B. Illian, D. P. Simpson & F. K. Lindgren"
  venue="Annual Review of Statistics and Its Application" year="2017"
  doi="10.1146/annurev-statistics-060116-054045"
  url="https://doi.org/10.1146/annurev-statistics-060116-054045"
  abstract="A modern review of the INLA methodology, the SPDE approach, and the R-INLA ecosystem." />
</div>
```

## See also

- Tutorials: [Getting started](../tutorials/getting_started.md),
  [Spatial disease mapping](../tutorials/disease_mapping_spatial.md).
- [Benchmarks](../benchmarks/index.md) — speed against the other engines and R-INLA.
- [Validation](../validation/index.md) — calibration of every engine.
- [Main Interface](@ref main-interface) — defining models and working with results.
