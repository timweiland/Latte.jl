# [INLA](@id engine-inla)

INLA is Latte's default inference engine. Given a
[latent Gaussian model](@ref main-interface), [`inla`](@ref) computes posterior
marginals for the latent field and the hyperparameters without any MCMC. It does
this by nesting two approximations.

## How it works

INLA never builds the full joint posterior. It goes after each marginal directly,
in two steps.

The first step is the latent field at a fixed value of the hyperparameters
``\theta``. The conditional ``p(x \mid y, \theta)`` is approximated by a Gaussian
centred at its mode. When the observations are Gaussian this is exact; otherwise
it is a Laplace approximation, and how accurately you take it is the main accuracy
setting (see Tuning below).

The second step is the hyperparameters. The marginal posterior
``p(\theta \mid y) \propto p(\theta)\,p(y \mid \theta)`` is reconstructed by
evaluating it at a handful of ``\theta`` points, where the marginal likelihood
``p(y \mid \theta)`` comes from the first step, and fitting a smooth surface
through them. Each latent marginal then falls out as a weighted mixture of the
Gaussians from the first step, one per ``\theta`` point.

So there are two things you actually control: how accurately the inner marginals
are taken, and where the ``\theta`` points are placed. Drag the grid in the
picture below to see the second one at work.

```@raw html
<InlaNested />
```

## Tuning

The defaults work well for most models. Here is what to reach for when one doesn't.

### Latent marginal accuracy (`latent_marginalization_method`)

The inner Gaussian is symmetric, but real latent marginals are often skewed. This
sets how much of that skew you recover.

- `GaussianMarginal()`: just the inner Gaussian. Fastest, but symmetric.
- `SimplifiedLaplace()`, the default: a cheap correction that adds the skewness
  back. Accurate enough for most models.
- `LaplaceMarginal()`: a full Laplace approximation for each component. The most
  accurate and the most expensive. Worth it when the simplified correction isn't
  enough, for instance with strongly non-Gaussian likelihoods or very few
  observations behind each latent value.

```julia
inla(model, y; latent_marginalization_method = LaplaceMarginal())
```

### Hyperparameter exploration (`exploration_strategy`)

This decides where the ``\theta`` points go.

- `AutoExplorationStrategy()`, the default: a dense grid when there are only a few
  hyperparameters, switching to a central composite design (CCD) as the count
  grows.
- `GridExplorationStrategy(integration_step_z = …, max_log_drop = …)`: an explicit
  grid. Lowering `integration_step_z` packs in more points. The default grid is
  fairly coarse, and a denser one noticeably sharpens skewed hyperparameter tails
  (the [Validation](../validation/index.md) page measures exactly this). Each extra
  point costs about one more inner solve.
- `CCDExplorationStrategy()`: forces the CCD design. The practical choice once you
  have several hyperparameters and a full grid gets too expensive.

```julia
inla(model, y; exploration_strategy = GridExplorationStrategy(integration_step_z = 0.5))
```

### Mode finding (`mode_init`, `mode_diagnostic`)

The ``\theta`` points are placed around the mode of ``p(\theta \mid y)``. On
difficult posteriors (multimodal, very skewed, nearly degenerate) the optimiser
can settle on a poor mode. Latte checks for this afterwards and, by default, warns
you if it finds a better point on the grid (`mode_diagnostic = :warn`). When that
happens, give the search a better starting point, or a few of them, through
`mode_init`.

### Other knobs

- `latent_indices` computes marginals only for the components you ask for, which
  saves a lot of time on large latent fields.
- DIC, WAIC, CPO and the marginal likelihood are computed along the way by default,
  through `accumulators`.

## Reference

```@docs
inla
```

## Limits

The [Validation](../validation/index.md) page measures all of these.

- **Skewed hyperparameter tails.** The coarse default grid does not resolve them
  well; densify it with `GridExplorationStrategy(integration_step_z = …)`. For
  non-Gaussian likelihoods a small gap survives even then, and that part comes from
  the inner Laplace approximation rather than the grid.
- **Non-identified hyperparameters.** When the data only pin down a combination of
  hyperparameters (say ``\sigma^2 + 1/\tau`` in a Gaussian-Gaussian IID model), the
  outer surface cannot follow the degenerate ridge, so those marginals drift even
  though the inner Laplace matches the exact posterior. A sampler like HMC-Laplace
  handles this case.
- **Many hyperparameters.** The full grid grows exponentially in their number. The
  `Auto`/CCD path softens that, and HMC-Laplace scales better when ``\theta`` is
  large.

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

- Tutorials: [Getting started](../tutorials/getting_started.md) and
  [Spatial disease mapping](../tutorials/disease_mapping_spatial.md).
- [Benchmarks](../benchmarks/index.md): speed against the other engines and R-INLA.
- [Validation](../validation/index.md): calibration of every engine.
- [Main Interface](@ref main-interface): defining models and working with results.
