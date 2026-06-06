# [INLA](@id engine-inla)

Integrated Nested Laplace Approximation — Latte's flagship engine, and the method
the package is named for. It computes posterior marginals for latent Gaussian
models **deterministically** and **fast**, with no MCMC. The entry point is
[`inla`](@ref).

## What it does

A latent Gaussian model factorises as

```math
p(\theta, x, y) = \underbrace{p(\theta)}_{\text{hyperprior}}\;
                  \underbrace{p(x \mid \theta)}_{\text{Gaussian latent (GMRF)}}\;
                  \underbrace{p(y \mid x, \theta)}_{\text{cond. indep. observations}}
```

with a Gaussian latent field ``x`` (a GMRF), a *small* number of hyperparameters
``\theta``, and conditionally-independent observations. INLA targets the marginals
``p(\theta_j \mid y)`` and ``p(x_i \mid y)`` directly, through two nested
approximations:

1. **Inner Laplace.** For a fixed ``\theta``, the conditional ``p(x \mid y, \theta)``
   is approximated by a Gaussian at its mode (a Laplace approximation). For Gaussian
   observations this is *exact*; otherwise it is the Gaussian matching the mode and
   curvature of the log-posterior.
2. **Outer grid.** The hyperparameter posterior
   ``p(\theta \mid y) \propto p(\theta)\,p(y \mid \theta)`` — with the marginal
   likelihood ``p(y \mid \theta)`` coming from that inner Laplace — is explored on a
   grid: a handful of points for one or two hyperparameters, a central-composite
   design as the dimension grows.

Latent marginals are then obtained by integrating the inner approximation over the
grid of ``\theta`` (optionally refined per grid point with a Laplace correction for
skew). Every step is deterministic numerical integration, so you get the same
answer every run and zero sampling noise.

## When to reach for it

INLA is the right default for latent Gaussian models with **few hyperparameters**
(roughly ``\le 5``) when you want **speed and determinism** — GLMMs, spatial and
temporal models, smoothing, disease mapping. See [Benchmarks](../benchmarks/index.md)
for wall-clock and the [Validation](../validation/index.md) page for calibration.

| reach for | when |
|-----------|------|
| **INLA** | standard LGM, low-dimensional ``\theta``, want fast deterministic marginals |
| **TMB** | the fastest approximate fit, ``\theta`` posterior close to Gaussian |
| **HMC-Laplace** | ``\theta`` higher-dimensional or awkward (skewed, correlated, ridged) — where the grid gets expensive or struggles |

## Using `inla`

```@docs
inla
```

Useful options: `marginalization_method` (`GaussianMarginal()` for speed,
`LaplaceMarginal()`/the simplified-Laplace for skew), `exploration_strategy`
(densify or switch the hyperparameter grid), `latent_indices` (compute marginals
only where you need them on large models), and `progress`.

## Limits & caveats

These are *measured*, not asserted — see the [Validation](../validation/index.md) page:

- **Skewed hyperparameter tails at the default grid.** The coarse default grid
  leaves a small gap on a skewed ``\theta`` marginal (KS ``\approx`` 0.03–0.07 in
  our SBC); a finer `integration_step_z` closes most of it. The remainder, for
  non-Gaussian likelihoods, is the Laplace marginal-likelihood approximation.
- **Non-identified / ridged hyperparameter posteriors.** When the data identify
  only a combination of hyperparameters (e.g. ``\sigma^2 + 1/\tau`` in a
  Gaussian–Gaussian IID model), the grid cannot cover the degenerate ridge and the
  marginals there are off — even though the inner Laplace and the *exact* posterior
  agree. A faithful sampler (HMC-Laplace) recovers it; reach for that on such
  models.
- **Hyperparameter count.** Grid cost grows with ``|\theta|``; beyond ``\approx 5``
  the `Auto` strategy switches to CCD, and HMC-Laplace becomes the better choice.

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
