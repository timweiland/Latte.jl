# [API reference](@id api-reference)

Latte fits *latent Gaussian models* (LGMs). The primary way to define one is the [`@latte`](latte.md) macro: it reads your model body, classifies each `~` block, and returns a model object. That object is engine-agnostic — the same definition runs through [`inla`](../engines/inla.md), [`tmb`](../engines/tmb.md), or [`hmc_laplace`](../engines/hmc_laplace.md), and every engine returns results through one shared accessor API.

## [Latent Gaussian models](@id latent-gaussian-models)

Every Latte engine targets the same model class:

```math
p(\theta, x, y) = \underbrace{p(\theta)}_{\text{hyperprior}}\;
                  \underbrace{p(x \mid \theta)}_{\text{Gaussian latent field (GMRF)}}\;
                  \underbrace{p(y \mid x)}_{\text{cond.-independent observations}}
```

A Gaussian latent field ``x`` (a GMRF) governed by a few hyperparameters ``\theta``, with observations conditionally independent given the latent field. GLMMs, spatial (Besag, SPDE) and temporal (AR, random-walk) models, splines, and disease mapping all fit this template. You specify the three pieces — the hyperparameter priors, the latent GMRF, and the observation model — once, and any engine consumes it.

## What's here

- [Defining models: the `@latte` macro](latte.md) — the primary entry point, how the macro builds the model, and what it does not yet handle.
- [Working with results](results.md) — the accessors that read latent and hyperparameter marginals off any engine's result.
- [Lower-level construction](lower_level.md) — building a model directly from `@hyperparams` and `LatentGaussianModel` when you need full control.

Component pages cover the individual pieces:

- [Observation models](observation_models.md) — the likelihood ``p(y \mid x)``.
- [Gaussian approximation](gaussian_approximation.md) — the inner Gaussian fit to ``p(x \mid y, \theta)``.
- [Marginalization](marginalization.md) — recovering latent-field marginals.
- [Hyperparameter posterior](hyperparameter_posterior.md) — exploring and marginalizing over ``\theta``.

For worked, end-to-end models see the [tutorials](../tutorials/index.md); for the inference methods themselves and how to tune each, see the [engine pages](../engines/inla.md).
