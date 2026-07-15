# Changelog

Notable changes to Latte.jl. The project will follow [Semantic Versioning](https://semver.org/)
from 1.0 onward; while pre-1.0, minor releases may carry breaking changes.

## [Unreleased]

### Added

- Vector-valued hyperparameters ([#41]): a free hyperparameter may carry a continuous
  vector prior — e.g. `κ ~ MvNormal(μ, Σ)` with a non-diagonal covariance — through
  `@hyperparams`, `latte_from_dppl`, and `@latte` (via the `@fixed` marker). Components
  share the joint prior; marginals are reported per coordinate (`κ[1]`, `κ[2]`, …) and
  `hyperparameter_groups` maps the name to its coordinate range. Vector entries admit
  `identity` or `elementwise(f)` transforms; models with a vector hyperparameter use the
  AD observation model (the exponential-family fast path handles scalar hyperparameters
  only).

[#41]: https://github.com/timweiland/Latte.jl/issues/41

## [0.1.0]

First public release.

### Added

- The `@latte` macro: define a latent Gaussian model from `~` statements; calling the
  resulting function returns a `LatentGaussianModel`.
- Three inference engines over one shared result protocol — `inla` (grid/CCD integration
  over the hyperparameters), `tmb` (MAP plus a Laplace covariance), and `hmc_laplace`
  (NUTS on the Laplace marginal).
- Re-export of the GaussianMarkovRandomFields latent- and observation-model layer, so
  `using Latte` is enough to name `IIDModel`, `BesagModel`, `MaternModel`, and the rest
  in an `@latte` body.
- Engine-agnostic result accessors: `latent_marginals`, `hyperparameter_marginals`,
  `linear_predictor_marginals`, `observation_marginals`, `derived`, and more.
- `diagnose`: a PSIS check on the inner Laplace approximation, uniform across engines.
- Lower-level construction via `@hyperparams` and direct `LatentGaussianModel` assembly.
