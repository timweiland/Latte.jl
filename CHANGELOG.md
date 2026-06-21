# Changelog

Notable changes to Latte.jl. The project will follow [Semantic Versioning](https://semver.org/)
from 1.0 onward; while pre-1.0, minor releases may carry breaking changes.

## [Unreleased]

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
