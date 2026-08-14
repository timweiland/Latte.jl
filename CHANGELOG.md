# Changelog

Notable changes to Latte.jl. The project will follow [Semantic Versioning](https://semver.org/)
from 1.0 onward; while pre-1.0, minor releases may carry breaking changes.

## [Unreleased]

### Fixed

- Negative binomial models now support prediction via `missing` observations. The
  observed-data extraction previously produced a plain vector that failed to
  materialize; it now wraps counts in `NegativeBinomialObservations`, and linearly
  transformed observation models delegate the extraction to their base model.
- `predicted_marginals` and `observed_marginals` on the compact (non-augmented)
  linear-predictor path now include the observation offset. Previously the offset
  was silently dropped, biasing predictions at `missing` observations for any model
  with a fixed offset (e.g. exposure terms). `linear_combinations` gains an
  `offsets` keyword to support this.

## [0.1.2] - 2026-08-06

### Added

- Vector-valued hyperparameters ([#41]): a free hyperparameter may carry a continuous
  vector prior — e.g. `κ ~ MvNormal(μ, Σ)` with a non-diagonal covariance — through
  `@hyperparams`, `latte_from_dppl`, and `@latte` (via the `@fixed` marker). Components
  share the joint prior; marginals are reported per coordinate (`κ[1]`, `κ[2]`, …) and
  `hyperparameter_groups` maps the name to its coordinate range. Vector entries admit
  `identity` or `elementwise(f)` transforms. The exponential-family fast path stays
  available when the likelihood is independent of the vector hyperparameter (the usual
  case — a joint prior on latent-precision parameters); a likelihood that depends on one
  falls back to the AD observation model.

[#41]: https://github.com/timweiland/Latte.jl/issues/41

## [0.1.1] - 2026-07-05

### Added

- Nonlinear-in-`x` Gaussian observations are recognized as a nonlinear least-squares
  model and handled with a Gauss–Newton observation Hessian, by default. A curvature
  check routes mildly-curved Normal means down the same path.
- The Gauss–Newton path covers heteroskedastic and hyperparameter-dependent σ,
  hyperparameter-dependent means, and composite observation blocks — including a flowing
  σ bound to a name other than `:σ`.
- `nls = false` on a model function forces the exact full-Hessian observation path.
- Fixed (hyperparameter-independent) GMRF latent priors ([#36]): `@random x ~ g` for a
  runtime `AbstractGMRF` value `g` is recognized as a constant latent prior, with any
  `ConstrainedGMRF` constraint threaded through. This previously failed at model
  construction, because the fixed-prior coercion probed `cov(d)` and hit the GMRF
  dense-covariance guard.
- `diagnose` surfaces the Gauss–Newton observation Hessian.

### Changed

- Observation groups are split by the Normal σ expression.
- Docs deploy to the lattejl.org custom domain.

### Fixed

- The observation fast-path probe punts to AD when it cannot seed the latent.

[#36]: https://github.com/timweiland/Latte.jl/pull/36

## [0.1.0] - 2026-06-25

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
