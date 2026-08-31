# Changelog

Notable changes to Latte.jl. The project will follow [Semantic Versioning](https://semver.org/)
from 1.0 onward; while pre-1.0, minor releases may carry breaking changes.

## [0.2.0] - 2026-08-31

### Added

- Second-order hyperparameter mode finding: passing an Optim second-order method
  (e.g. `mode_method = NewtonTrustRegion()`) now works and builds its model Hessian
  from forward differences of AD gradients every `hessian_refresh` accepted
  iterates (dimension-aware default: every iterate for `dim(θ) ≤ 2`, where a
  refresh costs at most two gradient evaluations and secant updates go stale
  near the mode; every 5 above), with SR1 secant updates from every gradient
  evaluation in between. Trust-region steps stay bounded, so
  no line search and none of its extreme-θ failure modes. Requires `ADStrategy`
  (the default differentiation strategy). The model Hessian at the accepted mode is
  handed to exploration (`mode_info.negative_hessian`), which reuses it as the
  reparameterization curvature instead of re-differencing gradients around θ*.
- Stall detection in mode finding: when `stall_iterations` consecutive outer
  iterations improve the objective by less than `stall_f_tol` (about the inner-solve
  noise floor), the optimization stops early with a diagnostic warning and
  `mode_info.stalled = true` instead of spending the remaining `mode_iterations`
  budget at the noise floor.
- `RoutedLatentModel` and the pattern-augmentation wrapper forward GMRFs.jl's
  `precision_logdet` structure hook (when the installed GMRFs version provides it),
  so recognized separable/combined priors keep their cheap prior log-determinant —
  without the forwarding, every hyperparameter evaluation pays a joint-scale prior
  factorization. Both forwards are exact: routing only renames hyperparameters, and
  pattern augmentation adds structural zeros.

### Changed

- The default mode-finding method is now resolved per differentiation strategy,
  dimension, and warm-start availability: `NewtonTrustRegion()` for AD gradients,
  `dim(θ) ≤ 6`, and non-augmented models; `BFGS` + backtracking otherwise
  (finite-difference gradients, higher dimension, or augmented models, whose
  disabled inner warm start defeats the trust region's fused evaluations). On the
  benchmark families the trust region is uniformly non-worse and strictly better on
  five of seven (identical optima everywhere); it certifies the gradient tolerance
  where BFGS's line search fails at the objective noise floor. Passing an explicit
  `mode_method` / `method` overrides the resolution; `tmb` inherits the same default.
  A non-converged stop whose gradient norm is near tolerance is now reported at info
  level (the mode is found; only the certificate is out of reach) instead of a
  warning.

### Fixed

- Constrained priors materialized as plain `ConstrainedGMRF` (e.g. a Besag prior
  through `latte_from_dppl` on the AD-gradient path) are no longer misclassified by
  the prior-logdet fast path; the classification is type-based and fails safe to the
  general `logpdf` fallback.
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
