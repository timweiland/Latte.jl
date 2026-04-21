# Latte — proposed `src/` layout

Design doc for Phase 3 of the rename task. Goal: restructure `src/` so that
**shared latent-Gaussian-model infrastructure** lives above the inference
algorithms that use it, rather than being buried in INLA-specific directories.

Living document — refine iteratively before executing Phase 4.

## Mental model

The LGM factors as `p(θ) · p(x|θ) · p(y|x,θ)`. That's the math object. A
**method** gives you some approximation to the posterior. Several methods
share the same *inner* Laplace machinery but differ on the *outer* treatment
of θ:

| method | outer treatment of θ | uses inner Laplace `q(x|θ)` ? |
|---|---|---|
| INLA | grid / CCD / interpolation | yes |
| TMB | point MAP + Hessian | yes |
| HMC-Laplace (tmbstan) | NUTS on `L(θ)` | yes |
| full HMC *(future)* | NUTS on joint `(θ, x)` | no |
| VI *(future)* | variational family | no |

The inner Laplace is **not** an INLA feature. It's shared machinery across
three methods. Concretely, today these things live under
`src/hyperparameter_posterior/` or `src/latent_marginalization/` as if they
were INLA internals — but they aren't:

- `find_hyperparameter_mode` — used by all three Laplace-based methods
- `hyperparameter_logpdf` (= `L(θ)`) — used by all three
- `gaussian_approximation` (GMRFs.jl) + `selinv_mat` — used by all three
- **Simplified Laplace** — a correction to `q(x|θ)` marginals.
  Orthogonal to the outer θ treatment. A TMB user gets
  skew-corrected x-marginals for free.
- **Full Laplace** (per-site) — same story.
- **PSIS-k̂** — diagnoses the quality of the Laplace approximation itself.
  Applies to any Laplace-based method.

## What *is* INLA-specific

The "I" in INLA is integration over θ. These pieces stay INLA-specific:

- Grid / CCD exploration of θ-space
- Reparameterisation-via-Hessian-eigen for the grid
- Posterior interpolation over grid points
- HCubature integration of the interpolant → hyperparameter marginals
- Mixture-over-grid latent marginals (the `WeightedMixture` assembly)
- `INLAResult` itself

TMB is tiny (Hessian at MAP + sdreport shape). HMC-Laplace is a thin
AdvancedHMC wrapper with TMB warm-start.

## What's method-agnostic post-processing

Most of what's in `main_interface/` today isn't INLA-specific at all — it
operates on "some posterior over `(x, θ)`":

- WAIC / CPO — pointwise likelihood aggregation
- `predict`, `observation_marginals` — observation model composed with x-marginals
- `linear_combinations` — affine transforms of x-marginals
- posterior sampling — any fitted posterior
- BMA — across inference results (currently ties to `INLAResult` but shouldn't have to in principle)

The right abstraction is a lightweight **protocol** over result types:
`latent_marginals(result)`, `hyperparameter_marginals(result)`,
`hyperparameter_mode(result)`, `rand(result)`, etc. Each inference method
produces a struct implementing the protocol; accumulators / predict / etc.
dispatch on it.

## Terminology

Latte uses **hyperparameters** (θ) and **latent variables** / **latent field**
(x) throughout. This matches:

- the INLA literature (Rue-Martino-Chopin 2009, R-INLA)
- `GaussianMarkovRandomFields.jl` (our core dependency)
- our own top-level type name, `LatentGaussianModel`

We deliberately **don't** adopt TMB's "fixed effects" / "random effects"
terminology at the protocol level. In TMB those terms mean "parameters
*not* integrated out" vs "parameters integrated out via Laplace" — a
machinery-defined split that aligns with our θ/x split. But they collide
with the GLMM meaning (population-level β vs group-level u) where both
live in the latent field x in an INLA-style model. Adopting TMB's names
would inherit that ambiguity; keeping "hyperparameters" / "latent" keeps
the distinction crisp.

Where it helps TMB users, `TMBResult` exposes aliases locally:

```julia
fixed_effects(r::TMBResult)  = hyperparameter_marginals(r)
random_effects(r::TMBResult) = latent_marginals(r)
fixef(r::TMBResult)          = hyperparameter_marginals(r)
ranef(r::TMBResult)          = latent_marginals(r)
```

Core protocol methods, docs, and non-TMB result types stay on the
hyperparameter / latent terminology.

## `InferenceResult` protocol (locked)

```julia
abstract type InferenceResult end

struct INLAResult        <: InferenceResult; ... end
struct TMBResult         <: InferenceResult; ... end
struct HMCLaplaceResult  <: InferenceResult; ... end
```

### Tier 1 — universal must-haves

```julia
# Marginals: always Vector{<:Distribution}, positional by construction.
latent_marginals(r)           :: Vector{<:Distribution}
hyperparameter_marginals(r)   :: Vector{<:Distribution}

# Name → slice mapping. Scalar hp τ → groups[:τ] == 3:3.
# Vector hp β (length p) → groups[:β] == 1:p. One mental model, no
# special-casing for scalar vs vector parameters.
latent_groups(r)              :: OrderedDict{Symbol, UnitRange{Int}}
hyperparameter_groups(r)      :: OrderedDict{Symbol, UnitRange{Int}}

# By-name slice — always returns a Vector (length 1 for scalar params);
# user calls only(...) to unwrap when they know it's scalar.
latent_marginals(r, name::Symbol)          :: Vector{<:Distribution}
hyperparameter_marginals(r, name::Symbol)  :: Vector{<:Distribution}

# Point summaries
hyperparameter_mode(r)        :: NaturalHyperparameters

# Posterior sampling
Random.rand(r)                :: NamedTuple{(:θ, :x), Tuple{Vector, Vector}}
Random.rand(r, n::Int)        :: PosteriorSamples

# Access to inputs
model(r)                      :: LatentGaussianModel
observations(r)               :: AbstractVector

# Meta
converged(r)                  :: Bool
time_elapsed(r)               :: Float64
```

### Tier 2 — method-dependent semantics, returned from the same symbol

```julia
# Float64 when the method has a natural way to produce an approximation
# to log p(y); nothing otherwise (e.g., HMC without bridge sampling).
# All methods produce an approximation (INLA is a doubly-approximate grid
# integral of a Laplace-approximated integrand; TMB is singly-approximate
# Laplace at MAP; bridge-sampled HMC is Monte-Carlo-approximate). No
# _estimate variant — the docstring on each method's implementation
# documents which approximation it computes.
log_marginal_likelihood(r)    :: Union{Float64, Nothing}
```

VI's ELBO is **not** `log_marginal_likelihood` — it's a lower bound, not
an estimate. When VI lands, it gets its own `elbo(r::VIResult)` method.

### Derived (method-agnostic, defined once on `InferenceResult`)

```julia
predict(r, new_df)
linear_combinations(r, A)
observation_marginals(r)
waic(r)
cpo(r)
model_average(rs::Vector{<:InferenceResult})
```

### Tier 3 — method-specific, NOT part of the protocol

```julia
# HMC only
samples(r::HMCLaplaceResult)      # θ chain
rhat(r::HMCLaplaceResult)         # R̂ per dimension
ess(r::HMCLaplaceResult)          # effective sample size
divergences(r::HMCLaplaceResult)  # count

# TMB only
standard_errors(r::TMBResult)
joint_covariance(r::TMBResult)

# INLA only
grid(r::INLAResult)               # exploration grid
exploration_method(r::INLAResult)
```

A call like `rhat(::INLAResult)` should be a `MethodError`, not return
`NaN` or similar. Keeps the universal protocol principled.

### `PosteriorSamples`

`rand(r, n)` returns a small struct rather than a `NamedTuple`, to make
the row-alignment contract explicit (θ[i, :] and x[i, :] come from the
same joint draw):

```julia
struct PosteriorSamples{Θ, X}
    θ::Θ   # (n_samples, n_hp) matrix
    x::X   # (n_samples, n_latent) matrix
end

Base.length(s::PosteriorSamples) = size(s.x, 1)
Base.iterate(s::PosteriorSamples, i=1) = i > length(s) ? nothing :
    ((θ = s.θ[i, :], x = s.x[i, :]), i+1)
```

Same-row-count contract enforced at construction. `collect(rand(r, n))`
decomposes into a `Vector{NamedTuple{(:θ, :x)}}` when per-sample iteration
is what the caller wants. For `n == 1`, `rand(r)` returns a plain
`NamedTuple{(:θ, :x)}` of vectors (matches Julia's `rand(dist)` vs
`rand(dist, n)` convention).

## Proposed layout

```
src/
├── Latte.jl
├── model/                         ← the math object
│   ├── hyperparameters.jl          (HyperparameterSpec, Working/Natural, @hyperparams)
│   ├── latent_gaussian_model.jl
│   ├── function_latent_model.jl
│   ├── augmentation.jl             (AugmentedLatentModel, AugmentationInfo)
│   ├── observation_models.jl       (link_to_bijector, AD helpers — thin shim over GMRFs.jl)
│   └── sampling.jl                 (rand(::LGM))
├── laplace/                       ← shared inner machinery
│   ├── gaussian_approximation.jl   (thin wrapper; most logic in GMRFs.jl)
│   ├── hyperparameter_logpdf.jl    (L(θ))
│   ├── mode_finding.jl             (find_hyperparameter_mode)
│   ├── marginals/
│   │   ├── gaussian.jl              ← from latent_marginalization/gaussian_marginal.jl
│   │   ├── simplified_laplace.jl    ← SLA; TMB and HMC-Laplace can use too
│   │   └── full_laplace.jl          ← per-site; any method can use
│   └── types.jl                    (MarginalApproximation abstract, dispatch protocol)
├── differentiation/               (AD strategies — unchanged)
├── parallel/                      (threading — unchanged)
├── dsl/                           ← DPPL → LGM adapter (from prototypes)
│   ├── structure_probing.jl
│   ├── dag_extraction.jl
│   ├── latent_prior.jl
│   ├── obs_model.jl
│   ├── hp_spec.jl
│   ├── pattern_augment.jl
│   └── adapter.jl                  (latte_from_dppl)
├── inference/                     ← method-specific
│   ├── inla/
│   │   ├── exploration/            (grid, CCD, transformation, utils — moved as-is)
│   │   ├── interpolation.jl
│   │   ├── hp_marginals.jl
│   │   ├── latent_marginals/       (spline-based grid-mixture builders)
│   │   ├── types.jl                (INLAResult)
│   │   ├── inference.jl            (inla())
│   │   └── validation.jl
│   ├── tmb/
│   │   ├── types.jl                (TMBResult)
│   │   └── inference.jl            (tmb())
│   └── hmc_laplace/
│       ├── types.jl                (HMCLaplaceResult)
│       └── inference.jl            (hmc_laplace())
├── diagnostics/                   ← cross-cutting
│   ├── gpd_fit.jl                  (vendored Zhang-Stephens, MIT attribution)
│   ├── psis.jl
│   └── laplace_quality.jl          (PSIS-k̂ on any Laplace-based result)
├── posterior/                     ← method-agnostic post-processing
│   ├── result_protocol.jl          (latent_marginals, hp_marginals, ...)
│   ├── accumulators/               (WAIC, CPO)
│   ├── predict.jl
│   ├── observation_marginals.jl
│   ├── linear_combinations.jl
│   ├── model_averaging.jl          (BMA — method-agnostic; only needs Tier 1 protocol)
│   └── sampling.jl                 (rand on any InferenceResult)
├── distributions/                 (WeightedMixture)
└── utils/                         (selinv, owens_t, kld, dist_summaries, plotting_stubs)
```

## Mapping from current layout

| current | proposed |
|---|---|
| `hyperparameters/` | `model/hyperparameters.jl` |
| `inla_model.jl` | `model/latent_gaussian_model.jl` (+ split out `FunctionLatentModel`) |
| `latent_augmentation/` | `model/augmentation.jl` |
| `observation_models/` | `model/observation_models.jl` |
| `latent_marginalization/gaussian_marginal.jl` | `laplace/marginals/gaussian.jl` |
| `latent_marginalization/simplified_laplace.jl` | `laplace/marginals/simplified_laplace.jl` |
| `latent_marginalization/laplace/` | `laplace/marginals/full_laplace.jl` (consolidate) |
| `latent_marginalization/adaptive_marginal.jl` | `inference/inla/latent_marginals/` (INLA-specific adaptive choice per θ) |
| `latent_marginalization/types.jl` | `laplace/types.jl` (protocol abstracts) |
| `hyperparameter_posterior/mode_finding.jl` | `laplace/mode_finding.jl` |
| `hyperparameter_posterior/exploration/` | `inference/inla/exploration/` |
| `hyperparameter_posterior/marginalization/` | `inference/inla/latent_marginals/` + `inference/inla/hp_marginals.jl` |
| `hyperparameter_posterior/ccd_interpolant.jl` | `inference/inla/interpolation.jl` |
| `hyperparameter_posterior/spline_marginal_builders.jl` | `inference/inla/latent_marginals/` |
| `main_interface/inla_inference.jl` | `inference/inla/inference.jl` |
| `main_interface/types.jl` (INLAResult) | `inference/inla/types.jl` |
| `main_interface/validation.jl` | `inference/inla/validation.jl` |
| `main_interface/progress.jl` | `inference/inla/progress.jl` (still INLA-specific phases) |
| `main_interface/posterior_sampling.jl` | `posterior/sampling.jl` |
| `main_interface/predict.jl` + `prediction.jl` | `posterior/predict.jl` |
| `main_interface/linear_combinations.jl` | `posterior/linear_combinations.jl` |
| `main_interface/observation_marginals.jl` | `posterior/observation_marginals.jl` |
| `main_interface/model_averaging.jl` | `posterior/model_averaging.jl` (BMA only needs Tier 1 protocol; method-agnostic) |
| `posterior_accumulators/` | `posterior/accumulators/` |
| `distributions/` | `distributions/` (unchanged) |
| `utils/` | `utils/` (unchanged) |
| `differentiation/` | `differentiation/` (unchanged) |
| `parallel/` | `parallel/` (unchanged) |

New directories populated by Phase 5 prototype migration:
- `dsl/` — from `prototypes/latte_mvp/dppl_unified_adapter.jl`
- `inference/tmb/` — from `prototypes/latte_mvp/tmb_from_dppl.jl`
- `inference/hmc_laplace/` — from `prototypes/latte_mvp/hmc_laplace.jl`
- `diagnostics/` — from `prototypes/latte_mvp/psis_diagnostic.jl`

## Non-obvious moves

1. **`latent_marginalization/` gets dissolved.** Its contents split: the
   Gaussian approximation wrapper goes to `laplace/`, SLA/full Laplace go to
   `laplace/marginals/` (no longer buried under "INLA latent marginalization").
   `adaptive_marginal.jl` is the INLA-specific adaptive-per-θ-point choice
   and stays under `inference/inla/latent_marginals/`.

2. **`hyperparameter_posterior/mode_finding.jl` → `laplace/mode_finding.jl`.**
   Not INLA-only; TMB and HMC-Laplace both need it.

3. **`main_interface/` gets dissolved.** `inla_inference.jl`, `types.jl`,
   `validation.jl`, `progress.jl` → `inference/inla/`. Everything else →
   `posterior/`.

4. **PSIS-k̂ as `diagnostics/`, top-level.** Currently only in the prototype.
   It diagnoses the *Laplace approximation quality*, so it's a cross-cutting
   concern over INLA, TMB, and HMC-Laplace — not buried under any one.

5. **`posterior_accumulators/` → `posterior/accumulators/`.** WAIC/CPO are
   method-agnostic; clarify that.

## Design decisions (previously open, now locked)

### 1. Protocol vs supertype for `InferenceResult` — **abstract supertype**

`abstract type InferenceResult end`, with `INLAResult`, `TMBResult`,
`HMCLaplaceResult` subtyping it. Rich contract justifies one explicit
place to hang documentation and a single method definition point for
fallbacks. Full protocol specified in the "InferenceResult protocol"
section above.

### 2. `laplace/` vs `approximations/` — **`laplace/`**

Matches what's in there today. If VI lands later we can rename to
`approximations/laplace/`, `approximations/vi/`. YAGNI for now.

### 3. Where does `observation_models/` go? — **`model/observation_models.jl`**

Thin shim (link-to-bijector + AD helpers) since the heavy lifting is in
GMRFs.jl. Part of the model specification, not its own concern.

### 4. BMA placement — **`posterior/model_averaging.jl`**

BMA only needs Tier 1 protocol (`hyperparameter_marginals`,
`latent_marginals`, `log_marginal_likelihood`). Method-agnostic in
principle. The INLA-grid assumption in the current code is an
implementation detail to generalise, not an intrinsic BMA property.

### 5. `distributions/WeightedMixture` — **keep at top-level `distributions/`**

Standalone distribution type usable beyond INLA (e.g., HMC-Laplace
producing a per-sample mixture of x-marginals). Leave the door open.

### 6. Progress infrastructure — **per-method**

Keep under `inference/*/progress.jl`. INLA's progress is phase-aware
(mode → exploration → marginalization). TMB and HMC-Laplace progress is
trivial enough that shared infrastructure would be over-engineering.

### 7. `pointwise_loglik`-style interfaces — **no move, already right**

WAIC/CPO rely on pointwise likelihoods. That's a method-agnostic
requirement on the model spec (observation model territory), not the
inference result. Already in the right place.

### 8. Marginal shape — **uniform `Vector{<:Distribution}` + name-to-slice map**

Same design for both hyperparameters and latent variables. Full spec in
the "InferenceResult protocol / Tier 1" section. Rationale: TMB
hyperparameter dimensions can be large (e.g., p-dim fixed-effect vectors
promoted to θ); `NamedTuple{names, Tuple{...}}` kills compile time at
scale. One pattern everywhere is cleaner than two.

### 9. `log_marginal_likelihood` shape — **single method, `Float64` or `nothing`**

All methods produce approximations; `_estimate` would be a false
distinction. One method, documented per implementation. VI's ELBO is a
different quantity — gets its own `elbo(r)`. Full spec in the
"InferenceResult protocol / Tier 2" section.

### 10. `rand(r, n)` shape — **`PosteriorSamples` struct**

Row-alignment of θ and x matters and shouldn't live only in a docstring.
`rand(r)` (singular) returns a `NamedTuple{(:θ, :x)}` of vectors; `rand(r, n)`
returns `PosteriorSamples`. Full spec in the "InferenceResult protocol"
section.

### 11. Terminology — **hyperparameters / latent**

Full discussion in the "Terminology" section. TMB-style aliases
(`fixed_effects`, `random_effects`, `fixef`, `ranef`) exposed on
`TMBResult` only.

## Interface compatibility with StatsAPI

`StatsAPI.jl` (the lightweight abstract) + `StatsBase.jl` (batteries) is the
de facto Julia protocol for statistical models. `GLM.jl`, `MixedModels.jl`,
`Survival.jl`, etc. all subtype `StatisticalModel` / `RegressionModel` and
implement the core methods:

```
nobs, loglikelihood, dof, deviance, nulldeviance, aic, bic, r2
coef, coefnames, coeftable, stderror, vcov, confint, informationmatrix
fitted, residuals, predict, modelmatrix, response, weights
```

### Fits cleanly for `InferenceResult`

- `nobs(r)` = `length(observations(r))`
- `response(r)` = `observations(r)`
- `predict(r, new_df)` — already on our interface
- `fitted(r)` = `mean.(observation_marginals(r))`
- `residuals(r)` = `response(r) .- fitted(r)`
- `loglikelihood(r)` = `log p(y | θ̂, x̂)` at the joint MAP. Well-defined for
  TMB; slightly unusual but still computable for INLA / HMC-Laplace.

### Awkward or doesn't fit

- `coef` / `stderror` / `vcov` assume *point estimates with frequentist
  SEs*. Latte's output is posterior marginals. Shape matches; semantics
  don't. TMB is the one method where these map 1:1. INLA users want
  **credible intervals from marginals**, not SEs of an estimator.
- `coefnames` assumes fixed-effect naming. Without a formula interface,
  Latte doesn't distinguish "fixed" vs "random" in the latent field.
- `dof`, `deviance`, `nulldeviance`, `r2` — not meaningful for arbitrary
  LGMs. Would have to return `missing` or throw.
- `confint` blurs Bayesian credible intervals with frequentist CIs.
  `Distributions.quantile` on our marginals is the cleaner interface for
  the Bayesian case.

### Ecosystem pattern

- Regression-flavour packages (GLM, MixedModels) subtype `StatisticalModel`
  / `RegressionModel` — they're genuinely frequentist point-estimate-with-
  SE tools.
- Bayesian PPL packages (Turing, MCMCChains, ArviZ) don't subtype StatsAPI
  — their abstraction is samples or chains, not a fitted model in the
  StatsAPI sense.

Latte sits between. The typed, structured side (LGM with identifiable
hyperparameters) makes StatsAPI integration tempting; the Bayesian side
makes `coef` / `stderror` / `confint` semantically wrong for three of our
four methods.

### Decision: partial adoption via method overloads, no subtyping

- **Implement where semantics are clean**: `nobs`, `response`, `fitted`,
  `residuals`, `predict`, `loglikelihood`. These give real ecosystem
  compatibility (plotting, diagnostics, users transitioning from GLM)
  without over-claiming.
- **Don't subtype `StatisticalModel`.** Subtyping implies the full
  contract holds; it doesn't.
- **Skip `coef` / `stderror` / `vcov` / `coefnames` / `coeftable` on the
  abstract.** TMB can expose them on `TMBResult` directly — they make
  genuine sense there.
- **`confint`**: implement as `quantile`-based credible intervals on each
  marginal, not as a point-estimate CI. Semantically honest because it
  operates on our marginals (distributions), not on a point estimate with
  a covariance.
- **Formula interface (future task)**: a wrapper `FormulaResult <:
  RegressionModel` can expose the full frequentist contract. That's the
  right level to commit to `coef` / `stderror` / `coeftable` — when the
  user has opted into a regression-shaped view of their model.

The clean story: **Tier 1 protocol** (`latent_marginals`,
`hyperparameter_marginals`, `rand`, etc.) is Latte-native and Bayesian-
shaped; we implement **the subset of StatsAPI that's semantically honest**
on top of it.

## Phase 4 execution notes (for later)

- Execute as a sequence of small commits: one subdirectory move at a time,
  `make test` green after each.
- Git-wise: use `git mv` to preserve history per file.
- Keep `src/Latte.jl` as the single `include` orchestrator; expect the
  include order to change (e.g., `laplace/` must include after `model/`
  but before `inference/`).
- Defer any API changes (new function names, new exports, interface
  adjustments) until Phase 5 — Phase 4 is *pure relocation*.
- The `hyperparameter_posterior/hyperparameter_posterior.jl` and
  `latent_marginalization/marginalization_module.jl` orchestrator files
  get deleted — their work subsumes into `src/Latte.jl`.
