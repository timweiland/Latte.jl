# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Latte.jl is a Julia package for approximate Bayesian inference in **latent Gaussian
models** (LGMs) — models that factor as `p(θ) · p(x|θ) · p(y|x,θ)` with `x | θ` a
Gaussian Markov random field. You define a model once with the `@latte` macro (a
`~`-statement DSL whose call returns a `LatentGaussianModel`) and run it through any of
three inference engines that share an inner Laplace approximation and a common result
protocol:

- `inla` — Integrated Nested Laplace Approximation, grid/CCD integration over θ (default engine).
- `tmb` — TMB-style MAP + Laplace (delta-method) covariance.
- `hmc_laplace` — NUTS on the Laplace marginal `L(θ)` (embedded Laplace + HMC, à la Margossian et al. 2020 / tmbstan).

The package is pre-1.0 (`v0.1.0-DEV`); the API is not yet stable.

## Common Commands

### Testing
```bash
make test     # run the full test suite
```
To run a single test file: start `julia --project`, then
`using TestEnv; TestEnv.activate(); include("test/…")`.

### Development
```bash
make setup    # install dev dependencies
make docs     # build the documentation (incremental tutorial rebuild)
make format   # format the whole repo with Runic
```

**Automated formatting**: a Claude Code hook (`.claude/hooks.json`) runs the formatter
after editing `.jl` files (Edit/Write/MultiEdit). Run `make format` on the whole repo —
don't target individual files.

## TTFX (cold-start) measurement

Single-number harness for tracking cold-start UX:

```bash
# Spawns a fresh Julia, loads Latte, runs `latte_from_dppl(...)` + `inla(...)`.
# Reports load_s + adapter_s + first_inla_s = total_cold_s.
julia --project benchmark/ttfx/measure_ttfx.jl

# Save JSON for diffing across attempts:
julia --project benchmark/ttfx/measure_ttfx.jl --json /tmp/ttfx.json
```

The package-level `@compile_workload` (`src/precompile_workloads.jl`) covers the
post-LGM pipeline for Poisson/Bernoulli/Binomial/Normal fast-path likelihoods — that's
what makes `inla(lgm, y)`'s first call fast on a fresh process. It does NOT cover
`latte_from_dppl` for arbitrary user `@model` types (DPPL specialises on the model type,
which only exists at user-call time). Users who ship Latte inside an application package
can call `Latte.warmup(model, y; random=...)` from their own `@compile_workload` to bake
in the per-model specialisation as well.

## Architecture

`src/LAYOUT.md` is the architectural reference — read it first. The organising idea:
shared LGM infrastructure (the inner Laplace, the model object, the result protocol)
lives *above* the inference algorithms that use it, rather than buried inside any one
engine. Top-level `src/`:

- `dsl/` — the `@latte` macro: AST recognition, factor-graph extraction, and the
  DynamicPPL adapter (`latte_from_dppl`). Recognition is macro-pure (no GMRF back-reference).
- `model/` — the `LatentGaussianModel`, hyperparameter specs (`@hyperparams`,
  natural/working space), and the augmented-latent machinery.
- `inference/` — the engines, each in its own subdirectory: `inla/` (mode finding,
  grid exploration, interpolation, the INLA result), `tmb/`, `hmc_laplace/`.
- `laplace/` — the inner Laplace approximation and latent marginalization (`marginalize`,
  Gaussian / simplified-Laplace / spline-augmented methods).
- `posterior/` — the shared `InferenceResult` protocol and the engine-agnostic accessors
  (`latent_marginals`, `hyperparameter_marginals`, `derived`, `diagnose`,
  `linear_predictor_marginals`, `observation_marginals`, …).
- `distributions/` — custom distributions (e.g. transformed/weighted mixtures).
- `diagnostics/` — the PSIS `diagnose` check and SBC calibration.
- `differentiation/`, `parallel/`, `utils/` — sparse-AD strategies, parallel helpers,
  and shared utilities. `precompile_workloads.jl` / `warmup.jl` handle cold start.

Observation models, latent-model constructors (`IIDModel`, `RWModel`, `BesagModel`,
`MaternModel`, `BYM2Model`, `SeparableModel`, …), Gaussian approximation, and link
functions come from **GaussianMarkovRandomFields.jl (v0.12)** and are re-exported by
Latte, so `using Latte` is enough to name them in an `@latte` body.

## Key Dependencies

- **GaussianMarkovRandomFields.jl** (v0.12): latent-model constructors, observation
  models, Gaussian approximation, and sparse-AD support — re-exported.
- **DynamicPPL.jl**: the probabilistic-programming layer the `@latte` macro builds on.
- **AdvancedHMC.jl**: NUTS sampler for `hmc_laplace`.
- **DifferentiationInterface / SparseConnectivityTracer / SparseMatrixColorings /
  ForwardDiff / FiniteDiff / ReverseDiff**: sparse Hessian/gradient automatic differentiation.
- **LinearSolve.jl, SelectedInversion.jl**: sparse linear solves and selected inversion.
- **Optim.jl, Roots.jl**: mode finding and root solving.
- **DataInterpolations.jl, FastGaussQuadrature.jl, HCubature.jl**: interpolation and
  quadrature for hyperparameter marginals.
- **Turing.jl, JLD2.jl, Aqua.jl** (test deps): MCMC cross-validation, serialization,
  and code-quality checks.

## Development Notes

- CI (`.github/workflows/CI.yml`) runs three jobs on **Julia 1.12**: the test suite, a
  docs build + deploy, and an analytic-conjugate correctness gate (`benchmark/validate_ci.jl`).
  Executed tutorial outputs are cached (rolling-week key) to keep the docs build fast.
- Aqua.jl enforces code quality (no method ambiguities, no undocumented exports, …) inside
  the suite.
- Exports are organised per module rather than centralised. All public functions carry
  docstrings, rendered in the reference via `@docs` blocks.
- `src/LAYOUT.md` documents the intended layout; keep it in sync with structural changes.

## Documentation Structure

Docs build with Documenter.jl + DocumenterVitepress and deploy from `main`
(`DocumenterVitepress.deploydocs(devbranch="main")`). Tutorials are Literate.jl sources
under `docs/src/literate-tutorials/`, executed and baked at build time (outputs are
gitignored). The reference lives under `docs/src/reference/`:

- `index.md` — overview / orientation,
- `latte.md` — the `@latte` macro, factor-graph compilation, and its limits,
- `results.md` — the result protocol and accessor functions,
- `lower_level.md` — `@hyperparams` and direct `LatentGaussianModel` construction,
- plus component pages (observation models, Gaussian approximation, marginalization,
  hyperparameter posterior).

Engine pages are under `docs/src/engines/`; the landing page, benchmarks, and validation
pages render Vue components under `docs/src/components/`.

## Task system

This project tracks development work with an org-mode task system in the `tasks/`
directory. Each task file has: Overview, Subtasks (TODO/DONE), Acceptance Criteria
(`- [ ]` / `- [x]`), Implementation Plan, and Implementation Notes.

```bash
cat tasks/<task>.org        # view a task
grep -r "TODO" tasks/       # find open work
```

A task is done when all acceptance criteria are checked, implementation notes document
the approach, the tests pass and the code is formatted, and the relevant docs are updated.
