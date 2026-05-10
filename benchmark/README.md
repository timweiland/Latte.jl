# Latte.jl benchmarks

This directory holds the benchmark suite. The structure separates three
concepts:

- **Scenarios** — model + data + target quantities + comparability metadata.
- **Engines** — Latte INLA / TMB / HMC-Laplace, plus a long-NUTS reference.
  External engines (R-INLA, brms, glmmTMB) plug in later via the same
  `Engine` interface; see `EXTERNAL.md`.
- **Runs** — one `(scenario × engine × mode)` invocation, producing a
  `Result` written to `results/`.

`runbench.jl` is the CLI entry point. It picks `(scenario, engine)` pairs at
invocation time so engine logic is never hard-coded into scenario files.

## Fairness policy

Benchmark credibility lives or dies on stated rules. These are ours.

### Targets

- Each scenario declares the **target quantities** it cares about (posterior
  marginals for fixed effects, the spatial field, predictive intervals, etc.).
  Speed-only comparisons across methods that fit *different targets* don't
  appear in this suite.
- For external comparisons, every scenario × engine pairing carries an
  explicit **comparability label**:
  - `same posterior` — identical likelihood and prior, same parameterisation.
  - `analogue` — similar but not identical; differences documented per pair.
  - `MLE baseline` — engine is frequentist; included as context, not a
    Bayesian comparison.
  - `gold standard` — long-run NUTS reference, used to score accuracy.

### Timing

- **Cold time** includes Julia startup, package load, first-run compilation,
  and the fit itself. The honest number a user sees on first invocation.
- **Warm time** is the median of `N ≥ 3` repeated fits within an
  already-warm process. The "what does another fit cost?" number.
- **Phase timings** (model construction, optimisation/sampling, posterior
  summarisation) are reported per engine where meaningful. A single
  "runtime" number gets disputed; phase timings preempt that.
- Wall time is reported with an uncertainty interval (median + IQR over
  warm runs).

### Tuning

- Engine-specific tuning is **allowed but documented per scenario**. If
  Latte is tuned and brms isn't, that's a credibility hit. Each scenario
  records exactly what knobs were turned.
- Defaults-only mode (every engine on its own defaults) is reported as a
  separate run mode where we have time.

### Failures

- Failures are **recorded, not hidden**. A scenario × engine pair that
  errors, times out, or diverges produces a result with `status: failed`
  and the failure reason — it doesn't crash the run.
- Timeouts are stated per scenario.

### Reproducibility

- Every result records: hardware (CPU, RAM, OS, BLAS threads), software
  versions (Julia, Latte, every engine package), git SHA, RNG seed,
  timestamp, and a hash of the scenario config + data.
- NUTS reference *summaries* (means, SDs, quantiles, ESS, R̂) are committed
  in `references/`. Full sample arrays are not committed; they're emitted
  to a local-ignored cache or a release artifact.
- Results from CI hardware are **not** treated as performance evidence.
  Shared runners are too noisy. CI runs a `quick` smoke subset to detect
  catastrophic regressions only.

### Corrections

If a comparison looks unfair or outdated, open an issue with a reproducible
script. We'd rather know.

## Layout

```
benchmark/
├── Project.toml          # benchmark env (Pkg.develop's the parent package)
├── runbench.jl           # CLI: --suite quick|full|scaling|reference
├── README.md             # you are here
├── EXTERNAL.md           # cross-language (R, Python) benchmark design
├── engines/              # one file per engine, exposes a uniform interface
│   ├── latte_inla.jl
│   ├── latte_tmb.jl
│   ├── latte_hmc_laplace.jl
│   └── nuts_reference.jl
├── scenarios/            # one file per scenario; no engine logic
│   ├── toy_iid_poisson.jl
│   ├── bym_disease_mapping.jl
│   ├── ar1_poisson.jl
│   ├── separable_spacetime.jl
│   └── scaling/
│       ├── n_sweep_spatial_glmm.jl
│       └── theta_dim_sweep.jl
├── utils/
│   ├── timing.jl         # cold/warm/phase timing
│   ├── environment.jl    # capture hardware + software metadata
│   ├── reference_store.jl # load/save NUTS reference summaries
│   └── reporting.jl      # Result struct + JSON serialization
├── references/           # NUTS reference summaries (committed)
│   └── <scenario_id>/{summary.json, diagnostics.json, manifest.json}
└── results/              # benchmark outputs (committed JSON)
    └── YYYY-MM-DD-<host>/
```

## Running

```sh
# Smoke (60-second sanity sweep across every scenario × engine):
julia --project=benchmark benchmark/runbench.jl --suite quick

# The full internal suite:
julia --project=benchmark benchmark/runbench.jl --suite full

# Scaling sweeps:
julia --project=benchmark benchmark/runbench.jl --suite scaling

# Refresh NUTS reference summaries (long-running):
julia --project=benchmark benchmark/runbench.jl --suite reference

# A specific scenario × engine:
julia --project=benchmark benchmark/runbench.jl \
    --scenario bym_disease_mapping \
    --engine latte_inla
```

Results land in `results/YYYY-MM-DD-<host>/<scenario>/<engine>.json`.

## Adding a scenario

1. Create `scenarios/<your_scenario>.jl` exposing a `scenario()` function
   that returns a `Scenario` struct (see `utils/reporting.jl`).
2. Declare which engines you support and which comparability labels apply
   to each external one.
3. Optionally provide `references/<your_scenario>/summary.json` if you want
   accuracy comparison against NUTS.
4. Run `runbench.jl --scenario <your_scenario>` to verify.

## Adding an engine

1. Create `engines/<your_engine>.jl` exposing a `run!(scenario, mode)`
   function returning a `Result`.
2. Make sure it captures phase timings and engine-specific diagnostics.
3. Add it to the registry in `runbench.jl`.

External engines (R, Python) follow `EXTERNAL.md`.
