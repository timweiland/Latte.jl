# External-package benchmarks: design

Cross-language (R, Python) engine comparisons share the same scenario files
and result schema as Latte's internal engines, but their mechanics are
different enough to deserve their own design doc. Nothing here is
implemented yet; this is the blueprint for when external benchmarks are
prioritised post-v0.1.

## Invocation: subprocess, not in-process

External engines run via subprocess: `runbench.jl` writes scenario inputs
to a temp directory, spawns the engine's interpreter (`Rscript`, `python`,
…), parses the JSON it emits.

Why subprocess and not `RCall.jl` / `PyCall.jl`:

- **Honest timing.** The user-facing "what does it cost to fit this
  externally?" includes interpreter startup. In-process tools hide that.
- **Failure isolation.** A crashing R session never takes down the
  benchmark runner.
- **Environment portability.** Each language manages its own deps without
  the runner needing to link against system R/Python. Adding brms / PyMC
  later is copy-paste of the pattern.

The cost: serialization overhead and a second-language script per
scenario. Acceptable for benchmarks; runs are seconds-to-minutes, JSON
overhead is in the milliseconds.

## Scenario files own their external implementations

A scenario file like `scenarios/bym_disease_mapping.jl` declares which
external engines it supports and points at the corresponding scripts:

```julia
function scenario()
    Scenario(
        id = "bym_disease_mapping",
        ...
        external_implementations = (
            r_inla = (
                script = joinpath(@__DIR__, "bym_disease_mapping.R"),
                target = :r_inla,
                comparability = "same posterior",
                comparability_notes = "Identical likelihood. Same Besag prior on the spatial field. Default INLA precision priors are PC by default in INLA ≥ 23.x.",
            ),
            brms = (
                script = joinpath(@__DIR__, "bym_disease_mapping.R"),
                target = :brms,
                comparability = "analogue",
                comparability_notes = "Same likelihood, different prior parameterisation. brms uses Gamma(0.01, 0.01) by default; we set PCPrior in the script for closer match.",
            ),
            glmmTMB = (
                script = joinpath(@__DIR__, "bym_disease_mapping.R"),
                target = :glmmTMB,
                comparability = "MLE baseline",
                comparability_notes = "Frequentist; no Bayesian comparison meaningful. Included for speed context only.",
            ),
        ),
    )
end
```

The script (`bym_disease_mapping.R`) implements all three R-side targets
in one file. CLI dispatches via `--target`. Keeps related code together;
avoids file proliferation when a single scenario has three R analogues.

## Cross-language data interchange

- **Inputs (Julia → external)**: JSON for small/structured data; Arrow for
  large dataframes (cross-language read support is good in both R and
  Python). Each scenario writes a `data.json` or `data.arrow` to a temp
  dir; the script reads from `--input` flag.
- **Outputs (external → Julia)**: JSON only. The engine script writes a
  `result.json` matching `utils/reporting.jl`'s schema (subset of fields
  the engine can fill in; the runner fills in environment metadata it
  knows).

Data is regenerated per run from a deterministic seed, not committed. This
keeps the repo small and avoids stale data divergence between
language-specific copies.

## Timing semantics for external engines

Codex's "cold vs warm" extends to subprocess timing:

- **Cold (process)** — Julia spawns a fresh `Rscript`. Includes R startup,
  `library(INLA)` load, and the fit. The user-facing reality.
- **Warm (process)** — median of `N ≥ 3` fits within the same already-warm
  R session. The runner achieves this by passing `--repetitions N` to the
  R script and letting the script time each fit internally; the script
  emits per-fit timings.
- **Inner timing (fit-only)** — just the engine's fit call (`inla(...)`)
  excluding all R startup or post-processing. Useful for like-for-like
  comparison with Latte's inner timings.

For R-INLA: cold ≈ 5–30s (R + INLA load); warm ≈ scenario fit time. Both
get reported.

## Environment management

Three levels of rigour, each appropriate at a different stage:

### Level 1 (now-ish): document + capture

Each external script begins with a `sessionInfo()` (R) or `pip freeze`
(Python) call that gets captured into the result JSON. The README
documents required versions; users install them themselves.

Lowest friction; works for "I have R installed" users; first design that
ships.

### Level 2 (when external comparisons stabilise): renv + uv lockfiles

R: `benchmark/external/r/renv.lock` pinning INLA + brms + glmmTMB.
Python: `benchmark/external/py/uv.lock` if/when PyMC is added.

Reproducible without containers. Recommended for published comparisons.

### Level 3 (only if external benchmarks become published material): Docker

Single image with everything baked in. Heaviest but bit-for-bit
reproducible. Probably overkill for our audience; skip unless asked.

## Failure handling

If R isn't installed, or `Rscript` errors, or `library(INLA)` fails, the
runner records:

```json
{
  "status": "skipped",
  "skip_reason": "external_unavailable",
  "skip_detail": "Rscript: command not found"
}
```

…and continues. Never crashes the suite because one external dependency
is missing. The CLI surfaces a count of skipped runs at the end.

## Comparability label discipline

Each external pairing must answer in the scenario file:

- Same likelihood family + parameters?
- Same prior structure?
- Same parameterisation (mean vs precision, etc.)?
- Same constraints (sum-to-zero, intercept handling, etc.)?

If the answer is "yes" to all four: `same posterior`.
If one or more diverge but the question asked is similar: `analogue`,
with notes explaining what differs.
If the engine doesn't produce a posterior at all: `MLE baseline`.

Labels live alongside results in JSON so the docs page can render them
prominently.

## Result schema additions for external

External results carry the same schema as internal, plus:

- `external_target` — engine identifier within the script (e.g. `r_inla`,
  `brms`, `glmmTMB`).
- `external_comparability` — the label.
- `external_comparability_notes` — short paragraph on what differs.
- `process_startup_seconds` — cold/warm distinction made explicit.
- `script_path` — relative path to the script that produced this run.

## What remains undesigned

This document covers the *running* of external comparisons. Things still
to design when we get there:

- Posterior summary alignment (parameter naming differs across packages —
  needs a per-scenario crosswalk).
- Accuracy metrics that work cross-package (KS distance? Wasserstein on
  marginals?).
- How to handle MCMC draws emitted by external samplers — same posterior
  summary shape as Latte's NUTS reference?
- Whether external scripts are CI-tested (probably not; CI doesn't have
  R installed and we don't want it to).

Defer until we're actually wiring R-INLA in. Don't over-engineer.
