# R-INLA cross-validation benchmarks

This directory holds Latte ↔ R-INLA comparison runners for stock R-INLA
example datasets. Each sub-directory pairs:

- `<scenario>_compare.jl` — fits Latte INLA, optionally calls R-INLA via
  `Rscript`, computes per-marginal KS distances, dumps a single
  `_workdir/result.json` with all the numbers needed for the docs.
- `<scenario>_compare.R` — R-INLA fitter that emits per-marginal CSVs the
  Julia runner reads back.
- Embedded data CSV(s).

## Run a single comparison

```bash
cd benchmark/external/rinla/<scenario>
julia --project=../../.. <scenario>_compare.jl
```

First-run is cold (~10 – 15 s of JIT for Latte). Each runner does a
warmup call plus a 3-or-5-rep median for "warm" timing. R-INLA is cached
under `_workdir/` after the first call; pass `--refresh-rinla` to re-fit.

Output: `_workdir/result.json` with shape

```json
{
  "scenario": "seeds",
  "n": 21,
  "ks_fixed": [...], "sgn_fixed": [...],
  "ks_b": [...],
  "t_latte_cold": 9.4,
  "t_latte_warm": 0.061,
  "t_rinla": 2.86,
  "inla_version": "25.6.7"
}
```

## Run all comparisons

There's no master script — just iterate the sub-directories. Each one is
independent.

## Render to docs

After the runners have all left their `result.json` files in place:

```bash
julia --project=benchmark benchmark/render_docs.jl
```

That writes `docs/src/data/benchmark_results.json`, which
`docs/src/components/Benchmarks.vue` imports at build time. Then a
normal `make docs` (or `make docs-skip` to skip tutorials) bakes the
numbers into the static site.

## Adding a new scenario

1. `mkdir benchmark/external/rinla/<name>` and drop the data + `_compare.R`.
2. Copy an existing `_compare.jl` (seeds is the simplest) and adapt the
   model, normalisation, and KS reporting.
3. Add a formatter in `benchmark/render_docs.jl` (`_format_<name>`) and
   register it in the `FORMATTERS` dict. Append the scenario id to
   `RECEIPT_ORDER`.
4. Optionally add a scenario file under `benchmark/scenarios/` if the
   model should be runnable through the suite harness; register in
   `SCENARIO_FILES` in `benchmark/runbench.jl`.

## Performance notes

Most of the warm-time gap between Latte and R-INLA is grid-point count, not
per-point cost.
