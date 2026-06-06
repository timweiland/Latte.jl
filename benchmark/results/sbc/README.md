# SBC calibration evidence

Simulation-Based Calibration runs from `benchmark/sbc/sbc_matrix.jl`, ranking
both the free hyperparameters and the joint `:loglik` `DataDependentQuantity`
per cell, with **PIT** ranking (`sbc_run(...; rank_method=:auto)` — required for
grid/Laplace engines, whose posterior θ is quantized to a few integration-grid
points).

**Verdict** (in `benchmark/render_validation.jl`): the Säilynoja, Bürkner &
Vehtari (2022) ECDF test with 95% *simultaneous* confidence bands defines
*within band*. Because at n≈10³ that test detects even tiny error, cells failing
it are tiered by KS effect size — *minor* (≤0.10 max-CDF deviation,
approximation-level) vs *substantial* (>0.10) — so an approximate-but-usable
engine isn't conflated with a genuinely-off one. The harness stores a 100-bin
rank histogram per cell to drive this offline.

**Sanity-checked**: on the Gaussian-IID model an SBC run against the *exact*
posterior is uniform (τ KS 0.059, σ KS 0.041 at the 0.068 band), confirming the
reference + harness + PIT machinery are correct independently of any engine.

| file | regime | n_nodes | pc_u | n_attempted | engines | 95% KS band |
|------|--------|--------:|-----:|------------:|---------|------------:|
| `stress_n1000_inla_tmb.json` | weak-id stress | 30 | 1.0 | 1000 | inla, tmb | 0.043 |
| `wellidentified_n500_inla_tmb.json` | well-identified | 100 | 1.0 | 500 | inla, tmb | 0.061 |
| `stress_hmc.json` | weak-id stress | 30 | 1.0 | 250 | hmc_laplace | 0.086 |

`benchmark/render_validation.jl` merges these by (engine, regime) into
`docs/src/data/validation_results.json`, rendered as **engine tabs** on the docs
Validation page. hmc_laplace is shown for the stress regime only — the
well-identified regime (n=100 nodes) is prohibitively slow for per-replicate NUTS.

## Verdicts

- **INLA** — calibrated when the model is well-posed (well-id: 5/6 cells with τ,
  σ, and `:loglik` inside the band). Under stress a small τ gap (~0.07): partly
  the default exploration grid (a finer grid pulls it 0.07→0.05, then plateaus),
  the rest a Laplace marginal-likelihood floor for low-count data. `:loglik`
  calibrated throughout.
- **hmc_laplace** — passes all 14 stress cells (band 0.086); samples the Laplace
  marginal accurately, so it avoids TMB's failure mode and even handles the
  non-identified Gaussian-IID better than INLA (see below). (Run lean — reduced
  NUTS chain — because full-thread NUTS SBC is GC-bound.)
- **TMB** — Gaussian-at-MAP: calibrated for Gaussian-like hyperparameter
  posteriors (Normal-RW1) but miscalibrated on skewed count/binary ones
  (τ 0.12–0.31) even when well-identified. A speed/accuracy trade-off.
- **Gaussian-IID** — structural non-identification (`y~N(x,σ), x~N(0,1/τ)` ⇒ only
  `σ²+1/τ` is identified). INLA/TMB miscalibrate on the ridge (KS 0.27–0.40);
  hmc_laplace's NUTS samples it far more faithfully (KS ~0.06–0.085, though at a
  looser band). RW1 structure breaks the degeneracy and all engines recover.

Binomial is deferred (forward-sampling needs per-site trial counts the LGM's
obs-model descriptor doesn't carry; Bernoulli covers the binary case). Full
narrative: `tasks/validation-report.org` (RAN IT — findings).

Regenerate, e.g.:

```bash
SBC_THREADS=8 SBC_N=1000 SBC_NPOST=500 SBC_NODES=30 SBC_PCU=1.0 \
  SBC_ENGINES=inla,tmb SBC_OUT=benchmark/results/sbc/stress_n1000_inla_tmb.json \
  julia -t 8 --project=benchmark benchmark/sbc/sbc_matrix.jl
```
