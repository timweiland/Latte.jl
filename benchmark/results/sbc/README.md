# SBC calibration evidence

Simulation-Based Calibration runs from `benchmark/sbc/sbc_matrix.jl`, ranking
both the free hyperparameters and the joint `:loglik` `DataDependentQuantity`
per cell, with **PIT** ranking (`sbc_run(...; rank_method=:auto)` — required for
grid/Laplace engines, whose posterior θ is quantized to a few integration-grid
points).

| file | regime | n_nodes | pc_u | n_attempted | engines | 95% KS band |
|------|--------|--------:|-----:|------------:|---------|------------:|
| `stress_n1000_inla_tmb.json` | weak-id stress | 30 | 1.0 | 1000 | inla, tmb | 0.043 |
| `wellidentified_n500_inla_tmb.json` | well-identified | 100 | 1.0 | 500 | inla, tmb | 0.061 |

## Verdicts

- **INLA** — calibrated when the model is well-posed (well-id: 5/6 cells with τ,
  σ, and `:loglik` inside the band). Under stress a small τ gap (~0.07): partly
  the default exploration grid (a finer grid pulls it 0.07→0.05, then plateaus),
  the rest a Laplace marginal-likelihood floor for low-count data. `:loglik`
  calibrated throughout.
- **hmc_laplace** — tracks INLA (stress τ ≈ 0.08–0.10, within band); samples the
  Laplace marginal accurately, so it avoids TMB's failure mode. (Run lean —
  reduced NUTS chain — because full-thread NUTS SBC is GC-bound.)
- **TMB** — Gaussian-at-MAP: calibrated for Gaussian-like hyperparameter
  posteriors (Normal-RW1) but miscalibrated on skewed count/binary ones
  (τ 0.12–0.31) even when well-identified. A speed/accuracy trade-off.
- **Gaussian-IID fails all engines** — structural non-identification
  (`y~N(x,σ), x~N(0,1/τ)` ⇒ only `σ²+1/τ` is identified). A model pathology, not
  an engine defect; RW1 structure breaks it and recovers.

Binomial is deferred (forward-sampling needs per-site trial counts the LGM's
obs-model descriptor doesn't carry; Bernoulli covers the binary case). Full
narrative: `tasks/validation-report.org` (RAN IT — findings).

Regenerate, e.g.:

```bash
SBC_THREADS=8 SBC_N=1000 SBC_NPOST=500 SBC_NODES=30 SBC_PCU=1.0 \
  SBC_ENGINES=inla,tmb SBC_OUT=benchmark/results/sbc/stress_n1000_inla_tmb.json \
  julia -t 8 --project=benchmark benchmark/sbc/sbc_matrix.jl
```
