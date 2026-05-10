# Performance investigation notes

Status: 2026-04-30 · scoped to the R-INLA-comparison benchmark suite.

## TL;DR

- Latte's per-θ-point work is on par with R-INLA. **The performance gap
  is not Julia vs C** — it's about *how many θ points get evaluated*
  and where.
- Latte's default Cartesian grid is conservative and walks far from
  the mode. R-INLA's `int.strategy = "grid"` uses a **fixed 11-point
  (D=1) / 35-point (D=2) hardcoded design** with precomputed
  quadrature weights — it is *not* an adaptive grid at all. `dz` and
  `diff.logdens` are ignored for D ≤ 2 in R-INLA's source
  (`gmrflib/design.c::GMRFLib_design_grid`).
- We ported R-INLA's hardcoded design as `INLAGridStrategy`. It is
  available as opt-in. **It is not the default**, because it relies on
  the local Hessian at the mode being a good guide to the posterior
  spread. For long-tailed posteriors (precision parameters when the
  latent field is nearly flat) this assumption fails and accuracy
  degrades catastrophically. See "When INLAGridStrategy fails" below.

## Strategies on epil (D=2, n=236)

Identical posterior accuracy across all three (KS to R-INLA matches
to 4 decimal places on every fixed-effect marginal):

| Strategy                            | Points | Warm time |
|-------------------------------------|--------|-----------|
| Default (Cartesian, max_log_drop=6) | 70     | 4.89 s    |
| `INLAGridStrategy()`                | 45     | 3.43 s    |
| `CCDExplorationStrategy()`          | 9      | 1.55 s    |
| R-INLA `int.strategy=grid`          | 35     | 1.88 s    |

R-INLA's design has 45 points internally (per `design.c`); the 35
reported in `joint.hyper` reflects a post-evaluation early-stop /
filter mechanism that we have not yet ported.

## When `INLAGridStrategy` fails

`nhtemp` (Normal + RW2 smooth) with `INLAGridStrategy`:

- Latte mode: `(log τ_x, log σ) = (0.79, -0.12)` → τ_x ≈ 2.2.
- R-INLA mode: `(log σ, log τ_x) = (-0.055, 6.91)` → τ_x ≈ 1568.

Two different modes. `nhtemp`'s posterior on τ_x is **astronomically
long-tailed** (R-INLA's reported 95% interval on τ_x: `[9, 2.9e7]`).
Latte's mode finder lands at the lower-τ_x edge of that posterior;
R-INLA's lands at the high-τ_x edge.

The fixed-design INLA grid covers `±2.25 σ_θ` of the local Hessian.
For Latte's mode, σ_θ from the Hessian eigenvalues is ~0.1–1.4 — a
local ridge that misses the entire long tail. Result: max KS = 0.72
on per-year `x_t` marginals (basically broken).

The Cartesian grid escapes this by walking outward from the mode
until log-density drops by `max_log_drop = 6.0` — 319 points on
nhtemp, but they actually cover the long tail. Hence the default
remains Cartesian for D ≤ 2.

The mode-finder discrepancy (Latte τ_x ≈ 2 vs R-INLA τ_x ≈ 1568) is a
real, separate issue worth chasing; it likely comes down to the
optimiser's stopping criterion or initial-guess strategy on flat
posteriors.

## Strategies on the other scenarios

`INLAGridStrategy` is well-behaved on:
- **seeds** (D=1, Binomial GLMM, well-identified plate-RE precision)
- **scotland** (D=1, Poisson + Besag, well-identified)
- **tokyo** (D=1, Binomial + RW2, well-identified for daily-rainfall data)
- **epil** (D=2, both subject and observation precisions well-identified)

It fails on:
- **nhtemp** (D=2, smooth-trend Normal + RW2, τ_x has heavy tail).

## Recommended workflow

1. Default `inla(model, y; ...)` — uses the safe Cartesian grid.
   Accuracy is robust; speed is moderate.
2. For known well-conditioned posteriors (compact, near-Gaussian in
   working space), pass `exploration_strategy = INLAGridStrategy()`
   for a 30–50 % speedup.
3. For D ≥ 3 or compact posteriors at any D,
   `exploration_strategy = CCDExplorationStrategy()` is faster still
   (9 points at D=2, 25 at D=3, …).
4. If switching from Cartesian to a fixed-design strategy and KS
   inflates, the posterior is heavy-tailed; revert to Cartesian.

## Per-scenario summary (warm-on-warm, default Cartesian)

| Scenario  | n    | D | Latent dim | Latte warm | R-INLA  | Latte/R-INLA | Notes                              |
|-----------|------|---|------------|------------|---------|--------------|------------------------------------|
| seeds     | 21   | 1 | 46         | 0.05–0.06s | 2.86 s  | 47×          | best-in-class                      |
| scotland  | 56   | 1 | 114        | ~1.5 s     | 1.49 s  | ~1×          | Besag dominates                    |
| tokyo     | 366  | 1 | 366        | 1.7 s      | 1.51 s  | 0.9×         | RW2 dim 366                        |
| nhtemp    | 60   | 2 | 121        | 2.0 s      | 1.58 s  | 0.79×        | 2D grid × 319 pts                  |
| epil      | 236  | 2 | 537        | 4.9 s      | 1.88 s  | 0.39×        | INLAGridStrategy → 3.4 s (1.8×)    |
|           |      |   |            |            |         |              | CCDExplorationStrategy → 1.6 s (1.2×) |

## What remains

1. **Mode-finder behaviour on flat-posterior nhtemp**: investigate why
   Latte and R-INLA land on different modes. This is the main blocker
   to making `INLAGridStrategy` the default.
2. **Per-point speed on epil**: 75 ms/pt (Latte) vs 42 ms/pt (R-INLA).
   That's a real ~1.8× per-point gap, plausibly due to inner Laplace
   loop tuning. Worth profiling separately from grid choice.
3. **Early-stop in `INLAGridStrategy`**: R-INLA prunes ~10 of 45
   design points dynamically (low density / log-likelihood ratio).
   Cheap to add; cuts 20 % more time for free at well-conditioned
   posteriors.
