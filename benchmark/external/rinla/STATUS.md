# R-INLA Cross-Validation Status

## Tokyo Rainfall (RW2 + binomial logit)

Files: `tokyo/tokyo_compare.{R,jl}`

**Numbers** (n=366, both with `simplified.laplace`, R-INLA `int.strategy=grid`):
- x_t KS: max 0.069 (day 177), median 0.039, 88/366 > 0.05
- Day 342 (worst case earlier): Latte mean −0.580, R-INLA −0.602; sd 0.224 vs 0.229
- τ posterior: matches R-INLA almost exactly
- Latte 11.6 s vs R-INLA 1.5 s

**Root cause of the original 0.18 KS** (now fixed):
Latte's `RW2Model` (in GaussianMarkovRandomFields.jl) imposes the full
polynomial null-space constraint set: sum-to-zero AND linear-trend-zero.
R-INLA's default `rw2` only imposes sum-to-zero; the linear (slope)
direction is left improper and identified by data. With Tokyo's seasonal
pattern, the data wants a slight non-zero slope; Latte's hard slope=0
constraint redistributes mass and tightens marginals, producing the
~0.085 mean shift and ~17% sd reduction seen at extreme days.

The benchmark uses a small `RW2SumOnly` wrapper (in `tokyo_compare.jl`)
that overrides `constraints()` to return only the sum-to-zero row.
This is the right fix for benchmarking against R-INLA. Whether to make
this the upstream default in `RWModel` is a separate design question
for `GaussianMarkovRandomFields.jl` — both choices are mathematically
valid, just different model specs.

**Investigation history** (for the curious):
- Mode finding lands at the correct basin (τ ≈ 18K) after commit 079fe63.
- Unnormalized log π(τ|y) shape AGREES between engines (constant offset
  −414.49 ± 0.02 across τ from 1K to 2M when read at exact R-INLA grid
  points; an earlier "matching" finding via linear interpolation was
  noise).
- Switching latent strategy (Gaussian / SLA / Laplace) doesn't change
  KS — confirms the issue isn't the latent marginal.
- `augment=false` doesn't change KS — not augmentation.
- Concentrating the τ-grid (max_log_drop=1) doesn't change KS — not
  tail integration.
- The fix above (R-INLA-matching constraints) cuts max KS from 0.18
  to 0.07.

## Toy IID Poisson (n=50, IID + Poisson)

Files: `iid_poisson/iid_poisson_compare.{R,jl}`

**Numbers**:
- x_i KS: max 0.053 (i=28), median 0.047, 15/50 > 0.05
- τ posterior: mean 1.83 vs 1.84, std 0.62 vs 0.63, median 1.73 vs 1.74,
  q975 3.44 vs 3.93 (Latte 12% tighter right tail)
- Latte 8.7 s vs R-INLA 1.3 s

This scenario has no smoothing prior (no constraints), so the bug above
doesn't show up. Used as the simpler "control" benchmark.

## Open

The Tokyo fix changes the `RW2SumOnly` wrapper inside the benchmark file.
For a permanent solution, consider adding a `constraint_orders` keyword
to `RWModel` upstream (`:full` for current behavior, `:sum_only` for
R-INLA-compatible). That's an opinionated change and worth a discussion.
