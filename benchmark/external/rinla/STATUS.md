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

## SPDEtoy (Matérn SPDE + Gaussian, shared mesh)

Files: `spdetoy/spdetoy_compare.{R,jl}`

The SPDE cross-validation. R-INLA builds the mesh (`inla.mesh.2d`,
`max.edge=c(0.05,0.2)`, `cutoff=0.01`, `offset=c(0.1,0.4)` → 1680 nodes,
3305 triangles) and dumps `nodes.csv`/`triangles.csv`; the Julia side
rebuilds the IDENTICAL Ferrite grid + FEM discretization, so both engines
solve the same discretized problem (isolates inference from meshing).
Gaussian response `y ~ N(β + A·field, σ)`, matched PC priors.

**Accuracy (excellent — validates Latte's SPDE):**
- field node KS: max 0.028, median 0.018, 0/1680 > 0.05
- intercept β KS 0.028, obs SD KS 0.049, range KS 0.055, field Stdev KS 0.053
- field-mean max-diff 0.059 over 1680 nodes
- posteriors: range 0.40 (R 0.404), σ_field 1.89 (R 1.93), β 9.44 (R 9.49)

On the same mesh Latte's SPDE posterior is essentially identical to R-INLA.

**Speed — RESOLVED. Warm 1.52 s, now 1.5× FASTER than R-INLA (2.36 s).**

Originally Latte warm 18.9 s (~8× slower). Profiling (`--profile`) + micro-benchmarks
(`selinv_experiment.jl`, `cached_matern_experiment.jl`) traced it to two redundant costs
in the GMRFs FEM extension, both since fixed upstream:
- `compute_selinv!` materialised the `SupernodalMatrix` selected-inverse via the generic
  element-wise `getindex` instead of SelectedInversion's vectorised `sparse()` — ~90% of
  the fit, ~47× recoverable, bit-identical. → GMRFs #144, fixed by #147 (lazy
  materialisation). The dominant lever.
- `matern_precision_only` re-assembled the κ-independent C, G on every call (~9× on
  `precision_matrix`, but only ~2.5% of the fit). → GMRFs #145, fixed by #146.

After updating the dev'd GMRFs (#146 + #147): warm 18.9 s → **1.52 s** (~12×), cold
25.4 s → 8.7 s, KS unchanged (field median 0.018). Latte's SPDE is now slightly faster
than R-INLA on the same mesh.

Ruled out earlier (kept for the record): latent augmentation (`augment=false` ≈ 13%),
grid density (`AutoExplorationStrategy` already uses CCD for D≥3), and the DIC/WAIC/CPO
accumulators (negligible — they ride on the selinv that's computed regardless).

**Smoothness gotcha:** GMRFs `smoothness_to_ν(s, D=2) = s + 1`, so
`MaternModel(disc; smoothness=0)` ⇒ ν=1 ⇒ alpha=2, matching R-INLA's
`inla.spde2.pcmatern(alpha=2)`. The SPDE tutorial's `smoothness=1` is alpha=3.
Using smoothness=1 here gave range KS 0.71; smoothness=0 dropped it to 0.055.

**Caveats:**
- R-INLA's `pcmatern` places a JOINT PC prior on (range, σ); Latte uses
  independent `PCPrior.Precision(τ)` + `PCPrior.Range` (prototyped in Latte for
  this work, `src/distributions/pc_prior/range.jl`; in 2D the SPDE PC range prior
  is exactly `1/range ~ Exponential(λ)`, `λ = -ρ0·log(p)`). Marginals are matched
  (P(σ_field>1)=0.5, P(range<0.3)=0.5); the joint coupling is not. Minor.
- `benchmark/Project.toml` gained Ferrite/FerriteGmsh/Gmsh/LibGEOS (FEM ext).
- Not wired into `render_docs.jl` / the Benchmarks page: the speed result has
  Latte losing, so publication is a separate call once the per-θ cost improves.

## Open

The Tokyo fix changes the `RW2SumOnly` wrapper inside the benchmark file.
For a permanent solution, consider adding a `constraint_orders` keyword
to `RWModel` upstream (`:full` for current behavior, `:sum_only` for
R-INLA-compatible). That's an opinionated change and worth a discussion.
