# Landing page preview

Scratch area for iterating on the Latte.jl landing page copy + design,
*before* porting to the real Vitepress theme. None of this ships — it's
a static HTML prototype for visual review.

## Files

- `landing-designer-original.html` — the original design handoff from
  Claude Design, copy-as-shipped. Fully beautiful but contains many
  fabricated claims (`brew()`, `@latte`, Enzyme, 12× speedup,
  specific benchmark numbers, etc.). Kept for reference.
- `landing.html` — same design, honest copy. Every fabricated claim
  has been rewritten to match the actual package:
  - Replaced `brew()` / `@latte` code examples with our DPPL
    `@model` + `latte_from_dppl` + `inla`/`tmb`/`hmc_laplace` flow.
    (The `brew()` call-out in the hero/menu stays as a planned
    one-line convenience wrapper — see the "brew() unified
    entrypoint" follow-up.)
  - Engine strip now shows `:inla`, `:hmc_laplace`, `:tmb` (our real
    engines, not `:laplace` / `:mcmc` / `:vi`).
  - Benchmark receipt replaced with `PREVIEW` stamp + `pending`
    rows. Numbers will land with the benchmark suite.
  - Gallery cards map to our actual tutorials (BYM disease mapping,
    earthquake intensity trends, Kronecker space-time).
  - Footer claims trimmed (no Slack channel, no JOSS paper yet).

## Next steps

1. Review `landing.html` for remaining untruths or framing issues.
2. Decide whether to build the `brew()` convenience wrapper before
   committing to the copy (see the "brew() unified entrypoint"
   follow-up below).
3. Port the HTML to a Vitepress Vue component under
   `docs/src/.vitepress/theme/` so Documenter-built pages can share
   the same header/footer but the landing page uses the custom
   layout.
4. Build a real benchmark suite to back the "speedup" pitch; update
   the receipt with actual numbers.

## Open follow-ups referenced above

- **`brew()` unified entrypoint** — one line dispatching to the
  existing `inla`/`tmb`/`hmc_laplace`. Matches the coffee metaphor
  and gives first-time users a single API surface. Planned.
- **Benchmark suite** — reproducible benchmarks vs R-INLA, glmmTMB,
  brms on canonical models. Unlocks real numbers on the landing
  receipt + specs section.
- **Vitepress port** — convert this HTML into Vue components wired
  into `DocumenterVitepress`. Custom landing + global theme tokens.

## Opening the files

Both files are standalone, self-contained HTML — open directly in
any browser:

```bash
open docs/landing_preview/landing.html
open docs/landing_preview/landing-designer-original.html
```
