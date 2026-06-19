<script setup lang="ts">
import benchmarkData from '../data/benchmark_results.json'
import BenchScatter from './BenchScatter.vue'
import BenchCard from './BenchCard.vue'
import BenchScaling from './BenchScaling.vue'

type ReceiptRow = {
  label: string
  value: string
  /** indented "vs" line under a parent row */
  muted?: boolean
}

type Receipt = {
  /** comparability label printed under the title */
  comparability: 'internal' | 'same posterior' | 'analogue' | 'mle baseline' | 'gold standard' | 'pending'
  title: string
  scenario: string
  rows: ReceiptRow[]
  /** small footer note */
  notes?: string
  /** when present, the "PREVIEW" stamp is hidden — this is a real run */
  live?: boolean
}

// Receipts produced by the actual benchmark runners — each entry comes
// from `benchmark/render_docs.jl` reading the per-scenario `result.json`.
const liveExternal: Receipt[] = (benchmarkData.receipts as any[]).map(r => ({
  comparability: r.comparability as Receipt['comparability'],
  title: r.title,
  scenario: r.scenario,
  rows: r.rows as ReceiptRow[],
  notes: r.notes,
  live: true,
}))

const external: Receipt[] = [...liveExternal]

const comparabilityLabel: Record<Receipt['comparability'], string> = {
  internal: 'INTERNAL',
  'same posterior': 'SAME POSTERIOR',
  analogue: 'ANALOGUE ONLY',
  'mle baseline': 'MLE BASELINE',
  'gold standard': 'GOLD STANDARD',
  pending: 'PENDING',
}
</script>

<template>
  <main class="bench-page">
    <div class="container">
      <header class="bench-hero">
        <div class="bench-eyebrow">BENCHMARKS</div>
        <h1>Benchmarks.</h1>
        <p class="bench-lede">
          How Latte's INLA fits compare on the same models, against an
          identical likelihood and prior — so the only difference is the
          approximation. Accuracy is the <strong>KS</strong> distance between
          the two engines' marginals (0 = identical, reported max / median per
          block); speed is warm-fit wall-clock (cold includes Julia's first-run
          compilation). Every figure links to a runnable script in
          <code>benchmark/</code>, versions and hardware recorded.
        </p>
      </header>

      <section class="bench-section">
        <header class="bench-section-head">
          <h2>Comparison against R-INLA</h2>
        </header>
        <BenchScatter />
        <BenchCard />
      </section>

      <section class="bench-section">
        <header class="bench-section-head">
          <h2>Scaling in <em>n</em></h2>
          <p>
            The same spatial Poisson–Matérn (SPDE) model fit at five mesh
            resolutions, on a mesh handed identically to both engines — so the
            curves isolate how each scales with the latent dimension, not the
            problem setup.
          </p>
        </header>
        <BenchScaling />
      </section>

      <section class="bench-section">
        <header class="bench-section-head">
          <h2>Coming next</h2>
          <p>
            Cross-engine comparisons — the same model under INLA, TMB, and
            HMC-Laplace — plus scaling in hyperparameter dimension. A dedicated
            TMB comparison (Latte's <code>tmb()</code> against the TMB R package
            on matched Laplace marginal-likelihood models) will follow in a later
            release. Scripts land in <code>benchmark/</code> as they're run.
          </p>
        </header>
      </section>

      <section class="bench-section">
        <header class="bench-section-head">
          <h2>Reproducibility</h2>
          <p>Run the benchmarks yourself; tell us when something looks off.</p>
        </header>
        <div class="repro-grid">
          <div class="repro-card">
            <div class="repro-tag">SCRIPTS</div>
            <p>All benchmark scripts live in <code>benchmark/</code> in the repo. Each receipt above maps to a single file.</p>
          </div>
          <div class="repro-card">
            <div class="repro-tag">CADENCE</div>
            <p>Benchmarks are rerun for each Latte minor release and at least every six months for external package versions.</p>
          </div>
          <div class="repro-card">
            <div class="repro-tag">CORRECTIONS</div>
            <p>Think a comparison is unfair or outdated? <a href="https://github.com/timweiland/Latte.jl/issues">Open an issue</a> with a reproducible script. We'd rather know.</p>
          </div>
        </div>
      </section>
    </div>
  </main>
</template>

<style scoped>
.bench-page {
  --bg:       #FAF7F2;
  --cream:    #F5E6D3;
  --tan:      #E8D5B7;
  --caramel:  #C9986A;
  --mocha:    #8B6F47;
  --bean:     #3D2817;
  --espresso: #2A1810;
  --berry:    #C04A2A;
  --foam:     #FFFCF7;
  background: var(--bg);
  color: var(--espresso);
  font-family: 'Inter', system-ui, sans-serif;
  padding: 56px 0 96px;
  min-height: 60vh;
}
.container { max-width: 1200px; margin: 0 auto; padding: 0 48px; }

/* ── Hero ── */
.bench-hero { max-width: 720px; margin-bottom: 64px; }
.bench-eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px; letter-spacing: 1.5px;
  color: var(--berry);
  margin-bottom: 16px;
}
.bench-hero h1 {
  font-family: 'Fraunces', Georgia, serif;
  font-weight: 400;
  font-size: clamp(40px, 5vw, 64px);
  line-height: 1;
  letter-spacing: -0.035em;
  margin: 0 0 20px;
}
.bench-hero h1 em { font-style: italic; color: var(--bean); font-weight: 400; }
.bench-lede { font-size: 18px; line-height: 1.55; color: #4A3828; max-width: 600px; margin: 0; }

/* ── How to read ── */
.how-to-read { margin-bottom: 72px; }
.how-to-read h2 {
  font-family: 'Fraunces', Georgia, serif;
  font-style: italic;
  font-weight: 400;
  font-size: 26px;
  margin: 0 0 24px;
  color: var(--espresso);
}
.how-grid {
  display: grid; grid-template-columns: repeat(2, 1fr);
  gap: 22px 32px;
}
.how-tag {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; letter-spacing: 1.3px;
  color: var(--caramel);
  margin-bottom: 6px;
}
.how-grid p { margin: 0; font-size: 14.5px; line-height: 1.55; color: #4A3828; }
.how-grid code { background: var(--tan); color: var(--bean); padding: 1px 6px; border-radius: 3px; font-family: 'JetBrains Mono', monospace; font-size: 0.92em; }

/* ── Section ── */
.bench-section { margin-bottom: 64px; }
.bench-section-head { margin-bottom: 28px; max-width: 720px; }
.bench-section-head h2 {
  font-family: 'Fraunces', Georgia, serif;
  font-style: italic;
  font-weight: 400;
  font-size: 36px;
  letter-spacing: -0.02em;
  line-height: 1.1;
  margin: 0 0 12px;
  color: var(--espresso);
}
.bench-section-head h2 em { font-style: italic; color: var(--berry); }
.bench-section-head p { margin: 0; font-size: 15px; line-height: 1.55; color: #4A3828; }
.bench-section-head code { background: var(--tan); color: var(--bean); padding: 1px 6px; border-radius: 3px; font-family: 'JetBrains Mono', monospace; font-size: 0.92em; }

/* ── Receipt grid ── */
.receipt-grid {
  display: grid; grid-template-columns: repeat(3, 1fr);
  gap: 24px;
}
.receipt {
  background: var(--foam);
  padding: 28px 26px 22px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px; line-height: 1.7;
  color: var(--espresso);
  box-shadow: 0 12px 32px rgba(42,24,16,0.08);
  position: relative;
}
.receipt .stamp {
  position: absolute; top: 18px; right: 14px;
  transform: rotate(-12deg);
  border: 2px solid var(--berry);
  color: var(--berry);
  padding: 3px 8px; border-radius: 4px;
  font-family: 'Fraunces', serif;
  font-style: italic; font-size: 11px;
  font-weight: 600; letter-spacing: 0.5px;
  opacity: 0.85;
}
.receipt .stamp.live {
  border-color: var(--mocha);
  color: var(--mocha);
}
.receipt .head { text-align: center; margin-bottom: 14px; }
.receipt .head .name {
  font-family: 'Fraunces', Georgia, serif;
  font-style: italic;
  font-size: 19px;
  font-weight: 500;
  letter-spacing: -0.3px;
  line-height: 1.15;
}
.receipt .head .sub {
  font-size: 10px;
  color: var(--mocha);
  margin-top: 4px;
  letter-spacing: 0.4px;
}
.receipt .comparability-label {
  font-size: 9.5px;
  letter-spacing: 1.2px;
  color: var(--caramel);
  margin-top: 8px;
}
.receipt hr { border: none; border-top: 1px dashed var(--mocha); margin: 10px 0; }
.receipt .row { display: flex; justify-content: space-between; }
.receipt .row.muted { color: var(--mocha); padding-left: 12px; font-size: 11px; }
.receipt .notes { font-size: 10.5px; color: var(--mocha); text-align: center; line-height: 1.5; }

/* ── Workflow placeholder ── */
.placeholder-card {
  background: var(--cream);
  border: 1px dashed var(--mocha);
  padding: 28px 26px;
}
.placeholder-title {
  font-family: 'Fraunces', serif;
  font-style: italic;
  font-size: 18px;
  color: var(--bean);
  margin-bottom: 6px;
}
.placeholder-body {
  font-size: 14px; line-height: 1.55; color: #4A3828;
}

/* ── Loses ── */
.loses-list {
  list-style: none; padding: 0; margin: 0;
  display: grid; grid-template-columns: 1fr; gap: 14px;
  max-width: 800px;
}
.loses-list li {
  background: var(--foam);
  border-left: 3px solid var(--berry);
  padding: 14px 18px;
  font-size: 14.5px; line-height: 1.55;
  color: #4A3828;
}
.loses-list li strong { color: var(--espresso); font-weight: 600; }
.loses-list code { background: var(--tan); color: var(--bean); padding: 1px 6px; border-radius: 3px; font-family: 'JetBrains Mono', monospace; font-size: 0.92em; }

/* ── Reproducibility ── */
.repro-grid {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px;
}
.repro-card {
  background: var(--foam);
  border: 1px solid var(--tan);
  padding: 20px 18px;
}
.repro-tag {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; letter-spacing: 1.3px;
  color: var(--caramel);
  margin-bottom: 8px;
}
.repro-card p { margin: 0; font-size: 14px; line-height: 1.55; color: #4A3828; }
.repro-card code { background: var(--tan); color: var(--bean); padding: 1px 6px; border-radius: 3px; font-family: 'JetBrains Mono', monospace; font-size: 0.9em; }
.repro-card a { color: var(--berry); }

/* Responsive */
@media (max-width: 1024px) {
  .receipt-grid { grid-template-columns: repeat(2, 1fr); }
  .how-grid { grid-template-columns: 1fr; }
  .repro-grid { grid-template-columns: 1fr; }
}
@media (max-width: 640px) {
  .receipt-grid { grid-template-columns: 1fr; }
}
</style>

<!--
  Dark mode. NOT scoped (scoped `:global(.dark)` rules are dropped by the
  CSS build), but every rule is nested under `.bench-page` so it cannot
  leak. Redefining the palette custom properties flips every `var(--x)`
  consumer at once; targeted rules below handle hardcoded colors.
-->
<style>
html.dark .bench-page {
  --bg:       #2A1810; /* page background */
  --cream:    #38241B; /* elevated surface (placeholder-card) */
  --tan:      rgba(201, 152, 106, 0.18); /* code bg / subtle border */
  --caramel:  #C9986A; /* accent — kept */
  --mocha:    #B79877; /* muted text / dashed borders */
  --bean:     #D4B896; /* secondary text (code text, em) */
  --espresso: #F5E6D3; /* primary text */
  --berry:    #D9603F; /* red accent */
  --foam:     #38241B; /* card surface */
}

/* Hardcoded #4A3828 dark text → light secondary */
html.dark .bench-page .bench-lede,
html.dark .bench-page .how-grid p,
html.dark .bench-page .bench-section-head p,
html.dark .bench-page .placeholder-body,
html.dark .bench-page .loses-list li,
html.dark .bench-page .repro-card p {
  color: #D4B896;
}

/* Card shadow: warm brown → neutral dark */
html.dark .bench-page .receipt {
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.4);
}
</style>
