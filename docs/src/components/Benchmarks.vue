<script setup lang="ts">
import benchmarkData from '../data/benchmark_results.json'

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

const internal: Receipt[] = [
  {
    comparability: 'internal',
    title: 'BYM disease mapping',
    scenario: 'Spatial Poisson · Besag + IID',
    rows: [
      { label: 'INLA', value: 'pending' },
      { label: 'TMB', value: 'pending' },
      { label: 'HMC-Laplace', value: 'pending' },
      { label: 'NUTS (reference)', value: 'pending' },
    ],
    notes: 'agreement vs NUTS · pending',
  },
  {
    comparability: 'internal',
    title: 'AR1 Poisson',
    scenario: 'Temporal counts · RW1 latent',
    rows: [
      { label: 'INLA', value: 'pending' },
      { label: 'TMB', value: 'pending' },
      { label: 'HMC-Laplace', value: 'pending' },
      { label: 'NUTS (reference)', value: 'pending' },
    ],
    notes: 'agreement vs NUTS · pending',
  },
  {
    comparability: 'internal',
    title: 'Separable space-time',
    scenario: 'Kronecker prior · Poisson likelihood',
    rows: [
      { label: 'INLA', value: 'pending' },
      { label: 'TMB', value: 'pending' },
      { label: 'HMC-Laplace', value: 'pending' },
    ],
    notes: 'NUTS infeasible at this scale · pending sims',
  },
]

const external: Receipt[] = [
  ...liveExternal,
  {
    comparability: 'pending',
    title: 'Matérn SPDE on a mesh',
    scenario: 'Latte INLA vs R-INLA SPDE',
    rows: [
      { label: 'Latte INLA', value: 'pending' },
      { label: 'R-INLA SPDE', value: 'pending', muted: true },
    ],
    notes: 'R-INLA SPDE has 15 years of optimization Latte hasn\'t matched yet',
  },
  {
    comparability: 'analogue',
    title: 'Hierarchical Poisson GLMM',
    scenario: 'Latte INLA vs brms / glmmTMB',
    rows: [
      { label: 'Latte INLA', value: 'pending' },
      { label: 'brms (NUTS)', value: 'pending', muted: true },
      { label: 'glmmTMB (MLE)', value: 'pending', muted: true },
    ],
    notes: 'glmmTMB is frequentist · MLE baseline only',
  },
]

const scaling: Receipt[] = [
  {
    comparability: 'internal',
    title: 'Spatial GLMM · scaling in n',
    scenario: 'INLA / TMB / HMC-Laplace, n ∈ {100, 1k, 10k}',
    rows: [
      { label: 'n = 100', value: 'pending' },
      { label: 'n = 1 000', value: 'pending' },
      { label: 'n = 10 000', value: 'pending' },
    ],
    notes: 'cold + warm reported separately',
  },
  {
    comparability: 'internal',
    title: 'Hyperparameter dim',
    scenario: 'INLA grid cost as |θ| grows',
    rows: [
      { label: '|θ| = 2', value: 'pending' },
      { label: '|θ| = 4', value: 'pending' },
      { label: '|θ| = 6 (CCD)', value: 'pending' },
    ],
    notes: 'guidance: prefer TMB / HMC-Laplace above |θ| ~ 5',
  },
]

const sections: { title: string; lede: string; receipts: Receipt[] }[] = [
  {
    title: 'Internal cross-engine',
    lede: 'The same DPPL @model runs under INLA, TMB, and HMC-Laplace. Posterior agreement and wall-clock cost for each, on representative scenarios.',
    receipts: internal,
  },
  {
    title: 'Reference comparisons',
    lede: 'External packages on the same scenarios. Each receipt is labelled for how comparable the targets are: identical posterior, analogue, or MLE baseline.',
    receipts: external,
  },
  {
    title: 'Scaling regimes',
    lede: 'Fit time across data size, latent dimension, and hyperparameter count. Helps decide when to switch engines.',
    receipts: scaling,
  },
]

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
        <h1>Choosing an<br/><em>inference path.</em></h1>
        <p class="bench-lede">
          Wall-clock cost and posterior agreement for each inference
          strategy, on representative latent Gaussian models. Use the
          numbers to pick a workflow.
        </p>
      </header>

      <section class="how-to-read">
        <h2>How to read the receipts</h2>
        <div class="how-grid">
          <div>
            <div class="how-tag">COLD vs WARM</div>
            <p>"Cold" includes Julia's first-run compilation. "Warm" is the second-run onward. Both are reported.</p>
          </div>
          <div>
            <div class="how-tag">AGREEMENT</div>
            <p>For posterior comparisons: median absolute difference in posterior means and SDs against the reference (long NUTS where feasible).</p>
          </div>
          <div>
            <div class="how-tag">COMPARABILITY</div>
            <p>Each external comparison carries a label: <code>same posterior</code> means identical likelihood + prior; <code>analogue</code> means similar but not identical; <code>MLE baseline</code> means the comparison is frequentist.</p>
          </div>
          <div>
            <div class="how-tag">REPRODUCIBILITY</div>
            <p>Every receipt links to its script in the <code>benchmark/</code> directory. Versions, hardware, and run policy are recorded with each result.</p>
          </div>
        </div>
      </section>

      <section v-for="sec in sections" :key="sec.title" class="bench-section">
        <header class="bench-section-head">
          <h2>{{ sec.title }}</h2>
          <p>{{ sec.lede }}</p>
        </header>
        <div class="receipt-grid">
          <article v-for="r in sec.receipts" :key="r.title" class="receipt">
            <div class="stamp" v-if="!r.live">PREVIEW</div>
            <div class="stamp live" v-else>LIVE</div>
            <div class="head">
              <div class="name">{{ r.title }}</div>
              <div class="sub">{{ r.scenario }}</div>
              <div class="comparability-label">{{ comparabilityLabel[r.comparability] }}</div>
            </div>
            <hr/>
            <div v-for="row in r.rows" :key="row.label"
                 class="row" :class="{ muted: row.muted }">
              <span>{{ row.label }}</span><span>{{ row.value }}</span>
            </div>
            <hr v-if="r.notes"/>
            <div v-if="r.notes" class="notes">{{ r.notes }}</div>
          </article>
        </div>
      </section>

      <section class="bench-section">
        <header class="bench-section-head">
          <h2>Workflow cost</h2>
          <p>
            Lines of model code, post-processing burden, and number of
            packages required to express a model. Published alongside
            wall-clock time so the same-<code>@model</code>-three-engines
            claim is measurable.
          </p>
        </header>
        <div class="placeholder-card">
          <div class="placeholder-title">Coming with v0.1</div>
          <div class="placeholder-body">
            Receipts comparing model definition + post-processing line counts
            across packages on identical scenarios. Forthcoming alongside
            the wall-clock benchmarks above.
          </div>
        </div>
      </section>

      <section class="bench-section">
        <header class="bench-section-head">
          <h2>Where Latte loses</h2>
          <p>Things Latte does worse than the alternatives, roughly ranked.</p>
        </header>
        <ul class="loses-list">
          <li><strong>Cold-start latency.</strong> Julia's first-run compilation costs hundreds of ms to seconds. R-INLA pays this once at package load and we pay it per-fit.</li>
          <li><strong>Matérn SPDE maturity.</strong> R-INLA's SPDE pipeline has 15 years of optimization. Latte's SPDE story (via GaussianMarkovRandomFields.jl) is sound but younger.</li>
          <li><strong>Joint multi-likelihood models.</strong> R-INLA supports multi-response joint models out of the box; Latte doesn't yet.</li>
          <li><strong>Hyperparameter scaling.</strong> Latte's INLA grid exploration becomes expensive above |θ| ≈ 5. Use <code>:hmc_laplace</code> there.</li>
          <li><strong>Documentation depth.</strong> Pre-release. Tutorials cover the major model classes; reference depth is still building.</li>
        </ul>
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
