<script setup lang="ts">
import validationData from '../data/validation_results.json'

type Row = {
  cell: string
  engine: string
  target: string
  ks: number | null
  verdict: 'pass' | 'border' | 'fail' | 'n/a'
}
type Rollup = { engine: string; n: number; n_pass: number; pass_frac: number }
type SbcTable = {
  id: string
  regime: string
  n_nodes: number
  pc_u: number
  n_attempted: number
  n_posterior: number
  band95: number
  engines: string[]
  is_calibration_claim: boolean
  rollup: Rollup[]
  rows: Row[]
}

const tables = validationData.sbc as SbcTable[]
const notes = validationData.notes as string[]
const generatedAt = (validationData.generated_at as string).slice(0, 16).replace('T', ' ')

const verdictLabel: Record<Row['verdict'], string> = {
  pass: 'pass', border: '≈ band', fail: 'fail', 'n/a': '—',
}
const ksText = (ks: number | null) => (ks === null ? '—' : ks.toFixed(3))
const regimeTitle = (r: string) =>
  r === 'well-identified' ? 'Well-identified' : r === 'stress (weak-id)' ? 'Weak-identification stress' : r
</script>

<template>
  <main class="val-page">
    <div class="container">
      <header class="val-hero">
        <div class="val-eyebrow">VALIDATION · CALIBRATION</div>
        <h1>Is the inference<br/><em>actually correct?</em></h1>
        <p class="val-lede">
          Simulation-Based Calibration: draw θ from the prior, simulate data,
          run inference, rank the truth among posterior draws. A calibrated
          procedure produces uniform ranks. Below: KS distance of the ranks to
          uniform, per model cell, engine, and quantity — ranked by PIT.
        </p>
      </header>

      <section class="how-to-read">
        <h2>How to read it</h2>
        <div class="how-grid">
          <div>
            <div class="how-tag">PIT RANKING</div>
            <p>Scalar hyperparameters are ranked by their PIT, <code>cdf(marginal, truth)</code>. Required for grid/Laplace engines — INLA draws θ from a few integration-grid points, so naive sample ranks pick up a spurious staircase.</p>
          </div>
          <div>
            <div class="how-tag">NULL BAND</div>
            <p>The 95% band is <code>1.36/√n</code>. <span class="v-pass">pass</span> ≤ band, <span class="v-border">≈ band</span> ≤ 1.6× band, <span class="v-fail">fail</span> beyond. Many cells ⇒ expect a few over the band by chance.</p>
          </div>
          <div>
            <div class="how-tag">REGIMES</div>
            <p>Well-identified (more data informs the variance component) vs weak-identification stress (heavy prior, little data). Calibration is expected to degrade under stress.</p>
          </div>
          <div>
            <div class="how-tag">:loglik</div>
            <p>A joint data-dependent test quantity — the observation log-likelihood at (θ, x) — catching joint miscalibration the per-hyperparameter ranks miss.</p>
          </div>
        </div>
      </section>

      <section v-for="t in tables" :key="t.id" class="val-section">
        <header class="val-section-head">
          <h2>{{ regimeTitle(t.regime) }}</h2>
          <p>
            {{ t.engines.join(' · ') }} ·
            n_nodes {{ t.n_nodes }} · PC u {{ t.pc_u }} ·
            {{ t.n_attempted }} replicates · band {{ t.band95.toFixed(3) }}
            <span v-if="!t.is_calibration_claim" class="smoke-flag">· smoke (not a claim)</span>
          </p>
        </header>

        <div class="rollup">
          <div v-for="r in t.rollup" :key="r.engine" class="rollup-chip"
               :class="{ good: r.pass_frac >= 0.7, weak: r.pass_frac < 0.4 }">
            <span class="rollup-eng">{{ r.engine }}</span>
            <span class="rollup-frac">{{ r.n_pass }}/{{ r.n }} pass</span>
          </div>
        </div>

        <table class="val-table">
          <thead>
            <tr><th>cell</th><th>engine</th><th>quantity</th><th class="num">KS</th><th>verdict</th></tr>
          </thead>
          <tbody>
            <tr v-for="(row, i) in t.rows" :key="i" :class="'vr-' + row.verdict.replace('/', '')">
              <td class="mono">{{ row.cell }}</td>
              <td class="mono">{{ row.engine }}</td>
              <td class="mono">{{ row.target }}</td>
              <td class="num mono">{{ ksText(row.ks) }}</td>
              <td><span class="badge" :class="'b-' + row.verdict.replace('/', '')">{{ verdictLabel[row.verdict] }}</span></td>
            </tr>
          </tbody>
        </table>
      </section>

      <section class="val-notes">
        <h2>Notes</h2>
        <ul>
          <li v-for="(n, i) in notes" :key="i">{{ n }}</li>
          <li>Gaussian-IID fails every engine by construction: <code>y~N(x,σ), x~N(0,1/τ)</code> ⇒ only <code>σ²+1/τ</code> is identified. A model pathology, not an engine defect.</li>
        </ul>
        <p class="gen">Generated {{ generatedAt }} · <code>benchmark/render_validation.jl</code></p>
      </section>
    </div>
  </main>
</template>

<style scoped>
.val-page {
  --bg:       #FAF7F2;
  --tan:      #E8D5B7;
  --caramel:  #C9986A;
  --mocha:    #8B6F47;
  --bean:     #3D2817;
  --espresso: #2A1810;
  --berry:    #C04A2A;
  --good:     #4F7A4A;
  --foam:     #FFFCF7;
  background: var(--bg);
  color: var(--espresso);
  font-family: 'Inter', system-ui, sans-serif;
  padding: 56px 0 96px;
  min-height: 60vh;
}
.container { max-width: 1100px; margin: 0 auto; padding: 0 48px; }

.val-hero { max-width: 720px; margin-bottom: 56px; }
.val-eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px; letter-spacing: 1.5px; color: var(--berry); margin-bottom: 16px;
}
.val-hero h1 {
  font-family: 'Fraunces', Georgia, serif; font-weight: 400;
  font-size: clamp(36px, 5vw, 60px); line-height: 1; letter-spacing: -0.035em; margin: 0 0 20px;
}
.val-hero h1 em { font-style: italic; color: var(--bean); }
.val-lede { font-size: 18px; line-height: 1.55; color: #4A3828; max-width: 640px; margin: 0; }

.how-to-read { margin-bottom: 64px; }
.how-to-read h2 {
  font-family: 'Fraunces', Georgia, serif; font-style: italic; font-weight: 400;
  font-size: 26px; margin: 0 0 24px;
}
.how-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 22px 32px; }
.how-tag {
  font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 1px;
  color: var(--mocha); margin-bottom: 8px;
}
.how-grid p { font-size: 14.5px; line-height: 1.5; color: #4A3828; margin: 0; }

.val-section { margin-bottom: 56px; }
.val-section-head h2 {
  font-family: 'Fraunces', Georgia, serif; font-weight: 400; font-size: 28px; margin: 0 0 6px;
}
.val-section-head p {
  font-family: 'JetBrains Mono', monospace; font-size: 12.5px; color: var(--mocha); margin: 0 0 18px;
}
.smoke-flag { color: var(--berry); }

.rollup { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 18px; }
.rollup-chip {
  display: flex; gap: 8px; align-items: baseline;
  background: var(--foam); border: 1px solid var(--tan); border-radius: 8px;
  padding: 7px 14px;
}
.rollup-chip.good { border-color: var(--good); }
.rollup-chip.weak { border-color: var(--berry); }
.rollup-eng { font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 600; }
.rollup-frac { font-size: 13px; color: #4A3828; }

.val-table { width: 100%; border-collapse: collapse; background: var(--foam); border: 1px solid var(--tan); border-radius: 10px; overflow: hidden; }
.val-table th {
  text-align: left; font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 1px;
  color: var(--mocha); padding: 10px 14px; border-bottom: 1px solid var(--tan); text-transform: uppercase;
}
.val-table td { padding: 8px 14px; font-size: 14px; border-bottom: 1px solid #F0E6D8; }
.val-table tbody tr:last-child td { border-bottom: none; }
.val-table .num { text-align: right; }
.mono { font-family: 'JetBrains Mono', monospace; font-size: 13px; }

.badge {
  font-family: 'JetBrains Mono', monospace; font-size: 11px; padding: 2px 9px; border-radius: 999px;
  letter-spacing: 0.5px;
}
.b-pass { background: rgba(79,122,74,0.15); color: var(--good); }
.b-border { background: rgba(201,152,106,0.2); color: var(--mocha); }
.b-fail { background: rgba(192,74,42,0.14); color: var(--berry); }
.b-na { background: #EFE7DA; color: var(--mocha); }
.vr-fail td { background: rgba(192,74,42,0.04); }

.v-pass { color: var(--good); font-weight: 600; }
.v-border { color: var(--mocha); font-weight: 600; }
.v-fail { color: var(--berry); font-weight: 600; }

.val-notes h2 { font-family: 'Fraunces', Georgia, serif; font-style: italic; font-weight: 400; font-size: 24px; margin: 0 0 16px; }
.val-notes ul { margin: 0 0 18px; padding-left: 20px; }
.val-notes li { font-size: 14.5px; line-height: 1.6; color: #4A3828; margin-bottom: 8px; }
.gen { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--mocha); }
code { font-family: 'JetBrains Mono', monospace; font-size: 0.9em; background: rgba(139,111,71,0.1); padding: 1px 5px; border-radius: 4px; }

@media (max-width: 760px) { .how-grid { grid-template-columns: 1fr; } .container { padding: 0 24px; } }
</style>
