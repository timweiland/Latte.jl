<script setup lang="ts">
import { ref, computed } from 'vue'
import validationData from '../data/validation_results.json'

type Verdict = 'pass' | 'minor' | 'substantial' | 'border' | 'fail' | 'n/a'
type Row = { cell: string; target: string; ks: number | null; verdict: Verdict; non_identified?: boolean; ecdf_diff?: number[] | null }
type Regime = {
  regime: string
  n_attempted: number
  n_nodes: number
  pc_u: number
  band95: number
  n: number
  n_pass: number
  n_minor: number
  n_substantial: number
  band_lo: number[]
  band_hi: number[]
  rows: Row[]
}
type Engine = { engine: string; regimes: Regime[] }

const engines = validationData.engines as Engine[]
const notes = validationData.notes as string[]
const generatedAt = (validationData.generated_at as string).slice(0, 16).replace('T', ' ')

const engineLabel: Record<string, string> = { inla: 'INLA', tmb: 'TMB', hmc_laplace: 'HMC-Laplace' }
const engName = (e: string) => engineLabel[e] ?? e
const engineBlurb: Record<string, string> = {
  inla: 'Full nested-Laplace over a hyperparameter grid. The flagship — calibrated when the model is well-posed.',
  tmb: 'Gaussian-at-MAP in working space. Fast; accurate for Gaussian-like hyperparameter posteriors, biased on skewed ones.',
  hmc_laplace: 'NUTS over the Laplace marginal. Samples the same marginal INLA integrates, so it tracks INLA.',
}

const active = ref(engines[0]?.engine ?? '')
const activeEngine = computed(() => engines.find(e => e.engine === active.value) ?? engines[0])

const verdictLabel: Record<Verdict, string> = {
  pass: 'within band', minor: 'minor', substantial: 'substantial',
  border: '≈ band', fail: 'fail', 'n/a': '—',
}
const cls = (v: string) => v.replace('/', '')

// ── ECDF-difference sparklines (Säilynoja): curve ECDF(z)−z vs the band ──
const bandZ = validationData.band_z as number[]
const PW = 132, PH = 40, YMAX = 0.2
const px = (z: number) => +(z * PW).toFixed(1)
const py = (v: number) => +(PH / 2 - Math.max(-YMAX, Math.min(YMAX, v)) / YMAX * (PH / 2 - 2)).toFixed(1)
const bandPath = (lo: number[], hi: number[]) => {
  if (!lo || !hi) return ''
  const top = bandZ.map((z, i) => `${px(z)},${py(hi[i])}`)
  const bot = bandZ.map((z, i) => `${px(z)},${py(lo[i])}`).reverse()
  return 'M' + top.join(' L') + ' L' + bot.join(' L') + ' Z'
}
const curvePath = (diff: number[] | null) =>
  diff ? 'M' + bandZ.map((z, i) => `${px(z)},${py(diff[i])}`).join(' L') : ''
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
          procedure produces uniform ranks. Pick an engine below; each tab shows
          its KS distance to uniform per model cell and quantity, ranked by PIT.
        </p>
      </header>

      <section class="how-to-read">
        <h2>How to read it</h2>
        <div class="how-grid">
          <div>
            <div class="how-tag">PIT RANKING</div>
            <p>Scalar hyperparameters ranked by their PIT, <code>cdf(marginal, truth)</code>. Required for grid/Laplace engines — INLA draws θ from a few integration-grid points, so naive sample ranks pick up a spurious staircase.</p>
          </div>
          <div>
            <div class="how-tag">VERDICT (effect-size tiered)</div>
            <p><span class="v-pass">within band</span> = passes the Säilynoja et al. (2022) 95% <em>simultaneous</em>-band ECDF test. At ~10³ replicates that test detects even tiny error, so a fail is tiered by KS: <span class="v-border">minor</span> ≤ 0.10 (approximation-level, fine), <span class="v-fail">substantial</span> &gt; 0.10. The sparkline plots each cell's rank ECDF minus uniform against that band — a curve hugging zero inside the shaded band is calibrated.</p>
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

      <div class="engine-tabs" role="tablist">
        <button v-for="e in engines" :key="e.engine" class="tab"
                :class="{ active: e.engine === active }" @click="active = e.engine">
          {{ engName(e.engine) }}
        </button>
      </div>

      <div v-if="activeEngine" class="engine-panel">
        <p class="engine-blurb">{{ engineBlurb[activeEngine.engine] ?? '' }}</p>

        <section v-for="reg in activeEngine.regimes" :key="reg.regime" class="val-section">
          <header class="val-section-head">
            <h2>{{ regimeTitle(reg.regime) }}</h2>
            <p>
              n_nodes {{ reg.n_nodes }} · PC u {{ reg.pc_u }} ·
              {{ reg.n_attempted }} replicates
              <span class="rollup-inline" :class="{ weak: reg.n_substantial > 0 }">
                · <span class="v-pass">{{ reg.n_pass }} within band</span> ·
                <span class="v-border">{{ reg.n_minor }} minor</span> ·
                <span class="v-fail">{{ reg.n_substantial }} substantial</span>
              </span>
            </p>
          </header>

          <table class="val-table">
            <thead>
              <tr><th>cell</th><th>quantity</th><th class="ecdf-h">ECDF − z vs band</th><th class="num">KS</th><th>verdict</th></tr>
            </thead>
            <tbody>
              <tr v-for="(row, i) in reg.rows" :key="i" :class="'vr-' + cls(row.verdict)">
                <td class="mono">{{ row.cell }}<span v-if="row.non_identified" class="nonid-tag" title="non-identified model: only σ²+1/τ is identified — a deliberate stress case">non-id</span></td>
                <td class="mono">{{ row.target }}</td>
                <td>
                  <svg class="spark" :viewBox="`0 0 ${PW} ${PH}`" preserveAspectRatio="none" aria-hidden="true">
                    <path class="spark-band" :d="bandPath(reg.band_lo, reg.band_hi)" />
                    <line class="spark-zero" :x1="0" :y1="py(0)" :x2="PW" :y2="py(0)" />
                    <path class="spark-curve" :class="'sc-' + cls(row.verdict)" :d="curvePath(row.ecdf_diff)" />
                  </svg>
                </td>
                <td class="num mono">{{ ksText(row.ks) }}</td>
                <td><span class="badge" :class="'b-' + cls(row.verdict)">{{ verdictLabel[row.verdict] }}</span></td>
              </tr>
            </tbody>
          </table>
        </section>
      </div>

      <section class="val-notes">
        <h2>Notes</h2>
        <ul>
          <li v-for="(n, i) in notes" :key="i">{{ n }}</li>
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

.how-to-read { margin-bottom: 56px; }
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

/* ── Engine tabs ── */
.engine-tabs { display: flex; gap: 6px; border-bottom: 2px solid var(--tan); margin-bottom: 8px; }
.tab {
  font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 600;
  background: transparent; border: none; cursor: pointer;
  color: var(--mocha); padding: 10px 20px; border-bottom: 3px solid transparent;
  margin-bottom: -2px; transition: color .15s, border-color .15s;
}
.tab:hover { color: var(--bean); }
.tab.active { color: var(--berry); border-bottom-color: var(--berry); }
.engine-panel { padding-top: 22px; }
.engine-blurb { font-size: 15px; line-height: 1.5; color: #4A3828; max-width: 640px; margin: 0 0 28px; font-style: italic; }

.val-section { margin-bottom: 40px; }
.val-section-head h2 {
  font-family: 'Fraunces', Georgia, serif; font-weight: 400; font-size: 24px; margin: 0 0 6px;
}
.val-section-head p {
  font-family: 'JetBrains Mono', monospace; font-size: 12.5px; color: var(--mocha); margin: 0 0 16px;
}
.rollup-inline.good { color: var(--good); }
.rollup-inline.weak { color: var(--berry); }

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
  font-family: 'JetBrains Mono', monospace; font-size: 11px; padding: 2px 9px; border-radius: 999px; letter-spacing: 0.5px;
}
.b-pass { background: rgba(79,122,74,0.15); color: var(--good); }
.b-minor, .b-border { background: rgba(201,152,106,0.22); color: #8a5a1e; }
.b-substantial, .b-fail { background: rgba(192,74,42,0.14); color: var(--berry); }
.b-na { background: #EFE7DA; color: var(--mocha); }
.vr-substantial td, .vr-fail td { background: rgba(192,74,42,0.04); }
.nonid-tag { font-family: 'JetBrains Mono', monospace; font-size: 9.5px; letter-spacing: 0.5px; color: var(--mocha); background: rgba(139,111,71,0.12); border-radius: 4px; padding: 1px 5px; margin-left: 8px; vertical-align: middle; }

.ecdf-h { width: 140px; }
.spark { width: 132px; height: 40px; display: block; }
.spark-band { fill: rgba(139,111,71,0.14); stroke: none; }
.spark-zero { stroke: rgba(139,111,71,0.45); stroke-width: 0.5; stroke-dasharray: 2 2; }
.spark-curve { fill: none; stroke-width: 1.4; }
.sc-pass { stroke: var(--good); }
.sc-minor, .sc-border { stroke: #C98A3A; }
.sc-substantial, .sc-fail { stroke: var(--berry); }

.v-pass { color: var(--good); font-weight: 600; }
.v-border { color: var(--mocha); font-weight: 600; }
.v-fail { color: var(--berry); font-weight: 600; }

.val-notes { margin-top: 48px; }
.val-notes h2 { font-family: 'Fraunces', Georgia, serif; font-style: italic; font-weight: 400; font-size: 24px; margin: 0 0 16px; }
.val-notes ul { margin: 0 0 18px; padding-left: 20px; }
.val-notes li { font-size: 14.5px; line-height: 1.6; color: #4A3828; margin-bottom: 8px; }
.gen { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--mocha); }
code { font-family: 'JetBrains Mono', monospace; font-size: 0.9em; background: rgba(139,111,71,0.1); padding: 1px 5px; border-radius: 4px; }

@media (max-width: 760px) { .how-grid { grid-template-columns: 1fr; } .container { padding: 0 24px; } }
</style>
