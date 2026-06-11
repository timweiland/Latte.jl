<script setup lang="ts">
import benchmarkData from '../data/benchmark_results.json'

// One point per reference-comparison benchmark: x = warm speedup vs R-INLA
// (log), y = median KS of the primary latent-marginal block (typical agreement),
// with a whisker up to the worst single component. Numbers come from
// `benchmark/render_docs.jl` (the per-receipt `summary`).
type Pt = { id: string; speedup: number; ksMed: number; ksMax: number }

const pts: Pt[] = (benchmarkData.receipts as any[])
  .filter(r => r.summary && r.summary.ks_median != null)
  .map(r => ({ id: r.id, speedup: r.summary.speedup, ksMed: r.summary.ks_median, ksMax: r.summary.ks_max }))

// ── geometry ──
const W = 620, H = 320
const ML = 52, MR = 132, MT = 22, MB = 50
const x0 = ML, x1 = W - MR, y0 = H - MB, y1 = MT
const SPMAX = 800     // log-x headroom past 607×
const KSMAX = 0.13    // y headroom past 0.123
const BAND = 0.05     // "indistinguishable" threshold

const gx = (sp: number) => +(x0 + Math.log10(sp) / Math.log10(SPMAX) * (x1 - x0)).toFixed(1)
const gy = (ks: number) => +(y0 - (ks / KSMAX) * (y0 - y1)).toFixed(1)

const xticks = [1, 3, 10, 30, 100, 300]
const yticks = [0, 0.05, 0.1]
const yMid = (y0 + y1) / 2

const shortName: Record<string, string> = {
  seeds: 'seeds', scotland: 'scotland', nhtemp: 'nhtemp', tokyo_rainfall: 'tokyo',
  epil: 'epil', spdetoy: 'spdetoy', paranaprec: 'paraná',
}
const label = (p: Pt) =>
  `${shortName[p.id] || p.id} · ${p.speedup >= 100 ? Math.round(p.speedup) : p.speedup.toFixed(1)}×`
</script>

<template>
  <figure class="sc">
    <div class="sc-head">
      <div class="sc-eyebrow">REFERENCE COMPARISON · WARM FIT</div>
      <h3>Accuracy against <em>wall-clock.</em></h3>
      <p class="sc-sub">each point is one model · vs R-INLA on the same posterior</p>
    </div>

    <svg :viewBox="`0 0 ${W} ${H}`" class="sc-svg" role="img"
         aria-label="Scatter of warm speedup versus KS divergence from R-INLA, one point per benchmark.">
      <!-- indistinguishable band (KS < 0.05) -->
      <rect class="sc-band" :x="x0" :y="gy(BAND)" :width="x1 - x0" :height="y0 - gy(BAND)" />
      <text class="sc-band-lbl" :x="x0 + 8" :y="gy(BAND) - 6">KS &lt; 0.05</text>

      <!-- axes -->
      <line class="sc-ax" :x1="x0" :y1="y0" :x2="x1" :y2="y0" />
      <line class="sc-ax" :x1="x0" :y1="y0" :x2="x0" :y2="y1" />

      <!-- x ticks -->
      <g v-for="t in xticks" :key="'x' + t">
        <line class="sc-tick" :x1="gx(t)" :y1="y0" :x2="gx(t)" :y2="y0 + 4" />
        <text class="sc-tlab" :x="gx(t)" :y="y0 + 16" text-anchor="middle">{{ t }}×</text>
      </g>
      <text class="sc-axlab" :x="(x0 + x1) / 2" :y="H - 8" text-anchor="middle">warm speedup vs R-INLA  (log) →</text>

      <!-- y ticks -->
      <g v-for="t in yticks" :key="'y' + t">
        <line class="sc-tick" :x1="x0 - 4" :y1="gy(t)" :x2="x0" :y2="gy(t)" />
        <text class="sc-tlab" :x="x0 - 8" :y="gy(t) + 3" text-anchor="end">{{ t.toFixed(2) }}</text>
      </g>
      <text class="sc-axlab" :x="14" :y="yMid" text-anchor="middle"
            :transform="`rotate(-90 14 ${yMid})`">KS vs R-INLA · lower = closer</text>

      <!-- points -->
      <g v-for="p in pts" :key="p.id">
        <line class="sc-whisker" :x1="gx(p.speedup)" :y1="gy(p.ksMed)" :x2="gx(p.speedup)" :y2="gy(p.ksMax)" />
        <circle class="sc-dot" :cx="gx(p.speedup)" :cy="gy(p.ksMed)" r="5" />
        <text class="sc-plab" :x="gx(p.speedup) + 9" :y="gy(p.ksMed) + 3.5">{{ label(p) }}</text>
      </g>
    </svg>

    <p class="sc-foot">
      Dot = median KS of the latent marginals; whisker reaches the worst single component.
      Large speedups at small <em>n</em> partly reflect R-INLA's fixed per-call overhead, not raw compute.
      Paraná's SPDE field sits above the band — weakly identified (variance- not mean-limited), a floor both
      engines hit. Full per-component numbers in the receipts below.
    </p>
  </figure>
</template>

<style scoped>
.sc {
  --tan: #E8D5B7; --caramel: #C9986A; --mocha: #8B6F47; --bean: #3D2817;
  --berry: #C04A2A; --foam: #FFFCF7;
  max-width: 760px; margin: 0 auto 8px; padding: 30px 34px;
  background: var(--foam); border: 1px solid var(--tan); border-radius: 14px;
  font-family: 'Inter', system-ui, sans-serif;
}
.sc-head { text-align: center; margin-bottom: 14px; }
.sc-eyebrow { font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 1.4px; color: var(--caramel); margin-bottom: 10px; }
.sc-head h3 { font-family: 'Fraunces', Georgia, serif; font-style: italic; font-weight: 400; font-size: 27px; letter-spacing: -0.4px; margin: 0; color: var(--bean); line-height: 1.12; }
.sc-head h3 em { font-style: italic; color: var(--berry); }
.sc-sub { font-size: 12.5px; color: var(--mocha); margin: 8px 0 0; font-family: 'JetBrains Mono', monospace; }

.sc-svg { width: 100%; height: auto; display: block; }
.sc-band { fill: var(--berry); opacity: 0.07; }
.sc-band-lbl { fill: var(--caramel); font-size: 11px; font-family: 'JetBrains Mono', monospace; }
.sc-ax { stroke: rgba(139, 111, 71, 0.55); stroke-width: 1; }
.sc-tick { stroke: rgba(139, 111, 71, 0.55); stroke-width: 1; }
.sc-tlab { fill: var(--mocha); font-size: 11px; font-family: 'JetBrains Mono', monospace; }
.sc-axlab { fill: var(--mocha); font-size: 11px; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.3px; }
.sc-whisker { stroke: var(--caramel); stroke-width: 1.4; opacity: 0.55; }
.sc-dot { fill: var(--berry); }
.sc-plab { fill: var(--bean); font-size: 12px; font-family: 'JetBrains Mono', monospace; }

.sc-foot { font-size: 11.5px; color: var(--mocha); margin: 12px 2px 0; line-height: 1.5; }
</style>
