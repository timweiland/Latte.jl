<script setup lang="ts">
import strip from '../data/overlay_strip.json'

// One representative marginal per benchmark: the Latte posterior density drawn
// over the R-INLA density on a shared grid. Data from the per-benchmark
// *_strip_overlay.jl exports, aggregated into overlay_strip.json.
type Rec = {
  id: string; dataset: string; scenario: string; label: string
  ks: number; speedup: number; grid: number[]; latte: number[]; rinla: number[]
}

const TW = 230, TH = 116, PAD = 8, BASE = 98, TOP = 12

function paths(r: Rec) {
  const g = r.grid, gmin = g[0], gmax = g[g.length - 1]
  const dmax = Math.max(...r.latte, ...r.rinla) * 1.08 || 1
  const gx = (v: number) => +(PAD + (v - gmin) / (gmax - gmin) * (TW - 2 * PAD)).toFixed(1)
  const gy = (d: number) => +(BASE - (d / dmax) * (BASE - TOP)).toFixed(1)
  const ln = (ys: number[]) => 'M' + g.map((v, i) => `${gx(v)},${gy(ys[i])}`).join(' L')
  const latte = ln(r.latte)
  return { latte, rinla: ln(r.rinla), area: `${latte} L${gx(gmax)},${BASE} L${gx(gmin)},${BASE} Z` }
}

const fmtSp = (s: number) => s >= 100 ? `${Math.round(s)}×` : `${s.toFixed(1)}×`
const tiles = (strip as Rec[]).map(r => ({ ...r, p: paths(r) }))
</script>

<template>
  <figure class="os">
    <div class="os-head">
      <div class="os-eyebrow">POSTERIORS · LATTE vs R-INLA</div>
      <h3>Marginal by marginal, <em>side by side.</em></h3>
      <p class="os-sub">a representative marginal per model · Latte drawn over R-INLA · warm fit</p>
    </div>

    <div class="os-grid">
      <figure v-for="r in tiles" :key="r.id" class="os-tile">
        <svg :viewBox="`0 0 ${TW} ${TH}`" class="os-svg" role="img"
             :aria-label="`${r.dataset}: Latte and R-INLA posterior densities overlaid`">
          <line class="os-ax" :x1="PAD" :y1="BASE" :x2="TW - PAD" :y2="BASE" />
          <path class="os-fill" :d="r.p.area" />
          <path class="os-rinla" :d="r.p.rinla" />
          <path class="os-latte" :d="r.p.latte" />
        </svg>
        <figcaption class="os-cap">
          <span class="os-name">{{ r.dataset }}</span>
          <span class="os-meta">{{ r.label }} · KS {{ r.ks.toFixed(3) }} · {{ fmtSp(r.speedup) }}</span>
        </figcaption>
      </figure>
    </div>

    <div class="os-legend">
      <span class="lg latte-lg">Latte INLA</span>
      <span class="lg rinla-lg">R-INLA</span>
    </div>
  </figure>
</template>

<style scoped>
.os {
  --tan: #E8D5B7; --caramel: #C9986A; --mocha: #8B6F47; --bean: #3D2817;
  --berry: #C04A2A; --foam: #FFFCF7;
  max-width: 880px; margin: 0 auto 8px; padding: 30px 34px;
  background: var(--foam); border: 1px solid var(--tan); border-radius: 14px;
  font-family: 'Inter', system-ui, sans-serif;
}
.os-head { text-align: center; margin-bottom: 18px; }
.os-eyebrow { font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 1.4px; color: var(--caramel); margin-bottom: 10px; }
.os-head h3 { font-family: 'Fraunces', Georgia, serif; font-style: italic; font-weight: 400; font-size: 27px; letter-spacing: -0.4px; margin: 0; color: var(--bean); line-height: 1.12; }
.os-head h3 em { font-style: italic; color: var(--berry); }
.os-sub { font-size: 12.5px; color: var(--mocha); margin: 8px 0 0; font-family: 'JetBrains Mono', monospace; }

.os-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px 22px; }
.os-tile { margin: 0; }
.os-svg { width: 100%; height: auto; display: block; }
.os-ax { stroke: rgba(139, 111, 71, 0.45); stroke-width: 1; }
.os-fill { fill: var(--berry); opacity: 0.08; }
.os-latte { fill: none; stroke: var(--berry); stroke-width: 2.2; }
.os-rinla { fill: none; stroke: var(--bean); stroke-width: 1.4; stroke-dasharray: 4 3; }
.os-cap { display: flex; flex-direction: column; gap: 1px; margin-top: 4px; }
.os-name { font-size: 12.5px; font-weight: 600; color: var(--bean); }
.os-meta { font-size: 11px; color: var(--mocha); font-family: 'JetBrains Mono', monospace; }

.os-legend { display: flex; gap: 18px; align-items: center; justify-content: center; margin-top: 20px; }
.lg { font-size: 12px; color: #4A3828; display: inline-flex; align-items: center; }
.lg::before { content: ''; width: 16px; height: 0; border-top-width: 2.5px; border-top-style: solid; margin-right: 7px; }
.latte-lg::before { border-top-color: var(--berry); }
.rinla-lg::before { border-top-color: var(--bean); border-top-style: dashed; }
</style>
