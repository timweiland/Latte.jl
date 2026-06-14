<script setup lang="ts">
import { reactive } from 'vue'
import cards from '../data/benchmark_cards.json'

// One card per benchmark. Speed (Latte vs R-INLA) is the headline — always
// shown. Tabs flip through the accuracy checks: each marginal (latent node +
// every hyperparameter) and the KS spread over all components.
type Marg = { kind: string; label: string; ks: number; grid: number[]; latte: number[]; rinla: number[] }
type Card = {
  id: string; title: string; scenario: string
  timing: { warm: number; cold: number; rinla: number; speedup: number }
  ks_block: { label: string; n: number; max: number; median: number; edges: number[]; counts: number[] }
  marginals: Marg[]
}

const fmtT = (t: number) => t >= 1 ? `${t.toFixed(2)} s` : `${Math.round(t * 1000)} ms`
const fmtSp = (s: number) => s >= 100 ? `${Math.round(s)}×` : `${s.toFixed(1)}×`

// ── speed bars (always shown) ──
const SBMAX = 208
const speedW = (c: Card) => +Math.max(SBMAX * Math.min(c.timing.warm / c.timing.rinla, 1), 2.5).toFixed(1)

// ── one overlay (shown one at a time → roomy) ──
const OW = 300, OH = 136, OPAD = 10, OBASE = 116, OTOP = 12
function ovPaths(m: Marg) {
  const g = m.grid, gmin = g[0], gmax = g[g.length - 1]
  const dmax = Math.max(...m.latte, ...m.rinla) * 1.08 || 1
  const gx = (v: number) => +(OPAD + (v - gmin) / (gmax - gmin) * (OW - 2 * OPAD)).toFixed(1)
  const gy = (d: number) => +(OBASE - (d / dmax) * (OBASE - OTOP)).toFixed(1)
  const ln = (ys: number[]) => 'M' + g.map((v, i) => `${gx(v)},${gy(ys[i])}`).join(' L')
  const latte = ln(m.latte)
  return { latte, rinla: ln(m.rinla), area: `${latte} L${gx(gmax)},${OBASE} L${gx(gmin)},${OBASE} Z` }
}

// ── KS histogram ──
const HW = 300, HH = 136, HL = 6, HB = 22, HT = 12
function hist(c: Card) {
  const { edges, counts } = c.ks_block
  const nb = counts.length, hi = edges[edges.length - 1], cmax = Math.max(...counts) || 1
  const bw = (HW - HL) / nb
  const bars = counts.map((n, i) => ({
    x: +(HL + i * bw).toFixed(1), y: +(HH - HB - (n / cmax) * (HH - HB - HT)).toFixed(1),
    w: +Math.max(bw - 1, 1).toFixed(1), h: +((n / cmax) * (HH - HB - HT)).toFixed(1),
  }))
  const xAt = (v: number) => +(HL + (v / hi) * (HW - HL)).toFixed(1)
  return { bars, x05: xAt(0.05), ticks: [0, 0.05, 0.1].filter(t => t <= hi).map(t => ({ t, x: xAt(t) })), yBase: HH - HB }
}

// short tab label per marginal
function tabOf(m: Marg) {
  if (m.kind === 'latent') return 'latent'
  return m.label.replace('subject precision τ', 'subj τ').replace('obs precision τ', 'obs τ')
    .replace('precision τ', 'τ').replace('field range', 'range')
}

const view = (cards as Card[]).map(c => ({
  ...c, margs: c.marginals.map(m => ({ ...m, p: ovPaths(m), tab: tabOf(m) })), histD: hist(c), sw: speedW(c),
}))
// active accuracy view per card: a marginal index, or 'spread'
const active = reactive<Record<string, number | string>>({})
view.forEach(c => { active[c.id] = 0 })
</script>

<template>
  <div class="bc-grid">
    <figure v-for="c in view" :key="c.id" class="bc">
      <div class="bc-head">
        <div class="bc-title">{{ c.title }}</div>
        <div class="bc-scn">{{ c.scenario }}</div>
      </div>

      <!-- SPEED — always visible (the headline) -->
      <div class="bc-speed">
        <svg viewBox="0 0 300 78" class="bc-svg">
          <text class="sp-name" x="2" y="13">Latte INLA · warm</text>
          <rect class="sp-latte" x="2" y="18" :width="c.sw" height="15" rx="2" />
          <text class="sp-val sp-latte-v" :x="c.sw + 8" y="30">{{ fmtT(c.timing.warm) }}</text>
          <text class="sp-name" x="2" y="55">R-INLA</text>
          <rect class="sp-rinla" x="2" y="60" :width="SBMAX" height="15" rx="2" />
          <text class="sp-val" :x="SBMAX + 8" y="72">{{ fmtT(c.timing.rinla) }}</text>
        </svg>
        <div class="bc-speed-cap"><strong>{{ fmtSp(c.timing.speedup) }}</strong> faster, warm · cold {{ c.timing.cold }} s</div>
      </div>

      <!-- TABS — accuracy checks -->
      <div class="bc-tabs">
        <button v-for="(m, i) in c.margs" :key="i" :class="['bc-tab', { active: active[c.id] === i }]"
                @click="active[c.id] = i">{{ m.tab }}</button>
        <button :class="['bc-tab', { active: active[c.id] === 'spread' }]" @click="active[c.id] = 'spread'">KS spread</button>
      </div>

      <!-- ACTIVE VIEW -->
      <div class="bc-view">
        <template v-if="active[c.id] === 'spread'">
          <svg :viewBox="`0 0 ${HW} ${HH}`" class="bc-svg">
            <line class="bc-ax" :x1="HL" :y1="c.histD.yBase" :x2="HW" :y2="c.histD.yBase" />
            <rect v-for="(b, i) in c.histD.bars" :key="i" class="hist-bar" :x="b.x" :y="b.y" :width="b.w" :height="b.h" />
            <line class="hist-05" :x1="c.histD.x05" :y1="HT" :x2="c.histD.x05" :y2="c.histD.yBase" />
            <text class="hist-lbl" :x="c.histD.x05 + 4" :y="HT + 9">0.05</text>
            <text v-for="tk in c.histD.ticks" :key="tk.t" class="hist-lbl" :x="tk.x" :y="HH - 7" text-anchor="middle">{{ tk.t.toFixed(2) }}</text>
          </svg>
          <div class="bc-cap">KS over all {{ c.ks_block.n }} {{ c.ks_block.label }} · max {{ c.ks_block.max }} · median {{ c.ks_block.median }}</div>
        </template>
        <template v-else>
          <svg :viewBox="`0 0 ${OW} ${OH}`" class="bc-svg">
            <line class="bc-ax" :x1="OPAD" :y1="OBASE" :x2="OW - OPAD" :y2="OBASE" />
            <path class="ov-fill" :d="c.margs[active[c.id]].p.area" />
            <path class="ov-rinla" :d="c.margs[active[c.id]].p.rinla" />
            <path class="ov-latte" :d="c.margs[active[c.id]].p.latte" />
          </svg>
          <div class="bc-cap">
            <span :class="{ 'bc-hyper': c.margs[active[c.id]].kind === 'hyper' }">{{ c.margs[active[c.id]].label }}</span>
            <span class="bc-dot">·</span> KS {{ c.margs[active[c.id]].ks.toFixed(3) }}
            <span class="bc-lg"><span class="lg latte-lg">Latte</span><span class="lg rinla-lg">R-INLA</span></span>
          </div>
        </template>
      </div>
    </figure>
  </div>
</template>

<style scoped>
.bc-grid {
  --tan: #E8D5B7; --caramel: #C9986A; --mocha: #8B6F47; --bean: #3D2817;
  --berry: #C04A2A; --foam: #FFFCF7;
  display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px;
  font-family: 'Inter', system-ui, sans-serif;
}
.bc { margin: 0; padding: 20px 22px; background: var(--foam); border: 1px solid var(--tan); border-radius: 14px; }
.bc-head { margin-bottom: 12px; }
.bc-title { font-family: 'Fraunces', Georgia, serif; font-style: italic; font-size: 20px; color: var(--bean); line-height: 1.1; }
.bc-scn { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--mocha); margin-top: 3px; }

.bc-speed { padding: 12px 0 14px; border-top: 1px solid var(--tan); border-bottom: 1px solid var(--tan); }
.bc-speed-cap { font-size: 12px; color: var(--mocha); font-family: 'JetBrains Mono', monospace; margin-top: 4px; }
.bc-speed-cap strong { color: var(--berry); font-weight: 600; }
.sp-name { fill: var(--bean); font-size: 11.5px; font-family: 'Inter', sans-serif; font-weight: 500; }
.sp-latte { fill: var(--berry); }
.sp-rinla { fill: var(--caramel); }
.sp-val { fill: var(--mocha); font-size: 11.5px; font-family: 'JetBrains Mono', monospace; dominant-baseline: middle; }
.sp-val.sp-latte-v { fill: var(--berry); font-weight: 600; }

.bc-tabs { display: flex; flex-wrap: wrap; gap: 5px; margin: 14px 0 10px; }
.bc-tab {
  font-family: 'JetBrains Mono', monospace; font-size: 11px; padding: 4px 10px; border-radius: 999px;
  cursor: pointer; background: transparent; color: var(--mocha); border: 1px solid var(--tan);
  transition: background .12s, color .12s, border-color .12s;
}
.bc-tab:hover { border-color: var(--caramel); color: var(--bean); }
.bc-tab.active { background: var(--berry); color: #fff; border-color: var(--berry); }

.bc-view { min-height: 150px; }
.bc-svg { width: 100%; height: auto; display: block; }
.bc-ax { stroke: rgba(139, 111, 71, 0.45); stroke-width: 1; }
.ov-fill { fill: var(--berry); opacity: 0.08; }
.ov-latte { fill: none; stroke: var(--berry); stroke-width: 2.1; }
.ov-rinla { fill: none; stroke: var(--bean); stroke-width: 1.3; stroke-dasharray: 4 3; }
.hist-bar { fill: var(--berry); opacity: 0.78; }
.hist-05 { stroke: var(--mocha); stroke-width: 1; stroke-dasharray: 3 3; }
.hist-lbl { fill: var(--mocha); font-size: 10px; font-family: 'JetBrains Mono', monospace; }

.bc-cap { font-size: 11.5px; color: var(--mocha); font-family: 'JetBrains Mono', monospace; margin-top: 6px; display: flex; align-items: center; flex-wrap: wrap; gap: 6px; }
.bc-cap > span:first-child { color: var(--bean); font-weight: 600; font-family: 'Inter', sans-serif; }
.bc-hyper { color: var(--caramel) !important; }
.bc-dot { color: var(--tan); }
.bc-lg { margin-left: auto; display: inline-flex; gap: 12px; }
.lg { display: inline-flex; align-items: center; color: #4A3828; }
.lg::before { content: ''; width: 13px; height: 0; border-top-width: 2.5px; border-top-style: solid; margin-right: 5px; }
.latte-lg::before { border-top-color: var(--berry); }
.rinla-lg::before { border-top-color: var(--bean); border-top-style: dashed; }
</style>

<style>
/* Dark mode. Non-scoped on purpose: this project's CSS build silently drops
   Vue scoped :global(.dark) rules, so we scope by hand under the unique
   .bc-grid root and prefix with html.dark. Cannot leak; outranks the scoped
   light defaults. Light mode is left untouched — this only ADDS dark behavior. */
html.dark .bc-grid {
  --foam: #38241B;                       /* light surface  → dark surface     */
  --tan: rgba(201, 152, 106, 0.2);       /* border/gridline → faint warm line */
  --caramel: #C9986A;                    /* accent          → kept            */
  --mocha: #B79877;                      /* muted text/axis → lifted          */
  --bean: #D4B896;                       /* dark text/stroke → light text     */
  --berry: #D9603F;                      /* red accent      → brighter red    */
}
/* --bean is text/stroke only (titles, R-INLA dashed line, caption headings),
   never a fill background, so it flips to a light tone. */

/* Hardcoded colors */
html.dark .bc-grid .bc-ax { stroke: rgba(201, 152, 106, 0.4); }   /* axis: warm dark → visible warm */
html.dark .bc-grid .lg { color: #D4B896; }                        /* #4A3828 legend text → light */
</style>
