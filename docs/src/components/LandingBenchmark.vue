<script setup lang="ts">
import { ref, computed } from 'vue'
import D from '../data/landing_overlay.json'

// Real numbers from benchmark/external/rinla/tokyo (tokyo_export_overlay.jl):
// warm-fit wall-clock for Latte INLA vs R-INLA, plus both posterior density
// curves for two days — the median-KS day (typical agreement) and the worst-KS
// day (honest bound). The tabs toggle between them; times are shared.
const latteT = D.t_latte_warm
const rinlaT = D.t_rinla

const tab = ref('median')
const active = computed(() => D.cases.find((c: any) => c.key === tab.value) || D.cases[0])

// ── bars (static across tabs) ──
const BW = 300, BH = 118, BX = 4, MAXBAR = 196
const barLatte = +(MAXBAR * latteT / rinlaT).toFixed(1)
const fmtT = (t: number) => t >= 1 ? `${t.toFixed(2)} s` : `${Math.round(t * 1000)} ms`

// ── overlay (per active day) ──
const OW = 300, OH = 150, PAD = 10, AX = 128, TOP = 14
const grid = computed(() => active.value.grid as number[])
const gmin = computed(() => grid.value[0])
const gmax = computed(() => grid.value[grid.value.length - 1])
const dmax = computed(() => Math.max(...active.value.latte, ...active.value.rinla) * 1.06)
const gx = (v: number) => +(PAD + (v - gmin.value) / (gmax.value - gmin.value) * (OW - 2 * PAD)).toFixed(1)
const gy = (d: number) => +(AX - (d / dmax.value) * (AX - TOP)).toFixed(1)
const line = (ys: number[]) => 'M' + grid.value.map((v, i) => `${gx(v)},${gy(ys[i])}`).join(' L')
const pLatte = computed(() => line(active.value.latte))
const pRinla = computed(() => line(active.value.rinla))
const area = computed(() => `${line(active.value.latte)} L${gx(gmax.value)},${AX} L${gx(gmin.value)},${AX} Z`)
</script>

<template>
  <figure class="lb">
    <div class="lb-head">
      <div class="lb-eyebrow">BENCHMARK · {{ D.dataset.toUpperCase() }}</div>
      <h3>Gold-standard accuracy, <em>in milliseconds.</em></h3>
      <p class="lb-sub">{{ D.scenario }} · warm fit</p>
    </div>

    <div class="lb-tabs">
      <button v-for="c in D.cases" :key="c.key"
              :class="['lb-tab', { active: tab === c.key }]" @click="tab = c.key">{{ c.label }}</button>
    </div>

    <div class="lb-panels">
      <div class="lb-panel">
        <div class="lb-cap">warm-fit time</div>
        <svg :viewBox="`0 0 ${BW} ${BH}`" class="lb-svg">
          <text class="bar-name" :x="BX" :y="18">Latte INLA</text>
          <rect class="bar-latte" :x="BX" :y="24" :width="barLatte" height="18" rx="2" />
          <text class="bar-val latte-val" :x="BX + barLatte + 8" :y="38">{{ fmtT(latteT) }}</text>
          <text class="bar-name" :x="BX" :y="74">R-INLA</text>
          <rect class="bar-rinla" :x="BX" :y="80" :width="MAXBAR" height="18" rx="2" />
          <text class="bar-val" :x="BX + MAXBAR + 8" :y="94">{{ fmtT(rinlaT) }}</text>
        </svg>
      </div>

      <div class="lb-panel">
        <div class="lb-cap">posterior · day #{{ active.day }} of 366 · KS {{ active.ks }}</div>
        <svg :viewBox="`0 0 ${OW} ${OH}`" class="lb-svg">
          <line class="ax" :x1="PAD" :y1="AX" :x2="OW - PAD" :y2="AX" />
          <path class="ov-fill" :d="area" />
          <path class="ov-latte" :d="pLatte" />
          <path class="ov-rinla" :d="pRinla" />
        </svg>
      </div>
    </div>

    <div class="lb-legend">
      <span class="lg latte-lg">Latte INLA</span>
      <span class="lg rinla-lg">R-INLA</span>
      <a class="lb-link" href="/benchmarks/">See all benchmarks →</a>
    </div>
  </figure>
</template>

<style scoped>
.lb {
  --tan: #E8D5B7; --caramel: #C9986A; --mocha: #8B6F47; --bean: #3D2817;
  --berry: #C04A2A; --foam: #FFFCF7;
  max-width: 760px; margin: 0 auto; padding: 30px 34px;
  background: var(--foam); border: 1px solid var(--tan); border-radius: 14px;
  font-family: 'Inter', system-ui, sans-serif;
}
.lb-head { text-align: center; margin-bottom: 16px; }
.lb-eyebrow { font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 1.4px; color: var(--caramel); margin-bottom: 10px; }
.lb-head h3 { font-family: 'Fraunces', Georgia, serif; font-style: italic; font-weight: 400; font-size: 27px; letter-spacing: -0.4px; margin: 0; color: var(--bean); line-height: 1.12; }
.lb-head h3 em { font-style: italic; color: var(--berry); }
.lb-sub { font-size: 12.5px; color: var(--mocha); margin: 8px 0 0; font-family: 'JetBrains Mono', monospace; }

.lb-tabs { display: flex; justify-content: center; gap: 8px; margin-bottom: 18px; }
.lb-tab {
  font-family: 'Inter', sans-serif; font-size: 12.5px; font-weight: 500;
  padding: 6px 16px; border-radius: 999px; cursor: pointer;
  background: transparent; color: var(--mocha); border: 1px solid var(--tan);
  transition: background .12s, color .12s, border-color .12s;
}
.lb-tab:hover { border-color: var(--caramel); color: var(--bean); }
.lb-tab.active { background: var(--berry); color: #fff; border-color: var(--berry); }

.lb-panels { display: flex; gap: 30px; align-items: center; }
.lb-panel { flex: 1; min-width: 0; }
.lb-cap { font-size: 11.5px; color: var(--mocha); margin-bottom: 4px; text-align: center; }
.lb-svg { width: 100%; height: auto; display: block; }

.bar-name { fill: var(--bean); font-size: 12.5px; font-family: 'Inter', sans-serif; font-weight: 500; }
.bar-latte { fill: var(--berry); }
.bar-rinla { fill: var(--caramel); }
.bar-val { fill: var(--mocha); font-size: 12px; font-family: 'JetBrains Mono', monospace; dominant-baseline: middle; }
.bar-val.latte-val { fill: var(--berry); font-weight: 600; }

.ax { stroke: rgba(139,111,71,0.45); stroke-width: 1; }
.ov-fill { fill: var(--berry); opacity: 0.09; }
.ov-latte { fill: none; stroke: var(--berry); stroke-width: 2.3; }
.ov-rinla { fill: none; stroke: var(--bean); stroke-width: 1.5; stroke-dasharray: 4 3; }

.lb-legend { display: flex; gap: 18px; align-items: center; justify-content: center; margin-top: 18px; flex-wrap: wrap; }
.lg { font-size: 12px; color: #4A3828; display: inline-flex; align-items: center; }
.lg::before { content: ''; width: 16px; height: 0; border-top-width: 2.5px; border-top-style: solid; margin-right: 7px; }
.latte-lg::before { border-top-color: var(--berry); }
.rinla-lg::before { border-top-color: var(--bean); border-top-style: dashed; }
.lb-link { font-family: 'JetBrains Mono', monospace; font-size: 11.5px; color: var(--berry); text-decoration: none; margin-left: 4px; }
.lb-link:hover { text-decoration: underline; }

@media (max-width: 720px) {
  .lb-panels { flex-direction: column; gap: 18px; }
}
</style>

<style>
/* Dark mode. Non-scoped on purpose: this project's CSS build silently drops
   Vue scoped :global(.dark) rules. Every rule is nested under .lb and gated on
   html.dark so it can't leak and outranks the scoped light defaults. */
html.dark .lb {
  --foam: #38241B;                       /* light surface  -> dark surface */
  --tan: rgba(201,152,106,0.2);          /* border/gridline */
  --caramel: #C9986A;                    /* accent (unchanged) */
  --mocha: #B79877;                      /* muted text/axis */
  --bean: #D4B896;                       /* used as text/stroke -> light */
  --berry: #D9603F;                      /* red accent */
}

/* Hardcoded legend text color (#4A3828, a dark text) -> light. */
html.dark .lb .lg { color: #D4B896; }

/* Axis line (warm-brown stroke) would vanish on the dark surface; lighten it. */
html.dark .lb .ax { stroke: rgba(212,184,150,0.4); }
</style>
