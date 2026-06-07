<script setup lang="ts">
import { ref, computed } from 'vue'

// ── HMC-Laplace, made visual ─────────────────────────────────────────
// NUTS draws θ from the true (skewed) p(θ|y) — the dots on the left. The point of
// this widget: each draw is NOT just a sample. Every draw runs an inner Laplace
// over the latent field, giving a Gaussian N(m(θ),s(θ)) — the faint curves on the
// right, one per draw, with one highlighted. Their average is the latent marginal.
const MU = 1.0, SL = 0.5, SR = 1.15               // split-normal p(θ|y): right-skewed
const TH0 = -2.4, TH1 = 6.0                        // θ plotting range
const X0 = -2.4, X1 = 5.0                          // xᵢ plotting range
const SQRT2PI = Math.sqrt(2 * Math.PI)
const npdf = (x: number, m: number, s: number) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * SQRT2PI)
const splitUn = (t: number) => Math.exp(-0.5 * ((t - MU) / (t < MU ? SL : SR)) ** 2)
const mOf = (t: number) => 0.6 * t
const sOf = (t: number) => Math.min(0.95, Math.max(0.32, 0.44 * Math.exp(-0.13 * (t - 1))))

// fine grid → normalised p(θ|y) and its CDF (for inverse-CDF sampling)
const NF = 240, dThF = (TH1 - TH0) / NF
const thF = Array.from({ length: NF + 1 }, (_, i) => TH0 + i * dThF)
const fPost = (() => {
  const raw = thF.map(splitUn); let Z = 0; raw.forEach(v => (Z += v * dThF))
  return raw.map(v => v / Z)
})()
const cdf = (() => {
  const c = [0]; for (let i = 1; i <= NF; i++) c[i] = c[i - 1] + 0.5 * (fPost[i - 1] + fPost[i]) * dThF
  const tot = c[NF]; return c.map(v => v / tot)
})()
const invCDF = (u: number) => {                    // θ such that CDF(θ)=u, linear interp
  let lo = 0, hi = NF
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (cdf[m] < u) lo = m; else hi = m }
  const span = cdf[hi] - cdf[lo] || 1
  return thF[lo] + (u - cdf[lo]) / span * dThF
}
const mulberry32 = (a: number) => () => {          // deterministic PRNG (stable across SSR)
  a |= 0; a = (a + 0x6D2B79F5) | 0
  let t = Math.imul(a ^ (a >>> 15), 1 | a)
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296
}
const NMAX = 60
const DRAWS = (() => { const r = mulberry32(0x9e3779b9); return Array.from({ length: NMAX }, () => invCDF(r())) })()

const N = ref(16)                                  // number of draws (slider)
const draws = computed(() => DRAWS.slice(0, N.value))

const NX = 132, dX = (X1 - X0) / NX
const xs = Array.from({ length: NX + 1 }, (_, i) => X0 + i * dX)
const exactMarg = xs.map(x => { let v = 0; for (let i = 0; i <= NF; i++) v += fPost[i] * npdf(x, mOf(thF[i]), sOf(thF[i])) * dThF; return v })
// marginal = average of the per-draw inner-Laplace Gaussians
const hmcMarg = computed(() => xs.map(x => { let v = 0; for (const t of draws.value) v += npdf(x, mOf(t), sOf(t)); return v / draws.value.length }))

// ── geometry ──
const W = 300, H = 196, PAD = 26, AX = 162
const thx = (t: number) => +(PAD + (t - TH0) / (TH1 - TH0) * (W - 2 * PAD)).toFixed(1)
const xx = (x: number) => +(PAD + (x - X0) / (X1 - X0) * (W - 2 * PAD)).toFixed(1)
const yy = (v: number, vmax: number) => +(AX - (v / vmax) * (AX - 14)).toFixed(1)

const thAll = Array.from({ length: 121 }, (_, i) => TH0 + i * (TH1 - TH0) / 120)
const lmaxL = Math.max(...thAll.map(splitUn)) / (() => { let Z = 0; thF.forEach(t => (Z += splitUn(t) * dThF)); return Z })() * 1.06
const fAt = (t: number) => { const i = Math.round((t - TH0) / dThF); return fPost[Math.max(0, Math.min(NF, i))] }
// one shared scale: per-draw Gaussians at full height, marginal (their average)
// naturally sits lower and wider — that lower bold curve is the honest picture.
const lmaxR = npdf(0, 0, sOf(MU)) * 1.18
const poly = (pts: number[], val: (i: number) => number, sx: (t: number) => number) =>
  'M' + pts.map((t, i) => `${sx(t)},${val(i)}`).join(' L')

const pPost = computed(() => poly(thAll, i => yy(fAt(thAll[i]), lmaxL), thx))
const pExact = computed(() => poly(xs, i => yy(exactMarg[i], lmaxR), xx))
const pMarg = computed(() => poly(xs, i => yy(hmcMarg.value[i], lmaxR), xx))
const drawCurves = computed(() => draws.value.map((t, i) => ({
  d: poly(xs, j => yy(npdf(xs[j], mOf(t), sOf(t)), lmaxR), xx),
  hi: i === 0,
})))
const drawDots = computed(() => draws.value.map((t, i) => ({ cx: thx(t), cy: yy(fAt(t), lmaxL), hi: i === 0 })))
</script>

<template>
  <figure class="hmc-fig">
    <div class="hmc-panels">
      <div class="hmc-panel">
        <div class="hmc-cap">NUTS draws θ from <code>p(θ|y)</code></div>
        <svg :viewBox="`0 0 ${W} ${H}`" class="hmc-svg">
          <line class="ax" :x1="PAD" :y1="AX" :x2="W - PAD + 6" :y2="AX" />
          <path class="truth" :d="pPost" />
          <circle v-for="(d, i) in drawDots" :key="i" :cx="d.cx" :cy="d.cy" :r="d.hi ? 4 : 2.6"
            :class="d.hi ? 'dot-hi' : 'dot'" />
          <text class="axlab" :x="W / 2" :y="H - 4">θ = log τ</text>
        </svg>
      </div>

      <div class="hmc-arrow">→</div>

      <div class="hmc-panel">
        <div class="hmc-cap">each draw runs an inner Laplace</div>
        <svg :viewBox="`0 0 ${W} ${H}`" class="hmc-svg">
          <line class="ax" :x1="PAD" :y1="AX" :x2="W - PAD + 6" :y2="AX" />
          <path v-for="(c, i) in drawCurves" :key="i" :d="c.d" :class="c.hi ? 'laplace-hi' : 'laplace'" />
          <path class="exact" :d="pExact" />
          <path class="marg" :d="pMarg" />
          <text class="axlab" :x="W / 2" :y="H - 4">xᵢ</text>
        </svg>
      </div>
    </div>

    <div class="hmc-controls">
      <label>draws
        <input type="range" min="6" max="60" step="1" v-model.number="N" />
        <span class="nval">{{ N }}</span>
      </label>
      <div class="legend">
        <span class="lg laplace-lg">inner Laplace / draw</span>
        <span class="lg marg-lg">marginal (their average)</span>
        <span class="lg exact-lg">exact</span>
      </div>
    </div>
    <p class="hmc-note">Not just NUTS: every draw solves the latent field at its own θ (one highlighted). Averaging those inner Laplace fits gives the marginal.</p>
  </figure>
</template>

<style scoped>
.hmc-fig {
  --tan: #E8D5B7; --caramel: #C9986A; --mocha: #8B6F47; --bean: #3D2817;
  --berry: #C04A2A; --good: #4F7A4A; --foam: #FFFCF7;
  margin: 22px 0; padding: 18px; background: var(--foam);
  border: 1px solid var(--tan); border-radius: 14px;
  font-family: 'Inter', system-ui, sans-serif;
}
.hmc-panels { display: flex; align-items: center; gap: 6px; }
.hmc-panel { flex: 1; min-width: 0; }
.hmc-arrow { color: var(--caramel); font-size: 22px; flex: 0 0 auto; padding: 0 2px; }
.hmc-cap { font-size: 12.5px; color: #4A3828; margin-bottom: 4px; }
.hmc-cap code { font-family: 'JetBrains Mono', monospace; font-size: 0.92em; background: rgba(139,111,71,0.1); padding: 0 4px; border-radius: 3px; }
.hmc-svg { width: 100%; height: auto; display: block; overflow: hidden; }

.ax { stroke: rgba(139,111,71,0.5); stroke-width: 1; }
.truth { fill: none; stroke: var(--mocha); stroke-width: 1.8; }
.dot { fill: var(--caramel); opacity: 0.55; }
.dot-hi { fill: var(--berry); opacity: 0.95; }
.laplace { fill: none; stroke: var(--caramel); stroke-width: 1; opacity: 0.32; }
.laplace-hi { fill: none; stroke: var(--berry); stroke-width: 1.7; opacity: 0.9; }
.marg { fill: none; stroke: var(--bean); stroke-width: 2.4; }
.exact { fill: none; stroke: var(--good); stroke-width: 1.6; stroke-dasharray: 4 3; }
.axlab { fill: var(--mocha); font-size: 9.5px; text-anchor: middle; font-family: 'JetBrains Mono', monospace; }

.hmc-controls { display: flex; flex-wrap: wrap; align-items: center; gap: 14px 26px; margin: 14px 4px 6px; }
.hmc-controls label { font-size: 13px; color: var(--bean); display: flex; align-items: center; gap: 10px; }
.hmc-controls input[type=range] { accent-color: var(--berry); width: 180px; }
.nval { font-family: 'JetBrains Mono', monospace; font-size: 12.5px; color: var(--berry); min-width: 1.6em; }
.legend { display: flex; gap: 14px; flex-wrap: wrap; }
.lg { font-size: 11.5px; color: #4A3828; display: inline-flex; align-items: center; }
.lg::before { content: ''; width: 16px; height: 0; border-top-width: 2px; border-top-style: solid; margin-right: 6px; }
.laplace-lg::before { border-top-color: var(--caramel); }
.marg-lg::before { border-top-color: var(--bean); }
.exact-lg::before { border-top-color: var(--good); border-top-style: dashed; }
.hmc-note { font-size: 11.5px; color: var(--mocha); margin: 2px 4px 0; line-height: 1.45; }

@media (max-width: 720px) {
  .hmc-panels { flex-direction: column; }
  .hmc-arrow { transform: rotate(90deg); }
}
</style>
