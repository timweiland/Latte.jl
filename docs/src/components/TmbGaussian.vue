<script setup lang="ts">
import { ref, computed } from 'vue'

// ── A toy-but-honest TMB picture ─────────────────────────────────────
// p(θ|y) is a skew-normal whose skewness you control. TMB takes ONE Gaussian at
// the mode (width from the curvature there), so at zero skew it is exact and as
// the skew grows the symmetric Gaussian cannot follow it. The latent marginal,
// built by integrating N(m(θ),s(θ)) over that Gaussian, drifts from the truth.
const XI = 1.0, OM = 1.15                         // skew-normal location, scale
const TH0 = -2.6, TH1 = 6.0                        // θ plotting range
const X0 = -2.6, X1 = 5.0                          // xᵢ plotting range
const SQRT2PI = Math.sqrt(2 * Math.PI)
const erf = (x: number) => {                       // Abramowitz–Stegun 7.1.26
  const s = x < 0 ? -1 : 1; x = Math.abs(x)
  const t = 1 / (1 + 0.3275911 * x)
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x)
  return s * y
}
const Phi = (x: number) => 0.5 * (1 + erf(x / Math.SQRT2))
const phi = (x: number) => Math.exp(-0.5 * x * x) / SQRT2PI
const npdf = (x: number, m: number, s: number) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * SQRT2PI)
const mOf = (t: number) => 0.6 * t
const sOf = (t: number) => Math.min(0.95, Math.max(0.32, 0.44 * Math.exp(-0.13 * (t - 1))))

const alpha = ref(3.5)                             // skewness (slider)
const skewUn = (t: number, a: number) => (2 / OM) * phi((t - XI) / OM) * Phi(a * (t - XI) / OM)

const NF = 200, dThF = (TH1 - TH0) / NF
const thF = Array.from({ length: NF + 1 }, (_, i) => TH0 + i * dThF)

// normalised p(θ|y), its mode, and TMB's Gaussian-at-the-mode
const post = computed(() => {
  const a = alpha.value
  const raw = thF.map(t => skewUn(t, a))
  let Z = 0; raw.forEach(v => (Z += v * dThF))
  const f = raw.map(v => v / Z)
  let im = 0; for (let i = 1; i < f.length; i++) if (f[i] > f[im]) im = i
  const mode = thF[im]
  const lf = (t: number) => Math.log(Math.max(1e-12, skewUn(t, a)))
  const h = 0.06, curv = -(lf(mode + h) - 2 * lf(mode) + lf(mode - h)) / (h * h)
  const sd = Math.min(2.2, Math.max(0.18, 1 / Math.sqrt(Math.max(curv, 0.05))))
  return { f, mode, sd }
})
const gTmb = (t: number) => npdf(t, post.value.mode, post.value.sd)

const NX = 130, dX = (X1 - X0) / NX
const xs = Array.from({ length: NX + 1 }, (_, i) => X0 + i * dX)
const marg = (w: (t: number) => number) => xs.map(x => {
  let v = 0; for (let i = 0; i <= NF; i++) { const t = thF[i]; v += w(t) * npdf(x, mOf(t), sOf(t)) * dThF }
  return v
})
const fAt = (t: number) => { const i = Math.round((t - TH0) / dThF); return post.value.f[Math.max(0, Math.min(NF, i))] }
const exactMarg = computed(() => marg(fAt))
const tmbMarg = computed(() => marg(gTmb))

// ── geometry ──
const W = 300, H = 188, PAD = 26, AX = 162
const thx = (t: number) => +(PAD + (t - TH0) / (TH1 - TH0) * (W - 2 * PAD)).toFixed(1)
const xx = (x: number) => +(PAD + (x - X0) / (X1 - X0) * (W - 2 * PAD)).toFixed(1)
const yy = (v: number, vmax: number) => +(AX - (v / vmax) * (AX - 12)).toFixed(1)

const thAll = Array.from({ length: 121 }, (_, i) => TH0 + i * (TH1 - TH0) / 120)
const lmaxL = computed(() => Math.max(...thAll.map(t => Math.max(fAt(t), gTmb(t)))) * 1.04)
const lmaxR = computed(() => Math.max(...exactMarg.value, ...tmbMarg.value) * 1.04)
const poly = (pts: number[], yval: (i: number) => number, sx: (t: number) => number) =>
  'M' + pts.map((t, i) => `${sx(t)},${yval(i)}`).join(' L')

const pPost = computed(() => poly(thAll, i => yy(fAt(thAll[i]), lmaxL.value), thx))
const pGauss = computed(() => poly(thAll, i => yy(gTmb(thAll[i]), lmaxL.value), thx))
const pExact = computed(() => poly(xs, i => yy(exactMarg.value[i], lmaxR.value), xx))
const pTmb = computed(() => poly(xs, i => yy(tmbMarg.value[i], lmaxR.value), xx))
</script>

<template>
  <figure class="tmb-fig">
    <div class="tmb-panels">
      <div class="tmb-panel">
        <div class="tmb-cap">the hyperparameter posterior <code>p(θ|y)</code></div>
        <svg :viewBox="`0 0 ${W} ${H}`" class="tmb-svg">
          <line class="ax" :x1="PAD" :y1="AX" :x2="W - PAD + 6" :y2="AX" />
          <line class="mode" :x1="thx(post.mode)" :y1="12" :x2="thx(post.mode)" :y2="AX" />
          <path class="truth" :d="pPost" />
          <path class="tmb" :d="pGauss" />
          <text class="axlab" :x="W / 2" :y="H - 4">θ = log τ</text>
        </svg>
      </div>

      <div class="tmb-arrow">→</div>

      <div class="tmb-panel">
        <div class="tmb-cap">the resulting latent marginal <code>p(xᵢ|y)</code></div>
        <svg :viewBox="`0 0 ${W} ${H}`" class="tmb-svg">
          <line class="ax" :x1="PAD" :y1="AX" :x2="W - PAD + 6" :y2="AX" />
          <path class="truth" :d="pExact" />
          <path class="tmb" :d="pTmb" />
          <text class="axlab" :x="W / 2" :y="H - 4">xᵢ</text>
        </svg>
      </div>
    </div>

    <div class="tmb-controls">
      <label>skewness of p(θ|y)
        <input type="range" min="0" max="6" step="0.25" v-model.number="alpha" />
        <span class="aval">{{ alpha.toFixed(2) }}</span>
      </label>
      <div class="legend">
        <span class="lg truth-lg">truth</span>
        <span class="lg tmb-lg">TMB (Gaussian at the mode)</span>
      </div>
    </div>
  </figure>
</template>

<style scoped>
.tmb-fig {
  --tan: #E8D5B7; --caramel: #C9986A; --mocha: #8B6F47; --bean: #3D2817;
  --berry: #C04A2A; --good: #4F7A4A; --foam: #FFFCF7;
  margin: 22px 0; padding: 18px; background: var(--foam);
  border: 1px solid var(--tan); border-radius: 14px;
  font-family: 'Inter', system-ui, sans-serif;
}
.tmb-panels { display: flex; align-items: center; gap: 6px; }
.tmb-panel { flex: 1; min-width: 0; }
.tmb-arrow { color: var(--caramel); font-size: 22px; flex: 0 0 auto; padding: 0 2px; }
.tmb-cap { font-size: 12.5px; color: #4A3828; margin-bottom: 4px; }
.tmb-cap code { font-family: 'JetBrains Mono', monospace; font-size: 0.92em; background: rgba(139,111,71,0.1); padding: 0 4px; border-radius: 3px; }
.tmb-svg { width: 100%; height: auto; display: block; }

.ax { stroke: rgba(139,111,71,0.5); stroke-width: 1; }
.mode { stroke: rgba(139,111,71,0.35); stroke-width: 1; stroke-dasharray: 2 3; }
.truth { fill: none; stroke: var(--good); stroke-width: 1.8; }
.tmb { fill: none; stroke: var(--berry); stroke-width: 2.1; }
.axlab { fill: var(--mocha); font-size: 9.5px; text-anchor: middle; font-family: 'JetBrains Mono', monospace; }

.tmb-controls { display: flex; flex-wrap: wrap; align-items: center; gap: 18px 26px; margin: 14px 4px 6px; }
.tmb-controls label { font-size: 13px; color: var(--bean); display: flex; align-items: center; gap: 10px; }
.tmb-controls input[type=range] { accent-color: var(--berry); width: 180px; }
.aval { font-family: 'JetBrains Mono', monospace; font-size: 12.5px; color: var(--berry); }
.legend { display: flex; gap: 16px; flex-wrap: wrap; }
.lg { font-size: 11.5px; color: #4A3828; display: inline-flex; align-items: center; }
.lg::before { content: ''; width: 16px; height: 0; border-top-width: 2px; border-top-style: solid; margin-right: 6px; }
.truth-lg::before { border-top-color: var(--good); }
.tmb-lg::before { border-top-color: var(--berry); }

@media (max-width: 720px) {
  .tmb-panels { flex-direction: column; }
  .tmb-arrow { transform: rotate(90deg); }
}
</style>
