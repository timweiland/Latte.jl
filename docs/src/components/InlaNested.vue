<script setup lang="ts">
import { ref, computed } from 'vue'

// ── A toy-but-honest INLA picture ────────────────────────────────────
// p(θ|y): a right-skewed hyperparameter posterior (split-normal in θ = log τ).
// Inner conditional p(xᵢ|y,θ) = N(m(θ), s(θ)) — centre AND width shift with θ,
// so mixing them over the skewed p(θ|y) yields a skewed marginal.
// INLA places K grid points, approximates the marginal as their weighted mixture.
const MU = 1.0, SL = 0.5, SR = 1.15            // split-normal: mode, left/right scale
const TH0 = -1.2, TH1 = 5.2                     // θ plotting range
const GL = -0.4, GH = 4.4                        // grid placement range (the bulk)
const X0 = -2.4, X1 = 4.4                        // xᵢ plotting range
const SQRT2PI = Math.sqrt(2 * Math.PI)

const gUn = (t: number) => { const s = t < MU ? SL : SR; const z = (t - MU) / s; return Math.exp(-0.5 * z * z) }
const mOf = (t: number) => 0.6 * t                                   // conditional mean of xᵢ
const sOf = (t: number) => Math.min(0.95, Math.max(0.32, 0.44 * Math.exp(-0.13 * (t - 1)))) // conditional sd
const npdf = (x: number, m: number, s: number) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * SQRT2PI)

// normalising constant of p(θ|y) over a fine grid
const NF = 240, dThF = (TH1 - TH0) / NF
const Zg = (() => { let z = 0; for (let i = 0; i <= NF; i++) z += gUn(TH0 + i * dThF) * dThF; return z })()
const gN = (t: number) => gUn(t) / Zg

const K = ref(5)                                 // grid points (INLA's default ≈ 5)

const NX = 140, dX = (X1 - X0) / NX
const xs = Array.from({ length: NX + 1 }, (_, i) => X0 + i * dX)

// exact marginal: fine integration over θ
const exact = computed(() => xs.map(x => {
  let v = 0
  for (let i = 0; i <= NF; i++) { const t = TH0 + i * dThF; v += gN(t) * npdf(x, mOf(t), sOf(t)) * dThF }
  return v
}))

// the K grid points, their weights, and the resulting mixture
const grid = computed(() => {
  const k = K.value, pts: { t: number; w: number }[] = []
  const d = (GH - GL) / (k - 1)
  let wsum = 0
  for (let j = 0; j < k; j++) { const t = GL + j * d; const w = gN(t) * d; pts.push({ t, w }); wsum += w }
  pts.forEach(p => (p.w /= wsum))               // renormalise so the mixture is a proper density
  return pts
})
const components = computed(() => grid.value.map(p => xs.map(x => p.w * npdf(x, mOf(p.t), sOf(p.t)))))
const inla = computed(() => xs.map((_, i) => components.value.reduce((a, c) => a + c[i], 0)))

// ── plotting geometry ──
const W = 300, H = 188, PAD = 26, AX = 162
const thx = (t: number) => PAD + (t - TH0) / (TH1 - TH0) * (W - 2 * PAD)
const xx = (x: number) => PAD + (x - X0) / (X1 - X0) * (W - 2 * PAD)
const yy = (v: number, vmax: number) => AX - (v / vmax) * (AX - 12)
const fmt = (n: number) => +n.toFixed(1)

const gThetas = Array.from({ length: 121 }, (_, i) => TH0 + i * (TH1 - TH0) / 120)
const gmax = computed(() => Math.max(...gThetas.map(gN)))
const ymaxR = computed(() => Math.max(...exact.value, ...inla.value) * 1.04)

const pathTheta = computed(() => 'M' + gThetas.map(t => `${fmt(thx(t))},${fmt(yy(gN(t), gmax.value))}`).join(' L'))
const pathExact = computed(() => 'M' + xs.map((x, i) => `${fmt(xx(x))},${fmt(yy(exact.value[i], ymaxR.value))}`).join(' L'))
const pathInla = computed(() => 'M' + xs.map((x, i) => `${fmt(xx(x))},${fmt(yy(inla.value[i], ymaxR.value))}`).join(' L'))
const pathComp = (c: number[]) => 'M' + xs.map((x, i) => `${fmt(xx(x))},${fmt(yy(c[i], ymaxR.value))}`).join(' L')
</script>

<template>
  <figure class="inla-fig">
    <div class="inla-panels">
      <div class="inla-panel">
        <div class="inla-cap">the hyperparameter posterior <code>p(θ|y)</code></div>
        <svg :viewBox="`0 0 ${W} ${H}`" class="inla-svg">
          <line class="ax" :x1="PAD" :y1="AX" :x2="W - PAD + 6" :y2="AX" />
          <line class="mode" :x1="thx(MU)" :y1="12" :x2="thx(MU)" :y2="AX" />
          <path class="curve-th" :d="pathTheta" />
          <g v-for="(p, j) in grid" :key="j">
            <line class="stem" :x1="thx(p.t)" :y1="AX" :x2="thx(p.t)" :y2="yy(gN(p.t), gmax)" />
            <circle class="gdot" :cx="thx(p.t)" :cy="yy(gN(p.t), gmax)" r="3" />
          </g>
          <text class="axlab" :x="W / 2" :y="H - 4">θ = log τ &nbsp;·&nbsp; {{ grid.length }} grid points</text>
        </svg>
      </div>

      <div class="inla-arrow">→</div>

      <div class="inla-panel">
        <div class="inla-cap">a latent marginal <code>p(xᵢ|y)</code> = weighted mix of per-θ Gaussians</div>
        <svg :viewBox="`0 0 ${W} ${H}`" class="inla-svg">
          <line class="ax" :x1="PAD" :y1="AX" :x2="W - PAD + 6" :y2="AX" />
          <path v-for="(c, j) in components" :key="j" class="comp" :d="pathComp(c)" />
          <path class="exact" :d="pathExact" />
          <path class="inla" :d="pathInla" />
          <text class="axlab" :x="W / 2" :y="H - 4">xᵢ</text>
        </svg>
      </div>
    </div>

    <div class="inla-controls">
      <label>grid density
        <input type="range" min="3" max="25" step="2" v-model.number="K" />
        <span class="kval">{{ K }} points</span>
      </label>
      <div class="legend">
        <span class="lg comp-lg">per-θ Gaussian</span>
        <span class="lg inla-lg">INLA marginal</span>
        <span class="lg exact-lg">exact (fine grid)</span>
      </div>
    </div>

    <figcaption>
      INLA places a few points in hyperparameter space (left), approximates the
      latent field by a Gaussian at each, and sums them — weighted by
      <code>p(θ|y)</code> — into every marginal (right). Drag the slider: at the
      default ~5 points the mixture is lumpy and misses the exact curve; densify it
      (a smaller <code>integration_step_z</code>) and it snaps to the truth. That is
      the whole method — and the whole tuning story — in one picture.
    </figcaption>
  </figure>
</template>

<style scoped>
.inla-fig {
  --tan: #E8D5B7; --caramel: #C9986A; --mocha: #8B6F47; --bean: #3D2817;
  --berry: #C04A2A; --good: #4F7A4A; --foam: #FFFCF7;
  margin: 22px 0; padding: 18px; background: var(--foam);
  border: 1px solid var(--tan); border-radius: 14px;
  font-family: 'Inter', system-ui, sans-serif;
}
.inla-panels { display: flex; align-items: center; gap: 6px; }
.inla-panel { flex: 1; min-width: 0; }
.inla-arrow { color: var(--caramel); font-size: 22px; flex: 0 0 auto; padding: 0 2px; }
.inla-cap { font-size: 12.5px; color: #4A3828; margin-bottom: 4px; min-height: 32px; }
.inla-cap code { font-family: 'JetBrains Mono', monospace; font-size: 0.92em; background: rgba(139,111,71,0.1); padding: 0 4px; border-radius: 3px; }
.inla-svg { width: 100%; height: auto; display: block; }

.ax { stroke: rgba(139,111,71,0.5); stroke-width: 1; }
.mode { stroke: rgba(139,111,71,0.35); stroke-width: 1; stroke-dasharray: 2 3; }
.curve-th { fill: none; stroke: var(--mocha); stroke-width: 1.8; }
.stem { stroke: var(--caramel); stroke-width: 1; }
.gdot { fill: var(--berry); }
.comp { fill: none; stroke: rgba(139,111,71,0.4); stroke-width: 0.8; }
.exact { fill: none; stroke: var(--good); stroke-width: 1.6; stroke-dasharray: 4 3; }
.inla { fill: none; stroke: var(--berry); stroke-width: 2.1; }
.axlab { fill: var(--mocha); font-size: 9.5px; text-anchor: middle; font-family: 'JetBrains Mono', monospace; }

.inla-controls { display: flex; flex-wrap: wrap; align-items: center; gap: 18px 26px; margin: 14px 4px 6px; }
.inla-controls label { font-size: 13px; color: var(--bean); display: flex; align-items: center; gap: 10px; }
.inla-controls input[type=range] { accent-color: var(--berry); width: 180px; }
.kval { font-family: 'JetBrains Mono', monospace; font-size: 12.5px; color: var(--berry); }
.legend { display: flex; gap: 16px; flex-wrap: wrap; }
.lg { font-size: 11.5px; color: #4A3828; display: inline-flex; align-items: center; }
.lg::before { content: ''; width: 16px; height: 0; border-top-width: 2px; border-top-style: solid; margin-right: 6px; }
.comp-lg::before { border-top-color: var(--caramel); }
.inla-lg::before { border-top-color: var(--berry); }
.exact-lg::before { border-top-color: var(--good); border-top-style: dashed; }
figcaption { font-size: 13px; line-height: 1.55; color: #5A4636; margin-top: 8px; padding-top: 12px; border-top: 1px solid #F0E6D8; }
figcaption code { font-family: 'JetBrains Mono', monospace; font-size: 0.9em; background: rgba(139,111,71,0.1); padding: 0 4px; border-radius: 3px; }

@media (max-width: 720px) {
  .inla-panels { flex-direction: column; }
  .inla-arrow { transform: rotate(90deg); }
}
</style>
