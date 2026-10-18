<script setup lang="ts">
// EU emblem per the official geometry (EU visual identity guidelines):
// 3:2 rectangle, 12 five-pointed stars on a circle of radius h/3, each
// star's outer radius h/18, one point straight up. Drawn on a 810×540 box.
const W = 810, H = 540, CX = W / 2, CY = H / 2
const RING = H / 3, R = H / 18, r = R * 0.381966

function star(cx: number, cy: number): string {
  const pts: string[] = []
  for (let k = 0; k < 10; k++) {
    const a = (-90 + 36 * k) * Math.PI / 180
    const rad = k % 2 === 0 ? R : r
    pts.push(`${(cx + rad * Math.cos(a)).toFixed(2)},${(cy + rad * Math.sin(a)).toFixed(2)}`)
  }
  return `M${pts.join('L')}Z`
}

const stars = Array.from({ length: 12 }, (_, i) => {
  const t = i * Math.PI / 6
  return star(CX + RING * Math.sin(t), CY - RING * Math.cos(t))
})
</script>

<template>
  <div class="eu-funding">
    <svg class="eu-flag" :viewBox="`0 0 ${W} ${H}`" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Flag of the European Union">
      <rect :width="W" :height="H" fill="#003399"/>
      <path v-for="(d, i) in stars" :key="i" :d="d" fill="#FFCC00"/>
    </svg>
    <p class="eu-text">
      Built with financial support of the European Union through
      ERC Consolidator Grant
      <a href="https://cordis.europa.eu/project/id/101123955" target="_blank" rel="noopener">ANUBIS (101123955)</a>.
    </p>
  </div>
</template>

<style scoped>
.eu-funding {
  display: flex;
  align-items: center;
  gap: 18px;
  margin-top: 44px;
}
/* On a non-white background the EU guidelines ask for a white border of
 * 1/25 of the flag's height around the emblem. */
.eu-flag {
  flex: none;
  width: 54px;
  height: 36px;
  display: block;
  box-sizing: content-box;
  border: 1.5px solid #FFFFFF;
  border-radius: 2px;
}
.eu-text {
  margin: 0;
  max-width: 560px;
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 13px;
  line-height: 1.5;
  color: #D4B896;
}
.eu-text a {
  color: #F5E6D3;
  text-decoration: underline;
  text-decoration-color: rgba(201,152,106,0.5);
  text-underline-offset: 3px;
  white-space: nowrap;
}
.eu-text a:hover { color: #C9986A; text-decoration-color: #C9986A; }

@media (max-width: 640px) {
  .eu-funding { align-items: flex-start; gap: 14px; }
  .eu-flag { width: 45px; height: 30px; margin-top: 2px; }
}
</style>
