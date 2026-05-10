<script setup lang="ts">
import { onMounted, ref } from 'vue'

const taglines = [
  'How do you take your posterior?',
  'Same beans, three brews.',
  'Pick your pour.',
  'Slow-roasted, fast-pulled.',
  'Posteriors on tap.',
  'Latent Gaussians, three brewing methods.',
  'Bayes by the cup.',
]
const tagline = ref(taglines[0])
onMounted(() => {
  tagline.value = taglines[Math.floor(Math.random() * taglines.length)]
})
</script>

<template>
  <div class="latte-awning">
    <div class="latte-awning-inner">
      <span><span class="latte-awning-dot"></span>OPEN · EST. 2026 · <code>latte-prototype</code> · v0.1 in preparation</span>
      <span>{{ tagline }}</span>
    </div>
  </div>
</template>

<style scoped>
.latte-awning {
  background: #2A1810;
  color: #F5E6D3;
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 12px;
  letter-spacing: 0.5px;
}
.latte-awning-inner {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 48px;
  max-width: 1200px;
  margin: 0 auto;
}
.latte-awning-dot {
  display: inline-block;
  width: 6px; height: 6px;
  border-radius: 50%;
  background: #86C068;
  margin-right: 8px;
  box-shadow: 0 0 8px rgba(134,192,104,0.5);
}
.latte-awning code {
  font: inherit;
  background: rgba(245,230,211,0.1);
  padding: 1px 5px;
  border-radius: 2px;
}

@media (max-width: 720px) {
  .latte-awning-inner > span:last-child { display: none; }
}
</style>
