<script setup lang="ts">
// A clickable citation card. Use anywhere (globally registered):
//   <PaperCite
//     tag="SBC"
//     title="Validating Bayesian Inference Algorithms with Simulation-Based Calibration"
//     authors="Talts, Betancourt, Simpson, Vehtari & Gelman"
//     venue="arXiv preprint" year="2018"
//     arxiv="1804.06788"
//     url="https://arxiv.org/abs/1804.06788"
//     abstract="…one or two sentences…" />
defineProps<{
  title: string
  authors: string
  url: string
  venue?: string
  year?: string | number
  doi?: string
  arxiv?: string
  abstract?: string
  tag?: string
}>()
</script>

<template>
  <a class="paper" :href="url" target="_blank" rel="noopener noreferrer">
    <div class="paper-tag" v-if="tag">{{ tag }}</div>
    <div class="paper-title">{{ title }}<span class="paper-arrow" aria-hidden="true">↗</span></div>
    <div class="paper-authors">{{ authors }}</div>
    <div class="paper-meta">
      <span v-if="venue">{{ venue }}</span><span v-if="year"> · {{ year }}</span><span
        v-if="doi" class="paper-id"> · doi:{{ doi }}</span><span
        v-else-if="arxiv" class="paper-id"> · arXiv:{{ arxiv }}</span>
    </div>
    <p class="paper-abstract" v-if="abstract">{{ abstract }}</p>
  </a>
</template>

<style scoped>
.paper {
  --tan:      #E8D5B7;
  --caramel:  #C9986A;
  --mocha:    #8B6F47;
  --bean:     #3D2817;
  --espresso: #2A1810;
  --berry:    #C04A2A;
  --foam:     #FFFCF7;
  display: block;
  text-decoration: none;
  background: var(--foam);
  border: 1px solid var(--tan);
  border-radius: 12px;
  padding: 18px 20px;
  color: var(--espresso);
  font-family: 'Inter', system-ui, sans-serif;
  transition: border-color .15s, box-shadow .15s, transform .15s;
}
.paper:hover {
  border-color: var(--caramel);
  box-shadow: 0 6px 22px rgba(61,40,23,0.09);
  transform: translateY(-1px);
}
.paper-tag {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px; letter-spacing: 1.2px; text-transform: uppercase;
  color: var(--berry); margin-bottom: 9px;
}
.paper-title {
  font-family: 'Fraunces', Georgia, serif;
  font-weight: 500; font-size: 17px; line-height: 1.28; letter-spacing: -0.01em;
  color: var(--bean); margin-bottom: 7px;
}
.paper-arrow { font-size: 12px; color: var(--caramel); margin-left: 6px; vertical-align: 2px; }
.paper:hover .paper-arrow { color: var(--berry); }
.paper-authors { font-size: 14px; color: #4A3828; margin-bottom: 4px; }
.paper-meta { font-family: 'JetBrains Mono', monospace; font-size: 11.5px; color: var(--mocha); }
.paper-id { color: var(--caramel); }
.paper-abstract {
  font-size: 13.5px; line-height: 1.5; color: #5A4636;
  margin: 12px 0 0; padding-top: 12px; border-top: 1px solid #F0E6D8;
}
</style>
