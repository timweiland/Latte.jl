<script setup lang="ts">
// Floating control for the wide doc layout (pages with `pageClass: latte-wide`).
// On those pages the left nav sidebar is hidden by default (see style.css) so
// the body reclaims its width; this button slides it back in as an overlay
// drawer and a scrim closes it. State is published as the `latte-sidebar-open`
// class on <html>, which the CSS keys off. Non-wide pages render nothing.
import { computed, ref, watch, onUnmounted } from 'vue'
import { useData, useRoute } from 'vitepress'

const { frontmatter } = useData()
const route = useRoute()

const isWide = computed(() =>
  String(frontmatter.value.pageClass || '').split(/\s+/).includes('latte-wide')
)

const open = ref(false)

function sync() {
  if (typeof document === 'undefined') return
  document.documentElement.classList.toggle(
    'latte-sidebar-open',
    isWide.value && open.value,
  )
}

watch([open, isWide], sync, { immediate: true })

// Collapse the drawer on every navigation so it never lingers across pages.
watch(() => route.path, () => { open.value = false })

onUnmounted(() => {
  if (typeof document !== 'undefined') {
    document.documentElement.classList.remove('latte-sidebar-open')
  }
})
</script>

<template>
  <template v-if="isWide">
    <button
      class="latte-sidebar-toggle"
      :class="{ 'is-open': open }"
      type="button"
      :aria-expanded="open"
      aria-label="Toggle navigation sidebar"
      @click="open = !open"
    >
      <svg v-if="!open" viewBox="0 0 24 24" width="17" height="17" aria-hidden="true">
        <path fill="none" stroke="currentColor" stroke-width="2"
          stroke-linecap="round" stroke-linejoin="round" d="M4 6h16M4 12h16M4 18h16" />
      </svg>
      <svg v-else viewBox="0 0 24 24" width="17" height="17" aria-hidden="true">
        <path fill="none" stroke="currentColor" stroke-width="2"
          stroke-linecap="round" stroke-linejoin="round" d="M6 6l12 12M18 6L6 18" />
      </svg>
      <span class="latte-sidebar-toggle__label">{{ open ? 'Hide menu' : 'Menu' }}</span>
    </button>

    <div v-show="open" class="latte-sidebar-scrim" @click="open = false" />
  </template>
</template>

<style scoped>
.latte-sidebar-toggle {
  position: fixed;
  top: calc(var(--latte-nav-height, 0px) + 14px);
  left: 14px;
  z-index: 60;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 13px 6px 10px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 999px;
  background: var(--vp-c-bg-elv);
  color: var(--vp-c-text-1);
  font-size: 13px;
  font-weight: 600;
  line-height: 1;
  cursor: pointer;
  box-shadow: 0 1px 4px rgba(42, 24, 16, 0.12);
  transition: border-color 0.2s, color 0.2s, box-shadow 0.2s;
}
.latte-sidebar-toggle:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
  box-shadow: 0 2px 8px rgba(42, 24, 16, 0.18);
}
.latte-sidebar-toggle__label { white-space: nowrap; }

/* While the drawer is open, sit the close button just over its top-left. */
.latte-sidebar-toggle.is-open { z-index: 60; }

.latte-sidebar-scrim {
  position: fixed;
  inset: var(--latte-nav-height, 0px) 0 0 0;
  z-index: 55;
  background: rgba(42, 24, 16, 0.32);
}

@media (max-width: 640px) {
  .latte-sidebar-toggle { padding: 8px; }
  .latte-sidebar-toggle__label { display: none; }
}
</style>
