<script setup lang="ts">
import { onMounted, onUnmounted, inject } from 'vue'
import { useData } from 'vitepress'

// Appearance toggle. The default Vitepress nav (which carries the switch) is
// hidden in style.css, so we surface our own here. Uses the same mechanism as
// Vitepress's VPSwitchAppearance: the injected `toggle-appearance` runs the
// view-transition animation when present, falling back to flipping `isDark`.
const { isDark } = useData()
const toggleAppearance = inject('toggle-appearance', () => { isDark.value = !isDark.value })

// LatteNav is `position: sticky; top: 0`, so its height is constant
// while it's pinned. We publish that height as `--latte-nav-height`
// once on mount and whenever the nav resizes, and style.css uses it as
// the `top` offset for Vitepress's fixed sidebar + aside so they don't
// paint over the nav.
let resizeObserver: ResizeObserver | null = null

function publishNavHeight() {
  const navEl = document.querySelector('.latte-nav-wrap') as HTMLElement | null
  if (!navEl) return
  const h = navEl.getBoundingClientRect().height
  document.documentElement.style.setProperty('--latte-nav-height', `${h}px`)
}

onMounted(() => {
  publishNavHeight()
  const navEl = document.querySelector('.latte-nav-wrap') as HTMLElement | null
  if (navEl && typeof ResizeObserver !== 'undefined') {
    resizeObserver = new ResizeObserver(publishNavHeight)
    resizeObserver.observe(navEl)
  }
})

onUnmounted(() => {
  if (resizeObserver) resizeObserver.disconnect()
})
</script>

<template>
  <div class="latte-nav-wrap">
    <!-- Nav -->
    <nav class="top">
      <div class="container">
        <a class="wm" href="/">
          <svg width="36" height="36" viewBox="0 0 220 220" xmlns="http://www.w3.org/2000/svg" aria-label="Latte.jl">
            <defs>
              <radialGradient id="lattecoffee-mininav" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stop-color="#D9B98A"/>
                <stop offset="70%" stop-color="#C9A373"/>
                <stop offset="100%" stop-color="#B88A5C"/>
              </radialGradient>
            </defs>
            <circle cx="110" cy="110" r="104" fill="#E8D9BE"/>
            <circle cx="110" cy="110" r="104" fill="none" stroke="#8B6F47" stroke-opacity="0.2" stroke-width="1"/>
            <circle cx="110" cy="110" r="84" fill="#6E4A2C"/>
            <circle cx="110" cy="110" r="80" fill="#B88A5C"/>
            <circle cx="110" cy="110" r="78" fill="url(#lattecoffee-mininav)"/>
            <path d="M110 48 C90 52 60 90 58 128 C56 162 85 178 110 178 C135 178 164 162 162 128 C160 90 130 52 110 48 Z" fill="#FFFFFF" fill-opacity="0.38"/>
            <path d="M110 66 C95 70 74 100 72 128 C70 152 92 166 110 166 C128 166 150 152 148 128 C146 100 125 70 110 66 Z" fill="#FFFFFF" fill-opacity="0.55"/>
            <path d="M110 84 C100 86 88 108 86 128 C84 146 98 156 110 156 C122 156 136 146 134 128 C132 108 120 86 110 84 Z" fill="#FFFFFF" fill-opacity="0.82"/>
            <path d="M110 102 C104 104 100 118 98 128 C96 140 105 146 110 146 C115 146 124 140 122 128 C120 118 116 104 110 102 Z" fill="#FFFFFF"/>
          </svg>
          <span class="wm-txt">Latte<span class="wm-jl">.jl</span></span>
        </a>
        <div class="links">
          <div class="nav-item nav-item-dropdown">
            <a href="/reference/" class="nav-link">Docs <span class="caret">▾</span></a>
            <div class="dropdown">
              <a href="/reference/latte">Defining models</a>
              <a href="/reference/results">Working with results</a>
              <a href="/reference/">API reference</a>
            </div>
          </div>
          <a class="nav-link" href="/tutorials/">Tutorials</a>
          <a class="nav-link" href="/benchmarks/">Benchmarks</a>
          <a class="nav-link" href="/validation/">Validation</a>
          <a class="nav-link" href="https://github.com/timweiland/Latte.jl">GitHub</a>
          <span class="ver">v0.1-dev</span>
          <button class="theme-toggle" type="button" @click="toggleAppearance"
                  aria-label="Toggle light/dark theme" title="Toggle light/dark theme">
            <svg class="icon-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
            <svg class="icon-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
          </button>
        </div>
      </div>
    </nav>
  </div>
</template>

<style scoped>
.latte-nav-wrap {
  --bg:       #FAF7F2;
  --cream:    #F5E6D3;
  --tan:      #E8D5B7;
  --caramel:  #C9986A;
  --mocha:    #8B6F47;
  --bean:     #3D2817;
  --espresso: #2A1810;
  --berry:    #C04A2A;
  --foam:     #FFFCF7;
  background: var(--bg);
  color: var(--espresso);
  font-family: 'Inter', system-ui, sans-serif;
  position: sticky;
  top: 0;
  z-index: 1000;
}

.container { max-width: 1200px; margin: 0 auto; padding: 0 48px; }

nav.top { padding: 22px 0; background: var(--bg); }
nav.top .container { display: flex; align-items: center; justify-content: space-between; }
.links { display: flex; gap: 28px; font-size: 14.5px; color: #4A3828; align-items: center; }
.links a { text-decoration: none; transition: color .15s; color: #4A3828; }
.links a:hover { color: var(--berry); }
.ver { background: var(--bean); color: var(--cream); padding: 5px 10px; border-radius: 3px; font-family: 'JetBrains Mono', monospace; font-size: 11.5px; letter-spacing: 0.3px; }

/* Hover dropdown for "Docs" (CSS-only; keyboard focus-within keeps it
 * available for tabbing through). */
.nav-item-dropdown { position: relative; }
.nav-item-dropdown .nav-link .caret {
  font-size: 0.75em;
  margin-left: 2px;
  color: var(--mocha);
  transition: transform .12s;
  display: inline-block;
}
.nav-item-dropdown:hover .nav-link .caret,
.nav-item-dropdown:focus-within .nav-link .caret { transform: rotate(180deg); }

.dropdown {
  display: none;
  position: absolute;
  top: 100%;
  left: -8px;
  background: var(--foam);
  border: 1px solid var(--tan);
  padding: 6px 0;
  min-width: 180px;
  z-index: 100;
  box-shadow: 0 12px 32px rgba(42,24,16,0.10);
  margin-top: 6px;
}
/* Invisible bridge: extends the dropdown's hit area up into the gap
 * between trigger and dropdown so the cursor can travel between them
 * without losing :hover. */
.dropdown::before {
  content: '';
  position: absolute;
  top: -6px;
  left: 0;
  right: 0;
  height: 6px;
}
.nav-item-dropdown:hover .dropdown,
.nav-item-dropdown:focus-within .dropdown { display: block; }

.dropdown a {
  display: block;
  padding: 8px 16px;
  font-size: 14px;
  color: var(--espresso);
  white-space: nowrap;
}
.dropdown a:hover { background: var(--cream); color: var(--berry); }

.wm { display: inline-flex; align-items: center; gap: 12px; text-decoration: none; }
.wm .wm-txt { font-family: 'Fraunces', serif; font-style: italic; font-weight: 500; font-size: 26px; color: var(--espresso); letter-spacing: -0.4px; line-height: 1; }
.wm .wm-jl { font-family: 'JetBrains Mono', monospace; font-style: normal; font-size: 20px; color: var(--caramel); font-weight: 500; }

/* Light/dark toggle. Both icons live in the DOM; CSS shows the one that
 * matches the active theme (driven by the global `.dark` class), so there's
 * no hydration mismatch from rendering based on JS state. */
.theme-toggle {
  display: inline-flex; align-items: center; justify-content: center;
  width: 32px; height: 32px; padding: 0;
  background: transparent; border: 1px solid var(--tan); border-radius: 6px;
  color: #4A3828; cursor: pointer; transition: color .15s, border-color .15s;
}
.theme-toggle:hover { border-color: var(--caramel); color: var(--berry); }
.theme-toggle svg { width: 17px; height: 17px; display: block; }
.theme-toggle .icon-sun { display: none; }
</style>

<!-- Dark-mode overrides in a NON-scoped block. Vue's scoped `:global()` is
     dropped by this project's CSS pipeline, so we use plain selectors instead.
     Every rule is nested under the unique `.latte-nav-wrap` class (no leakage),
     prefixed with `html.dark` to outrank the scoped light defaults, and reuses
     the local palette vars defined on `.latte-nav-wrap` above. -->
<style>
html.dark .latte-nav-wrap { background: var(--espresso); color: var(--cream); border-bottom: 1px solid rgba(201, 152, 106, 0.18); }
html.dark .latte-nav-wrap nav.top { background: var(--espresso); }
html.dark .latte-nav-wrap .links,
html.dark .latte-nav-wrap .links a { color: var(--tan); }
html.dark .latte-nav-wrap .links a:hover { color: var(--caramel); }
html.dark .latte-nav-wrap .wm .wm-txt { color: var(--cream); }
html.dark .latte-nav-wrap .nav-item-dropdown .caret { color: var(--caramel); }
html.dark .latte-nav-wrap .dropdown { background: #38241B; border-color: rgba(201, 152, 106, 0.25); box-shadow: 0 12px 32px rgba(0, 0, 0, 0.45); }
html.dark .latte-nav-wrap .dropdown a { color: var(--cream); }
html.dark .latte-nav-wrap .dropdown a:hover { background: rgba(201, 152, 106, 0.15); color: var(--caramel); }
html.dark .latte-nav-wrap .ver { background: rgba(201, 152, 106, 0.18); color: var(--cream); }
html.dark .latte-nav-wrap .theme-toggle { border-color: rgba(201, 152, 106, 0.3); color: var(--tan); }
html.dark .latte-nav-wrap .theme-toggle:hover { border-color: var(--caramel); color: var(--caramel); }
html.dark .latte-nav-wrap .theme-toggle .icon-sun { display: block; }
html.dark .latte-nav-wrap .theme-toggle .icon-moon { display: none; }
</style>
