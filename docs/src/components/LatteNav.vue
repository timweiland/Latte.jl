<script setup lang="ts">
import { onMounted, onUnmounted } from 'vue'

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
            <a href="/main_interface" class="nav-link">Docs <span class="caret">▾</span></a>
            <div class="dropdown">
              <a href="/main_interface">Main Interface</a>
              <a href="/reference/observation_models">Reference</a>
            </div>
          </div>
          <a class="nav-link" href="/tutorials/">Tutorials</a>
          <a class="nav-link" href="/benchmarks/">Benchmarks</a>
          <a class="nav-link" href="/validation/">Validation</a>
          <a class="nav-link" href="https://github.com/timweiland/Latte.jl">GitHub</a>
          <span class="ver">v0.1-dev</span>
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
</style>
