<template>
  <footer class="latte-foot">
    <div class="container">
      <div class="foot-grid">
        <div>
          <a class="wm foot-wm" href="/">
            <svg width="32" height="32" viewBox="0 0 220 220" xmlns="http://www.w3.org/2000/svg" aria-label="Latte.jl">
              <defs>
                <radialGradient id="lattecoffee-minifoot" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stop-color="#D9B98A"/>
                  <stop offset="70%" stop-color="#C9A373"/>
                  <stop offset="100%" stop-color="#B88A5C"/>
                </radialGradient>
              </defs>
              <circle cx="110" cy="110" r="104" fill="#E8D9BE"/>
              <circle cx="110" cy="110" r="84" fill="#6E4A2C"/>
              <circle cx="110" cy="110" r="80" fill="#B88A5C"/>
              <circle cx="110" cy="110" r="78" fill="url(#lattecoffee-minifoot)"/>
              <path d="M110 48 C90 52 60 90 58 128 C56 162 85 178 110 178 C135 178 164 162 162 128 C160 90 130 52 110 48 Z" fill="#FFFFFF" fill-opacity="0.38"/>
              <path d="M110 66 C95 70 74 100 72 128 C70 152 92 166 110 166 C128 166 150 152 148 128 C146 100 125 70 110 66 Z" fill="#FFFFFF" fill-opacity="0.55"/>
              <path d="M110 84 C100 86 88 108 86 128 C84 146 98 156 110 156 C122 156 136 146 134 128 C132 108 120 86 110 84 Z" fill="#FFFFFF" fill-opacity="0.82"/>
              <path d="M110 102 C104 104 100 118 98 128 C96 140 105 146 110 146 C115 146 124 140 122 128 C120 118 116 104 110 102 Z" fill="#FFFFFF"/>
            </svg>
            <span class="wm-txt">Latte<span class="wm-jl">.jl</span></span>
          </a>
          <div class="foot-about">
            Probabilistic programming for latent Gaussian models in Julia. INLA, TMB, and HMC-Laplace, all behind one <code>@latte</code> macro.
          </div>
        </div>
        <div>
          <h5>Learn</h5>
          <a href="/main_interface">Documentation</a>
          <a href="/tutorials/">Tutorials</a>
          <a href="/reference/observation_models">API reference</a>
        </div>
        <div>
          <h5>Community</h5>
          <a href="https://github.com/timweiland/Latte.jl">GitHub</a>
          <a href="https://github.com/timweiland/Latte.jl/issues">Issue tracker</a>
          <a href="https://github.com/timweiland/Latte.jl/blob/main/CONTRIBUTING.md">Contributing</a>
        </div>
      </div>
      <div class="foot-bottom">
        <span>© 2026 · Latte.jl maintainers · MIT licensed</span>
        <span>brewed with ♥ in Julia</span>
      </div>
    </div>
  </footer>
</template>

<style scoped>
.latte-foot {
  background: #2A1810;
  color: #F5E6D3;
  padding: 56px 0 36px;
  font-family: 'Inter', system-ui, sans-serif;
  /* Sit above Vitepress's fixed sidebar + sticky aside so neither can
   * paint over the footer when the user scrolls all the way down.
   * `position: relative` is what lets z-index take effect. Value chosen
   * comfortably above any Vitepress default and below LatteNav (1000). */
  position: relative;
  z-index: 100;
}
.container { max-width: 1200px; margin: 0 auto; padding: 0 48px; }

.foot-grid { display: grid; grid-template-columns: 1.5fr 1fr 1fr; gap: 48px; }
.foot-grid h5 { font-family: 'Fraunces', serif; font-style: italic; font-weight: 500; font-size: 16px; color: #C9986A; margin: 0 0 12px; }
.foot-grid a { display: block; color: #F5E6D3; text-decoration: none; font-size: 14px; padding: 4px 0; opacity: 0.85; }
.foot-grid a:hover { opacity: 1; color: #C9986A; }
.foot-about { font-size: 14px; line-height: 1.55; color: #D4B896; max-width: 320px; margin-top: 8px; }
.foot-bottom { display: flex; justify-content: space-between; margin-top: 48px; padding-top: 22px; border-top: 1px dashed rgba(201,152,106,0.25); font-size: 12px; color: #9B8268; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.3px; }

footer .wm.foot-wm { display: inline-flex; align-items: center; padding: 0; opacity: 1; }
.wm { display: inline-flex; align-items: center; gap: 12px; text-decoration: none; }
.wm .wm-txt { font-family: 'Fraunces', serif; font-style: italic; font-weight: 500; font-size: 26px; color: #F5E6D3; letter-spacing: -0.4px; line-height: 1; }
.wm .wm-jl { font-family: 'JetBrains Mono', monospace; font-style: normal; font-size: 20px; color: #C9986A; font-weight: 500; }

@media (max-width: 1024px) {
  .foot-grid { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 640px) {
  .foot-grid { grid-template-columns: 1fr; }
}
</style>
