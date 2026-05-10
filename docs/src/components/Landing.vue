<script setup lang="ts">
import { onMounted, ref } from 'vue'

// Random awning tagline on every load.
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
  <div class="latte-landing">

    <!-- Awning -->
    <div class="awning">
      <div class="awning-inner">
        <span><span class="dot"></span>OPEN · EST. 2026 · <code>latte-prototype</code> · v0.1 in preparation</span>
        <span>{{ tagline }}</span>
      </div>
    </div>

    <!-- Nav -->
    <nav class="top">
      <div class="container">
        <a class="wm" href="/">
          <svg width="36" height="36" viewBox="0 0 220 220" xmlns="http://www.w3.org/2000/svg" aria-label="Latte.jl">
            <defs>
              <radialGradient id="lattecoffee-nav" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stop-color="#D9B98A"/>
                <stop offset="70%" stop-color="#C9A373"/>
                <stop offset="100%" stop-color="#B88A5C"/>
              </radialGradient>
            </defs>
            <circle cx="110" cy="110" r="104" fill="#E8D9BE"/>
            <circle cx="110" cy="110" r="104" fill="none" stroke="#8B6F47" stroke-opacity="0.2" stroke-width="1"/>
            <circle cx="110" cy="110" r="84" fill="#6E4A2C"/>
            <circle cx="110" cy="110" r="80" fill="#B88A5C"/>
            <circle cx="110" cy="110" r="78" fill="url(#lattecoffee-nav)"/>
            <path d="M110 48 C90 52 60 90 58 128 C56 162 85 178 110 178 C135 178 164 162 162 128 C160 90 130 52 110 48 Z" fill="#FFFFFF" fill-opacity="0.38"/>
            <path d="M110 66 C95 70 74 100 72 128 C70 152 92 166 110 166 C128 166 150 152 148 128 C146 100 125 70 110 66 Z" fill="#FFFFFF" fill-opacity="0.55"/>
            <path d="M110 84 C100 86 88 108 86 128 C84 146 98 156 110 156 C122 156 136 146 134 128 C132 108 120 86 110 84 Z" fill="#FFFFFF" fill-opacity="0.82"/>
            <path d="M110 102 C104 104 100 118 98 128 C96 140 105 146 110 146 C115 146 124 140 122 128 C120 118 116 104 110 102 Z" fill="#FFFFFF"/>
          </svg>
          <span class="wm-txt">Latte<span class="wm-jl">.jl</span></span>
        </a>
        <div class="links">
          <a href="/main_interface">Docs</a>
          <a href="/tutorials/">Tutorials</a>
          <a href="/benchmarks/">Benchmarks</a>
          <a href="https://github.com/timweiland/Latte.jl">GitHub</a>
          <span class="ver">v0.1-dev</span>
        </div>
      </div>
    </nav>

    <!-- Hero: pitch + code -->
    <section class="hero">
      <svg class="hero-mark" width="640" height="640" viewBox="0 0 220 220" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <g fill="none" stroke="#2A1810" stroke-width="1.4">
          <circle cx="110" cy="110" r="104"/>
          <circle cx="110" cy="110" r="80"/>
          <path d="M110 48 C90 52 60 90 58 128 C56 162 85 178 110 178 C135 178 164 162 162 128 C160 90 130 52 110 48 Z"/>
          <path d="M110 66 C95 70 74 100 72 128 C70 152 92 166 110 166 C128 166 150 152 148 128 C146 100 125 70 110 66 Z"/>
          <path d="M110 84 C100 86 88 108 86 128 C84 146 98 156 110 156 C122 156 136 146 134 128 C132 108 120 86 110 84 Z"/>
          <path d="M110 102 C104 104 100 118 98 128 C96 140 105 146 110 146 C115 146 124 140 122 128 C120 118 116 104 110 102 Z"/>
        </g>
      </svg>

      <div class="container">
        <div class="hero-grid">
          <div>
            <h1>Latent Gaussians,<br/><em>simply brewed.</em></h1>
            <p class="lede">
              Probabilistic programming for latent Gaussian models in Julia.
            </p>
            <div class="cta-row">
              <a class="btn btn-primary" href="/main_interface">] add Latte</a>
              <a class="btn btn-ghost" href="/main_interface">Read the docs →</a>
              <span class="install">Julia ≥ 1.10</span>
            </div>
          </div>

          <div>
            <div class="code-window">
              <div class="code-chrome">
                <div class="tl" style="background: #E37A5A"></div>
                <div class="tl" style="background: #C9986A"></div>
                <div class="tl" style="background: #86C068"></div>
                <div class="fn">disease_map.jl</div>
              </div>
<pre class="code"><span class="c-comment"># Disease mapping with a Besag ICAR prior</span>
<span class="c-sym">@model</span> <span class="c-kw">function</span> <span class="c-fn">disease</span>(<span class="c-var">y</span>, <span class="c-var">E</span>, <span class="c-var">X</span>, <span class="c-var">W</span>)
  <span class="c-kw">β</span> ~ <span class="c-fn">MvNormal</span>(<span class="c-fn">zeros</span>(<span class="c-str">2</span>), <span class="c-str">100.0</span> * <span class="c-var">I</span>)
  <span class="c-kw">τ</span> ~ <span class="c-fn">PCPrior</span>.<span class="c-fn">Precision</span>(<span class="c-str">1.0</span>, α = <span class="c-str">0.01</span>)
  <span class="c-kw">u</span> ~ <span class="c-fn">BesagModel</span>(<span class="c-var">W</span>)(τ = <span class="c-kw">τ</span>)
  <span class="c-kw">for</span> <span class="c-var">i</span> <span class="c-kw">in</span> <span class="c-fn">eachindex</span>(<span class="c-var">y</span>)
    <span class="c-var">y</span>[<span class="c-var">i</span>] ~ <span class="c-fn">Poisson</span>(<span class="c-var">E</span>[<span class="c-var">i</span>] * <span class="c-fn">exp</span>(<span class="c-var">X</span>[<span class="c-var">i</span>,:]⋅<span class="c-kw">β</span> + <span class="c-kw">u</span>[<span class="c-var">i</span>]))
  <span class="c-kw">end</span>
<span class="c-kw">end</span>

<span class="c-var">fit</span> = <span class="c-fn">brew</span>(<span class="c-fn">disease</span>(<span class="c-var">y</span>, <span class="c-var">E</span>, <span class="c-var">X</span>, <span class="c-var">W</span>), <span class="c-var">y</span>; random = (<span class="c-sym">:β</span>, <span class="c-sym">:u</span>))
</pre>
            </div>
          </div>
        </div>

        <div class="engines">
          <div class="engine default">
            <div class="method">INLA <span class="badge default">default</span></div>
            <div class="desc">Nested Laplace approximation with per-hyperparameter marginals.</div>
            <div class="fit">typical fit · ms – s</div>
          </div>
          <div class="engine">
            <div class="method">TMB</div>
            <div class="desc">Gaussian approximation around the hyperparameter MAP. Cheap point estimate when you want one.</div>
            <div class="fit">typical fit · ~ms</div>
          </div>
          <div class="engine">
            <div class="method">HMC-Laplace</div>
            <div class="desc">NUTS over hyperparameters with a Laplace step on the latent.</div>
            <div class="fit">typical fit · seconds – minutes</div>
          </div>
        </div>
      </div>
    </section>

    <!-- Receipt: standalone, centered -->
    <section class="receipt-section">
      <div class="container">
        <div class="receipt-wrap">
          <div class="receipt">
            <div class="stamp">PREVIEW</div>
            <div class="head">
              <div class="name">Latte.jl Café</div>
              <div class="sub">— Benchmark Receipt —</div>
              <div class="sub">numbers landing with v0.1</div>
            </div>
            <hr/>
            <div class="row"><span>Server</span><span>MBP M3 · 16GB</span></div>
            <div class="row"><span>Julia</span><span>1.11.2</span></div>
            <hr/>
            <div class="row" style="font-weight:600;"><span>Spatial GLMM (n=3k)</span><span>pending</span></div>
            <div class="row muted"><span>vs glmmTMB</span><span>—</span></div>
            <div class="row muted"><span>vs R-INLA</span><span>—</span></div>
            <br/>
            <div class="row" style="font-weight:600;"><span>Separable space-time</span><span>pending</span></div>
            <div class="row muted"><span>vs R-INLA</span><span>—</span></div>
            <br/>
            <div class="row" style="font-weight:600;"><span>Hierarchical GLMM</span><span>pending</span></div>
            <div class="row muted"><span>vs brms</span><span>—</span></div>
            <hr/>
            <div class="row total"><span>HOUSE SPEEDUP</span><span>tbd</span></div>
          </div>
        </div>
      </div>
    </section>

    <!-- Gallery -->
    <section class="gallery">
      <div class="container">
        <h2>Tutorials</h2>
        <div class="cards">
          <a class="card" href="/tutorials/disease_mapping_spatial">
            <div class="tag">SPATIAL · DISEASE MAPPING</div>
            <h4>BYM: Besag + IID</h4>
            <p>Regional disease risk smoothed with an ICAR spatial prior plus an IID residual. PC priors on both precisions, fit with INLA in a handful of lines.</p>
          </a>
          <a class="card" href="/tutorials/temporal_trend_earthquakes">
            <div class="tag">TEMPORAL · COUNTS</div>
            <h4>Earthquake intensity trends</h4>
            <p>RW1 vs RW2 priors on annual seismicity, compared by DIC, WAIC, and marginal likelihood. All three from the same <code>inla()</code> fit.</p>
          </a>
          <a class="card" href="/tutorials/spatio_temporal_separable">
            <div class="tag">SPACE-TIME · SEPARABLE</div>
            <h4>Kronecker space-time</h4>
            <p>Region-specific temporal dynamics via a <code>SeparableModel</code> Kronecker prior. Additive vs interaction-only vs full, on one DPPL spec.</p>
          </a>
        </div>
      </div>
    </section>

    <!-- Footer -->
    <footer>
      <div class="container">
        <div class="foot-grid">
          <div>
            <a class="wm foot-wm" href="/">
              <svg width="32" height="32" viewBox="0 0 220 220" xmlns="http://www.w3.org/2000/svg" aria-label="Latte.jl">
                <defs>
                  <radialGradient id="lattecoffee-foot" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stop-color="#D9B98A"/>
                    <stop offset="70%" stop-color="#C9A373"/>
                    <stop offset="100%" stop-color="#B88A5C"/>
                  </radialGradient>
                </defs>
                <circle cx="110" cy="110" r="104" fill="#E8D9BE"/>
                <circle cx="110" cy="110" r="84" fill="#6E4A2C"/>
                <circle cx="110" cy="110" r="80" fill="#B88A5C"/>
                <circle cx="110" cy="110" r="78" fill="url(#lattecoffee-foot)"/>
                <path d="M110 48 C90 52 60 90 58 128 C56 162 85 178 110 178 C135 178 164 162 162 128 C160 90 130 52 110 48 Z" fill="#FFFFFF" fill-opacity="0.38"/>
                <path d="M110 66 C95 70 74 100 72 128 C70 152 92 166 110 166 C128 166 150 152 148 128 C146 100 125 70 110 66 Z" fill="#FFFFFF" fill-opacity="0.55"/>
                <path d="M110 84 C100 86 88 108 86 128 C84 146 98 156 110 156 C122 156 136 146 134 128 C132 108 120 86 110 84 Z" fill="#FFFFFF" fill-opacity="0.82"/>
                <path d="M110 102 C104 104 100 118 98 128 C96 140 105 146 110 146 C115 146 124 140 122 128 C120 118 116 104 110 102 Z" fill="#FFFFFF"/>
              </svg>
              <span class="wm-txt">Latte<span class="wm-jl">.jl</span></span>
            </a>
            <div class="foot-about">
              A Julia package for latent Gaussian models. INLA, TMB, and HMC-Laplace, behind one DPPL <code>@model</code>.
            </div>
          </div>
          <div>
            <h5>Learn</h5>
            <a href="/main_interface">Documentation</a>
            <a href="/tutorials/">Tutorials</a>
            <a href="/main_interface">API reference</a>
            <a href="#">Migration from R-INLA</a>
          </div>
          <div>
            <h5>Community</h5>
            <a href="https://github.com/timweiland/Latte.jl">GitHub</a>
            <a href="https://discourse.julialang.org/">Julia Discourse</a>
            <a href="https://github.com/timweiland/Latte.jl/issues">Issue tracker</a>
            <a href="#">Contributing</a>
          </div>
          <div>
            <h5>Cite</h5>
            <a href="#">Paper (in prep)</a>
            <a href="#">CITATION.bib</a>
            <a href="#">Acknowledgements</a>
          </div>
        </div>
        <div class="foot-bottom">
          <span>© 2026 · Latte.jl maintainers · MIT licensed</span>
          <span>brewed with ♥ in Julia</span>
        </div>
      </div>
    </footer>

  </div>
</template>

<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;0,9..144,600;0,9..144,700;1,9..144,400;1,9..144,500;1,9..144,600;1,9..144,700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
</style>

<style scoped>
.latte-landing {
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
  -webkit-font-smoothing: antialiased;
  margin: 0;
  padding: 0;
}
.latte-landing * { box-sizing: border-box; }
.latte-landing a { color: inherit; }
.latte-landing ::selection { background: var(--caramel); color: var(--espresso); }

.container { max-width: 1200px; margin: 0 auto; padding: 0 48px; }

/* ── Awning strip ── */
.awning { background: var(--espresso); color: var(--cream); font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 12px; letter-spacing: 0.5px; }
.awning-inner { display: flex; justify-content: space-between; align-items: center; padding: 10px 48px; max-width: 1200px; margin: 0 auto; }
.awning .dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #86C068; margin-right: 8px; box-shadow: 0 0 8px rgba(134,192,104,0.5); }
.awning code { font: inherit; background: rgba(245,230,211,0.1); padding: 1px 5px; border-radius: 2px; }

/* ── Nav ── */
nav.top { padding: 22px 0; }
nav.top .container { display: flex; align-items: center; justify-content: space-between; }
.links { display: flex; gap: 28px; font-size: 14.5px; color: #4A3828; align-items: center; }
.links a { text-decoration: none; transition: color .15s; }
.links a:hover { color: var(--berry); }
.ver { background: var(--bean); color: var(--cream); padding: 5px 10px; border-radius: 3px; font-family: 'JetBrains Mono', monospace; font-size: 11.5px; letter-spacing: 0.3px; }

/* ── Hero ── */
.hero { position: relative; padding: 32px 0 80px; overflow: hidden; }
.hero-mark { position: absolute; top: -120px; right: -160px; opacity: 0.09; pointer-events: none; }
.hero-grid { display: grid; grid-template-columns: 1.2fr 1fr; gap: 56px; align-items: center; }
.hero h1 { font-family: 'Fraunces', Georgia, serif; font-weight: 400; font-size: clamp(42px, 5.4vw, 72px); line-height: 0.98; letter-spacing: -0.045em; margin: 0; }
.hero h1 em { font-style: italic; color: var(--bean); font-weight: 400; }
.hero .lede { font-size: 18.5px; line-height: 1.55; color: #4A3828; margin: 26px 0 0; max-width: 540px; }
.hero .lede code { background: var(--tan); padding: 2px 7px; border-radius: 3px; font-family: 'JetBrains Mono', monospace; font-size: 15.5px; }
.cta-row { display: flex; gap: 12px; margin-top: 32px; align-items: center; flex-wrap: wrap; }
.btn { border: none; font-family: inherit; font-size: 15px; font-weight: 500; padding: 14px 22px; cursor: pointer; transition: transform .12s, box-shadow .12s; text-decoration: none; display: inline-block; }
/* Two-class selectors so we beat `.latte-landing a { color: inherit }`
 * on specificity — otherwise the inherit rule wins and the button text
 * disappears against the espresso button background. */
.btn.btn-primary { background: var(--espresso); color: var(--cream); font-family: 'JetBrains Mono', monospace; font-size: 14px; }
.btn.btn-primary:hover { transform: translateY(-1px); box-shadow: 0 8px 24px rgba(42,24,16,0.18); }
.btn.btn-ghost { background: transparent; color: var(--espresso); border: 1.5px solid var(--bean); }
.btn.btn-ghost:hover { background: var(--bean); color: var(--cream); }
.install { font-family: 'JetBrains Mono', monospace; font-size: 12.5px; color: var(--mocha); margin-left: 4px; }

.engines { margin-top: 64px; display: grid; grid-template-columns: repeat(3, 1fr); border: 1px solid var(--tan); background: var(--foam); }
.engine { padding: 22px 24px; border-right: 1px solid var(--tan); position: relative; }
.engine:last-child { border-right: none; }
.engine.default { background: #fff; }
.engine .method { font-family: 'Fraunces', Georgia, serif; font-style: italic; font-weight: 500; font-size: 22px; color: var(--espresso); display: flex; align-items: center; gap: 10px; margin-bottom: 8px; letter-spacing: -0.3px; }
.engine .badge { font-family: 'Inter', sans-serif; font-size: 10px; letter-spacing: 0.6px; text-transform: uppercase; padding: 2px 7px; border-radius: 2px; font-weight: 500; }
.engine .badge.default { background: var(--mocha); color: var(--cream); }
.engine .desc { font-size: 13.5px; color: var(--mocha); line-height: 1.5; }
.engine .fit { font-size: 11px; color: var(--caramel); font-family: 'JetBrains Mono', monospace; margin-top: 10px; font-variant-numeric: tabular-nums; }

/* ── Code window ── */
.code-window { background: var(--espresso); border-radius: 6px; overflow: hidden; box-shadow: 0 16px 56px rgba(42,24,16,0.14); }
.code-chrome { display: flex; align-items: center; gap: 8px; padding: 10px 14px; border-bottom: 1px solid rgba(255,255,255,0.08); }
.code-chrome .tl { width: 10px; height: 10px; border-radius: 50%; }
.code-chrome .fn { margin-left: 12px; font-family: 'JetBrains Mono', monospace; font-size: 11.5px; color: #9B8268; }
pre.code { font-family: 'JetBrains Mono', monospace; font-size: 12.5px; line-height: 1.6; margin: 0; padding: 22px 24px; color: var(--cream); overflow-x: auto; background: var(--espresso); }
.c-comment { color: #9B8268; }
.c-kw { color: #E8D5B7; }
.c-fn { color: #86C068; }
.c-sym { color: var(--berry); }
.c-str { color: #C9986A; }
.c-var { color: var(--caramel); }

/* ── Receipt section ── */
.receipt-section { padding: 96px 0; background: var(--bg); }
.receipt-wrap { max-width: 480px; margin: 0 auto; }
.receipt { background: #FFFCF7; padding: 36px 32px 24px; font-family: 'JetBrains Mono', monospace; font-size: 12.5px; color: var(--espresso); line-height: 1.75; box-shadow: 0 18px 48px rgba(42,24,16,0.1); position: relative; }
.receipt::after { content: ''; position: absolute; left: 0; right: 0; bottom: -12px; height: 14px; background-image: linear-gradient(-45deg, #FFFCF7 7px, transparent 7px), linear-gradient(45deg, #FFFCF7 7px, transparent 7px); background-size: 14px 14px; background-position: 0 0; }
.receipt .head { text-align: center; margin-bottom: 16px; }
.receipt .head .name { font-family: 'Fraunces', Georgia, serif; font-style: italic; font-size: 22px; font-weight: 500; }
.receipt .head .sub { font-size: 10px; color: var(--mocha); margin-top: 2px; letter-spacing: 0.5px; }
.receipt hr { border: none; border-top: 1px dashed var(--mocha); margin: 12px 0; }
.receipt .row { display: flex; justify-content: space-between; }
.receipt .row.muted { color: var(--mocha); padding-left: 12px; font-size: 11.5px; }
.receipt .row.total { font-weight: 700; font-size: 13px; padding-top: 4px; }
.receipt .stamp { position: absolute; top: 24px; right: 18px; transform: rotate(-12deg); border: 2px solid var(--berry); color: var(--berry); padding: 4px 10px; border-radius: 4px; font-family: 'Fraunces', serif; font-style: italic; font-size: 12px; font-weight: 600; letter-spacing: 0.5px; opacity: 0.85; }

/* ── Gallery ── */
.gallery { padding: 96px 0; background: var(--bg); }
.gallery h2 { font-family: 'Fraunces', Georgia, serif; font-style: italic; font-weight: 400; font-size: 44px; line-height: 1; letter-spacing: -1.3px; margin: 0 0 32px; }
.cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; }
.card { background: var(--foam); border: 1px solid var(--tan); padding: 26px 24px; display: flex; flex-direction: column; gap: 12px; min-height: 220px; transition: transform .15s, box-shadow .15s; cursor: pointer; text-decoration: none; color: inherit; }
.card:hover { transform: translateY(-2px); box-shadow: 0 16px 40px rgba(42,24,16,0.08); }
.card .tag { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--caramel); text-transform: uppercase; letter-spacing: 1.3px; }
.card h4 { font-family: 'Fraunces', Georgia, serif; font-style: italic; font-weight: 500; font-size: 24px; margin: 0; letter-spacing: -0.4px; line-height: 1.15; }
.card p { font-size: 14px; line-height: 1.55; color: #4A3828; margin: 0; }

/* ── Footer ── */
footer { background: var(--espresso); color: var(--cream); padding: 56px 0 36px; }
.foot-grid { display: grid; grid-template-columns: 1.5fr 1fr 1fr 1fr; gap: 48px; }
.foot-grid h5 { font-family: 'Fraunces', serif; font-style: italic; font-weight: 500; font-size: 16px; color: var(--caramel); margin: 0 0 12px; }
.foot-grid a { display: block; color: var(--cream); text-decoration: none; font-size: 14px; padding: 4px 0; opacity: 0.85; }
.foot-grid a:hover { opacity: 1; color: var(--caramel); }
.foot-about { font-size: 14px; line-height: 1.55; color: #D4B896; max-width: 320px; margin-top: 8px; }
.foot-bottom { display: flex; justify-content: space-between; margin-top: 48px; padding-top: 22px; border-top: 1px dashed rgba(201,152,106,0.25); font-size: 12px; color: #9B8268; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.3px; }
footer .wm.foot-wm { display: inline-flex; align-items: center; padding: 0; opacity: 1; }

/* Wordmark */
.wm { display: inline-flex; align-items: center; gap: 12px; text-decoration: none; }
.wm .wm-txt { font-family: 'Fraunces', serif; font-style: italic; font-weight: 500; font-size: 26px; color: var(--espresso); letter-spacing: -0.4px; line-height: 1; }
.wm .wm-jl { font-family: 'JetBrains Mono', monospace; font-style: normal; font-size: 20px; color: var(--caramel); font-weight: 500; }
footer .wm .wm-txt { color: var(--cream); }

/* Responsive */
@media (max-width: 1024px) {
  .hero-grid { grid-template-columns: 1fr; gap: 40px; }
  .engines { grid-template-columns: 1fr; }
  .engine { border-right: none; border-bottom: 1px solid var(--tan); }
  .engine:last-child { border-bottom: none; }
  .cards { grid-template-columns: 1fr; }
  .foot-grid { grid-template-columns: 1fr 1fr; }
}
</style>
