```@raw html
---
layout: false
---

<LatteNav />

<main class="tutorials-page">
  <div class="container">
    <h1>Tutorials</h1>
    <p class="lede">
      End-to-end examples, ordered roughly from simplest to most involved.
      Each follows the same recipe — define the model as an <code>@latte</code>
      function (or a DPPL <code>@model</code>), then fit it with a single
      <code>inla()</code> call.
    </p>
    <TutorialGallery />
  </div>
</main>

<LatteFooter />

<style>
.tutorials-page {
  background: #FAF7F2;
  color: #2A1810;
  font-family: 'Inter', system-ui, sans-serif;
  padding: 64px 0 96px;
  min-height: 60vh;
}
.tutorials-page .container { max-width: 1200px; margin: 0 auto; padding: 0 48px; }
.tutorials-page h1 {
  font-family: 'Fraunces', Georgia, serif;
  font-style: italic;
  font-weight: 400;
  font-size: clamp(40px, 5vw, 64px);
  letter-spacing: -0.03em;
  line-height: 1;
  margin: 0 0 20px;
  color: #2A1810;
}
.tutorials-page .lede {
  font-size: 18px;
  line-height: 1.55;
  color: #4A3828;
  max-width: 640px;
  margin: 0 0 24px;
}
.tutorials-page code {
  background: #E8D5B7;
  color: #3D2817;
  padding: 1px 6px;
  border-radius: 3px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.92em;
}
</style>
```
