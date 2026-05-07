<script setup lang="ts">
type Tutorial = {
  href: string
  tag: string
  title: string
  blurb: string
  image?: string  // optional preview image path; future use
}

const tutorials: Tutorial[] = [
  {
    href: '/tutorials/getting_started',
    tag: 'GETTING STARTED',
    title: 'Surgery mortality across hospitals',
    blurb: 'A walkthrough of the simplest possible Bayesian analysis with Latte: hospital-by-hospital mortality rates, an IID random effect, and INLA in a few lines.',
  },
  {
    href: '/tutorials/disease_mapping_spatial',
    tag: 'SPATIAL · DISEASE MAPPING',
    title: 'BYM: Besag + IID',
    blurb: 'Regional disease risk smoothed with an ICAR spatial prior plus an IID residual. PC priors on both precisions, fit with INLA.',
  },
  {
    href: '/tutorials/temporal_trend_earthquakes',
    tag: 'TEMPORAL · COUNTS',
    title: 'Earthquake intensity trends',
    blurb: 'RW1 vs RW2 priors on annual seismicity counts, compared by DIC, WAIC, and marginal likelihood — all from the same fit object.',
  },
  {
    href: '/tutorials/nonlinear_regression_gam',
    tag: 'REGRESSION · GAM',
    title: 'Nonlinear regression with RW2',
    blurb: 'A generalised-additive-model take on a smooth regression: an RW2 prior on the latent function, Gaussian likelihood.',
  },
  {
    href: '/tutorials/tweedie_insurance',
    tag: 'CUSTOM LIKELIHOOD',
    title: 'Tweedie regression on insurance claims',
    blurb: 'Write any logpdf, get full INLA inference. A hand-coded compound Poisson-Gamma likelihood — zero-inflated continuous, no fast-path support — fits in the same DPPL @model + inla() flow.',
  },
  {
    href: '/tutorials/spatial_spde',
    tag: 'SPATIAL · SPDE',
    title: 'Matérn SPDE on a mesh',
    blurb: 'Continuous-domain spatial smoothing via the SPDE/Matérn approach. Build a triangulated mesh, define the Matérn precision, fit.',
  },
  {
    href: '/tutorials/spatio_temporal_separable',
    tag: 'SPACE-TIME · SEPARABLE',
    title: 'Kronecker space-time',
    blurb: 'Region-specific temporal dynamics via a SeparableModel Kronecker prior. Additive vs interaction-only vs full, on one DPPL spec.',
  },
  {
    href: '/tutorials/bayesian_model_averaging',
    tag: 'MODEL COMPARISON',
    title: 'Bayesian model averaging',
    blurb: 'Three competing Poisson regressions, scored by marginal likelihood, then weighted into a posterior-averaged prediction.',
  },
  {
    href: '/tutorials/posterior_predictive_checks',
    tag: 'DIAGNOSTICS · PPC',
    title: 'Posterior predictive checks',
    blurb: 'Catch model misspecification by simulating from the fitted posterior. A Poisson model fitted to overdispersed counts, exposed by a `std` PPC.',
  },
  {
    href: '/tutorials/sbc_calibration',
    tag: 'DIAGNOSTICS · SBC',
    title: 'Simulation-based calibration',
    blurb: 'Validate the inference procedure itself: rank true parameters in the posterior across many simulated datasets. Uniform ranks ⇒ calibrated.',
  },
  {
    href: '/tutorials/turing_handoff',
    tag: 'INTEROP · MCMC',
    title: 'Handoff to Turing',
    blurb: 'The same DPPL @model that fits with INLA also samples cleanly under Turing.sample(NUTS()). Use it as a gold-standard cross-check.',
  },
]
</script>

<template>
  <div class="tutorial-gallery">
    <a
      v-for="t in tutorials"
      :key="t.href"
      class="t-card"
      :href="t.href"
    >
      <div v-if="t.image" class="t-card-image">
        <img :src="t.image" :alt="t.title" />
      </div>
      <div class="t-card-body">
        <div class="t-card-tag">{{ t.tag }}</div>
        <h3 class="t-card-title">{{ t.title }}</h3>
        <p class="t-card-blurb">{{ t.blurb }}</p>
      </div>
    </a>
  </div>
</template>

<style scoped>
.tutorial-gallery {
  --t-bg:       #FAF7F2;
  --t-foam:     #FFFCF7;
  --t-tan:      #E8D5B7;
  --t-caramel:  #C9986A;
  --t-mocha:    #8B6F47;
  --t-bean:     #3D2817;
  --t-espresso: #2A1810;

  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 18px;
  margin: 32px 0 64px;
}

.t-card {
  background: var(--t-foam);
  border: 1px solid var(--t-tan);
  padding: 0;
  display: flex;
  flex-direction: column;
  text-decoration: none;
  color: inherit;
  transition: transform .15s, box-shadow .15s;
  overflow: hidden;
}
.t-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 16px 40px rgba(42,24,16,0.08);
}

.t-card-image {
  width: 100%;
  aspect-ratio: 16 / 9;
  background: var(--t-tan);
  overflow: hidden;
  border-bottom: 1px solid var(--t-tan);
}
.t-card-image img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.t-card-body {
  padding: 24px 22px 22px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  flex-grow: 1;
}

.t-card-tag {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--t-caramel);
  text-transform: uppercase;
  letter-spacing: 1.3px;
}

.t-card-title {
  font-family: 'Fraunces', Georgia, serif;
  font-style: italic;
  font-weight: 500;
  font-size: 22px;
  letter-spacing: -0.4px;
  line-height: 1.18;
  margin: 0;
  color: var(--t-espresso);
}

.t-card-blurb {
  font-size: 14px;
  line-height: 1.55;
  color: #4A3828;
  margin: 0;
}

@media (max-width: 1024px) {
  .tutorial-gallery { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 640px) {
  .tutorial-gallery { grid-template-columns: 1fr; }
}

/* Dark mode */
:global(.dark) .tutorial-gallery {
  --t-foam:     #38241B;
  --t-tan:      rgba(201, 152, 106, 0.18);
  --t-caramel:  #C9986A;
  --t-mocha:    #B79877;
  --t-bean:     #3D2817;
  --t-espresso: #F5E6D3;
}
:global(.dark) .t-card-blurb {
  color: #D4B896;
}
:global(.dark) .t-card:hover {
  box-shadow: 0 16px 40px rgba(0,0,0,0.4);
}
</style>
