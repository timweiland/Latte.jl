# Tutorial Idea: Species Occurrence Modeling — Bernoulli GLMM for Ecological Surveys

## One-liner

Model the presence or absence of a species across survey sites using Bernoulli regression with environmental covariates and site-level random effects.

## Application

A field ecologist surveys 200 forest plots for the presence/absence of a target species (e.g., a ground-nesting bird). At each site, they record environmental covariates: elevation, canopy cover, and distance to nearest water source. The scientific questions are: Which environmental factors drive occurrence? What is the predicted probability of finding the species at a new site? How does accounting for unobserved site-level heterogeneity change our conclusions?

## Technical setup

- **Family**: Bernoulli (i.e., Binomial with n=1, logit link)
- **Latent structure**: Fixed effects (elevation, canopy cover, distance to water) + IID site-level random effects for overdispersion
- **Hyperparameters**: Precision of the IID site-level effect
- **Dataset**: Simulated from known parameters. This is a deliberate pedagogical choice — having ground truth allows validating that the posteriors cover the true values, which builds trust in the method.

## Tutorial outline

1. **Motivation and data generation**: Describe the ecological scenario. Simulate 200 sites with three environmental covariates and a known logistic model: `logit(p) = beta_0 + beta_1 * elevation + beta_2 * canopy + beta_3 * water + epsilon_site`. Plot the raw presence/absence data against each covariate.
2. **Naive model (no random effects)**: Fit a Bernoulli model with fixed effects only. Examine the coefficient posteriors. Compare to the known true values.
3. **Full model (with site-level random effects)**: Add IID random effects to capture unobserved heterogeneity. Compare coefficient posteriors to the naive model — do the credible intervals widen? Does the point estimate shift? This demonstrates the importance of accounting for overdispersion in binary data.
4. **Predicted occurrence probabilities**: For a grid of covariate values, compute the posterior predictive probability of occurrence. Plot occurrence probability as a function of elevation (holding other covariates at their means), with a credible band showing uncertainty. The logistic curve with uncertainty is a classic and visually appealing result.
5. **Posterior calibration check**: Since we simulated the data, verify that the 95% credible intervals contain the true parameters ~95% of the time. This is a powerful pedagogical moment: it shows that INLA's approximate posteriors are well-calibrated.
6. **Model comparison**: Compare the model with and without random effects via DIC. When is the extra complexity justified?
7. **Discussion**: Mention extensions to spatial random effects (linking to the disease_mapping tutorial), and to multi-species models. Note that this same setup applies to any binary outcome: disease diagnosis, customer churn, equipment failure, etc.

## Why it's valuable

- **Binary data is everywhere**: Bernoulli/logistic regression is one of the most common models in all of applied statistics, yet no existing tutorial covers it. Anyone doing classification, medical diagnostics, or ecological surveys will relate.
- **Exercises the logit link**: The existing tutorials use the log link (Poisson) and logit-for-Binomial-with-known-n. Individual-level binary data with a logit link is a distinct and important use case.
- **Ecology is INLA's home turf**: Species distribution modeling is one of the largest application areas for INLA in practice. Ecologists coming to this package will look for exactly this kind of tutorial.
- **Simulated data with known truth is pedagogically powerful**: It lets you validate the posterior — something you can never do with real data. This builds confidence in the method and teaches good statistical practice.
- **Natural bridge to spatial models**: The tutorial can end by noting that the IID site-level effects could be replaced by a spatial random field — providing a clear on-ramp to the existing disease_mapping tutorial.

## How it differs from existing tutorials

| Aspect | Existing tutorials | This tutorial |
|--------|-------------------|---------------|
| Family | Binomial (known n), Poisson | **Bernoulli** (individual binary outcomes) |
| Link function | log (Poisson), logit-with-totals (Binomial) | **logit** on individual observations |
| Domain | Healthcare, epidemiology, regression | **Ecology / conservation biology** |
| Data source | Real data (hospital, county-level) or simple simulation | **Simulated with known ground truth** for validation |
| Pedagogical focus | How to use the API | How to **validate** that posteriors are calibrated |
| Extension path | Standalone | Explicitly bridges to **spatial tutorial** as next step |

## References

- [Intro to modelling using INLA (Our Coding Club)](https://ourcodingclub.github.io/tutorials/inla/) — ecology-focused INLA tutorial covering spatial species models.
- [Beginner's Guide to Spatial, Temporal and Spatial-Temporal Ecological Data Analysis with R-INLA](https://www.highstat.com/index.php/beginner-s-guide-to-regression-models-with-spatial-and-temporal-correlation) — comprehensive ecology + INLA resource by Zuur et al.
- [INLA gitbook, Chapter 7: Spatial Models](https://becarioprecario.bitbucket.io/inla-gitbook/ch-spatial.html) — for the spatial extensions mentioned in the discussion.
