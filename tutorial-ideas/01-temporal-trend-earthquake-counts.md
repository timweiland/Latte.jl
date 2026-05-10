# Tutorial Idea: Temporal Trend Smoothing — Global Earthquake Activity

## One-liner

Use INLA with random walk models to smooth annual counts of major earthquakes and uncover long-term trends.

## Application

Tracking changes in global earthquake frequency over time (1900–2006). Do major earthquakes cluster in certain decades? Is the rate stable or changing? The tutorial fits Poisson models with RW1 and RW2 temporal random effects to separate signal from noise in a century of earthquake counts.

## Technical setup

- **Family**: Poisson (log link)
- **Latent structure**: RW1 and RW2 random walks for temporal smoothing
- **Hyperparameters**: Precision of the random walk (controls smoothness)
- **Dataset**: Annual counts of major earthquakes (magnitude >= 7), 1900–2006. This is a classic dataset used in the INLA gitbook (originally from Zucchini et al., 2016). ~107 observations — compact enough to hard-code directly in the tutorial.

## Tutorial outline

1. **Motivation**: Plot the raw earthquake counts. They're noisy — is there structure?
2. **RW1 model**: Fit a Poisson-RW1 model. Discuss how RW1 penalizes first differences (piecewise linear).
3. **RW2 model**: Fit a Poisson-RW2 model. Discuss how RW2 penalizes second differences (piecewise quadratic), producing smoother estimates.
4. **Visual comparison**: Overlay both fitted trends on the raw data. The RW1 is wiggly, the RW2 is smooth — a vivid illustration of model choice affecting inference.
5. **Hyperparameter interpretation**: The random walk precision controls the bias-variance tradeoff. Low precision = wiggly (follows data closely), high precision = smooth (shrinks toward a simple trend). Examine the posterior of this precision.
6. **Model comparison**: Compare RW1 vs RW2 via DIC. Which one fits better? Which one is more scientifically interpretable?
7. **Posterior predictive checks**: Do the fitted models capture the observed variability?

## Why it's valuable

- **Time series is INLA's bread and butter**, but no existing tutorial covers temporal data at all. Every existing tutorial uses cross-sectional data.
- **RW1/RW2 are fundamental GMRF types** — arguably the simplest structured random effects beyond IID. A user who understands these can build toward AR1, seasonal models, and spatio-temporal models.
- **The story is compelling and visual**: raw counts are noisy, the smoothed trend reveals structure, and comparing RW1 vs RW2 makes the concept of "smoothness" tangible.
- **Small, self-contained dataset** with no external dependencies — can be hard-coded in ~10 lines.

## How it differs from existing tutorials

| Aspect | Existing tutorials | This tutorial |
|--------|-------------------|---------------|
| Data structure | Cross-sectional (hospitals, counties, simulated covariates) | **Temporal** (ordered time points) |
| Latent model | IID, Besag | **RW1/RW2** (random walks) |
| Story | Group-specific effects, spatial risk, model selection | **Trend estimation** over time |
| Model comparison angle | BMA tutorial compares models with different covariates | Compares models with **different smoothness assumptions** |

## References

- Zucchini, W., MacDonald, I. L., & Langrock, R. (2016). *Hidden Markov Models for Time Series*. Chapter on earthquake data.
- [INLA gitbook, Chapter 8: Temporal Models](https://becarioprecario.bitbucket.io/inla-gitbook/ch-temporal.html) — uses this exact dataset.
- [Time Series in R using INLA](https://tem11010.github.io/timeseries-inla/) — demonstrates RW2 on Great Lakes water levels.
