# Tutorial Idea: Nonlinear Regression as a Bayesian GAM — Motorcycle Crash Acceleration

## One-liner

Use INLA with an RW2 smooth to fit a nonlinear curve to crash test data, demonstrating INLA as a Bayesian alternative to GAMs.

## Application

In a simulated motorcycle crash experiment, an accelerometer on the rider's helmet measures head acceleration at 133 time points after impact. The acceleration curve is highly nonlinear — a sharp spike followed by damped oscillations. The goal is to estimate this curve with proper uncertainty quantification. This is nonparametric regression: we don't know the functional form, so we let the data speak through a smooth random effect.

## Technical setup

- **Family**: Gaussian
- **Latent structure**: RW2 random walk used as a nonlinear smooth (i.e., Bayesian P-spline / GAM equivalent)
- **Hyperparameters**: Observation noise precision + RW2 precision (controls smoothness)
- **Dataset**: The `mcycle` dataset (Silverman, 1985). 133 observations of (time, acceleration). Compact enough to hard-code directly. This is arguably the most iconic dataset in nonparametric regression — used in every GAM textbook.

## Tutorial outline

1. **Motivation**: Plot the raw data. The relationship between time and acceleration is clearly nonlinear — no polynomial will capture the sharp peak and subsequent oscillation. Mention that this is exactly the kind of problem GAMs (e.g., `mgcv::gam()`) are designed for, and that INLA provides a fully Bayesian alternative.
2. **The connection between smoothing and GMRFs**: Brief explanation of how an RW2 prior on a discretized function is equivalent to penalizing the second derivative — the same principle underlying cubic smoothing splines. The RW2 precision plays the role of the smoothing parameter.
3. **Fitting the model**: Set up the RW2 smooth over the time covariate. Discuss the PC prior on the RW2 precision: it penalizes complexity (wigglyness) relative to a simple linear baseline.
4. **Visualizing the fit**: Plot the posterior mean curve with 95% credible band overlaid on the raw data. The credible band widens where data is sparse — a natural advantage of the Bayesian approach.
5. **Hyperparameter posteriors**: Examine the posteriors of both the observation noise precision and the RW2 precision. The ratio of these two precisions determines the effective smoothness.
6. **Comparison with simpler models**: Fit a plain linear model and a low-order polynomial. Show via DIC (or visual inspection) that these are inadequate, motivating the nonparametric approach.
7. **Discussion**: When to use this approach vs. a frequentist GAM. The Bayesian version gives you full posterior uncertainty on both the curve and the smoothness. Mention extensions: heteroscedastic noise, multiple smooths, spatial smoothing (thin-plate splines via SPDE).

## Why it's valuable

- **Shows INLA beyond count data**: All existing tutorials use Poisson or Binomial. This is the first with a **Gaussian family**, demonstrating that INLA is a general inference framework, not just a tool for disease mapping.
- **Connects INLA to GAMs**: Many users come from the GAM world (`mgcv`, `brms`). Showing that INLA can do the same thing — with proper Bayesian uncertainty — is a powerful selling point.
- **Teaches a deep insight**: The equivalence between random walks and spline smoothing is one of the most beautiful ideas in computational statistics. This tutorial makes it concrete.
- **Iconic dataset**: The mcycle data produces a visually striking fit that immediately demonstrates the value of nonparametric regression. It's familiar to anyone who has studied smoothing.
- **Two hyperparameters with distinct roles**: The interplay between noise precision and smoothness precision is pedagogically rich and different from the single-precision setups in other tutorials.

## How it differs from existing tutorials

| Aspect | Existing tutorials | This tutorial |
|--------|-------------------|---------------|
| Family | Binomial, Poisson | **Gaussian** |
| Task | Group effects, spatial risk, model selection | **Nonlinear curve estimation** |
| Latent model role | Random effect capturing group/spatial variation | Random effect **IS the function** being estimated |
| Uncertainty story | Credible intervals on risk ratios or model weights | Credible **band** on a continuous curve |
| Connection to other methods | Standalone INLA demos | Explicitly connects to **GAMs and smoothing splines** |

## References

- Silverman, B. W. (1985). Some aspects of the spline smoothing approach to non-parametric regression curve fitting. *JRSS-B*.
- Rue, H. & Held, L. (2005). *Gaussian Markov Random Fields: Theory and Applications*. Chapter on RW models and smoothing.
- The mcycle dataset is available in R's `MASS` package and has been reproduced in many open sources.
