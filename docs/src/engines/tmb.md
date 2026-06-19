# [TMB](@id engine-tmb)

TMB is Latte's fast engine. It fits a latent Gaussian model the way the TMB R
package does: find the most likely hyperparameters, then describe the uncertainty
around them with a single Gaussian. The entry point is [`tmb`](@ref).

## How it works

Like INLA, TMB approximates the latent field at a fixed value of the
hyperparameters ``\theta`` by a Gaussian (the inner Laplace step). The difference
is what it does with the hyperparameters themselves.

Instead of spreading a grid of ``\theta`` points, TMB finds the mode of
``p(\theta \mid y)`` (the MAP) and fits one Gaussian right there, taking its width
from the curvature at the mode (the Hessian of the log posterior). That gives a MAP
estimate, a standard error for each hyperparameter, and Gaussian-propagated
marginals for the latent field. It is the same information TMB's `sdreport`
returns.

This is cheaper than a grid: one optimisation and one Hessian, with no integration
over ``\theta``. It is also exact when ``p(\theta \mid y)`` really is Gaussian,
which it often nearly is in working (log or logit) space for well-identified
models. The cost shows up when that posterior is skewed. A single Gaussian at the
mode cannot bend to follow the skew, so the marginals drift. The picture below
shows it: with no skew the Gaussian and the true posterior sit on top of each
other, and as the skew grows they come apart.

```@raw html
<TmbGaussian />
```

## Tuning

TMB has far fewer knobs than INLA. The one that matters is how the Hessian is
computed.

### Hessian computation (`diff_strategy`)

- `ADStrategy()`, the default: ForwardDiff gradients with central differences for
  the Hessian. Accurate and robust for `@latte` models with recognised GMRF
  latents, and for the broad class of custom-`logpdf` likelihoods.
- `FiniteDiffStrategy()`: a plain finite-difference fallback. The narrow case that
  needs it is a hyperparameter-derived value hoisted into the observation payload by
  the `@latte` prelude-lift (`φ = exp(log_φ)` in the Tweedie tutorial), which the
  outer Hessian can't keep dual-typed. It is not required for custom likelihoods in
  general.

```julia
tmb(model, y; diff_strategy = FiniteDiffStrategy())
```

## Reference

```@docs
tmb
```

## Limits

- **Skewed hyperparameter posteriors.** The Gaussian at the mode is symmetric, so
  it misses any skew in ``p(\theta \mid y)``. The [Validation](../validation/index.md)
  page shows this plainly: TMB is well calibrated when the hyperparameter posterior
  is close to Gaussian (the Normal models), and off when it is skewed (count and
  binary models with little data). When you need the skew, reach for
  [INLA](@ref engine-inla) or HMC-Laplace.
- **It reports a mode, not a full marginal.** The credible intervals come from the
  Gaussian standard errors, so they are only as good as that Gaussian assumption.

## References

```@raw html
<div class="ref-grid-2">
<PaperCite
  tag="TMB"
  title="TMB: Automatic Differentiation and Laplace Approximation"
  authors="K. Kristensen, A. Nielsen, C. W. Berg, H. Skaug & B. M. Bell"
  venue="Journal of Statistical Software" year="2016"
  doi="10.18637/jss.v070.i05"
  url="https://doi.org/10.18637/jss.v070.i05"
  abstract="The TMB R package: fast Laplace approximation of the marginal likelihood for latent-variable models, with automatic differentiation for the gradients and Hessians." />
<PaperCite
  tag="Laplace for random effects"
  title="Automatic Approximation of the Marginal Likelihood in Non-Gaussian Hierarchical Models"
  authors="H. J. Skaug & D. A. Fournier"
  venue="Computational Statistics & Data Analysis" year="2006"
  doi="10.1016/j.csda.2006.03.005"
  url="https://doi.org/10.1016/j.csda.2006.03.005"
  abstract="The Laplace approximation for random-effect models that TMB builds on, combined with automatic differentiation." />
</div>
```

## See also

- [INLA](@ref engine-inla): the grid-based engine, for when you need the
  hyperparameter skew TMB misses.
- [Benchmarks](../benchmarks/index.md): where TMB's speed pays off.
- [Validation](../validation/index.md): where its Gaussian assumption holds and
  where it doesn't.
- [API reference](@ref api-reference): defining models and working with results.
