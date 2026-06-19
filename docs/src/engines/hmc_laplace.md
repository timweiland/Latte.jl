# [HMC-Laplace](@id engine-hmc-laplace)

HMC-Laplace is Latte's sampling engine. It runs the No-U-Turn Sampler (NUTS) over
the hyperparameters, with the latent field marginalised out by a Laplace
approximation at every step. The entry point is [`hmc_laplace`](@ref).

## How it works

This is not plain NUTS over the whole model. Sampling the latent field and the
hyperparameters together is the funnel-shaped problem that makes Hamiltonian Monte
Carlo struggle on hierarchical models. HMC-Laplace sidesteps it by letting NUTS
move only in the low-dimensional hyperparameter space ``\theta``.

The price is that, to evaluate its target ``p(\theta \mid y)``, NUTS needs the
marginal likelihood ``p(y \mid \theta)``, and that means integrating out the latent
field. HMC-Laplace does this the same way INLA and TMB do, with an inner Laplace
approximation. At every ``\theta`` the sampler visits, it solves the latent field's
Gaussian approximation, reads off ``p(y \mid \theta)`` and its gradient, and hands
them back to NUTS. So every step the sampler takes carries an inner Laplace solve.
The picture below shows it: each draw on the left runs its own inner Laplace, drawn
on the right.

What the cost buys is faithfulness. Because NUTS samples the real
``p(\theta \mid y)``, the hyperparameter posterior comes out with whatever skew it
genuinely has, which is exactly where TMB's single Gaussian and INLA's coarse grid
can fall short. The latent marginals are the average of the inner Laplace
approximations across the draws.

```@raw html
<HmcLaplace />
```

## Tuning

### Chain length (`n_samples`, `n_warmup`)

The usual MCMC controls. `n_warmup` (default 200) is the adaptation and burn-in,
and the kept draws number `n_samples` (default 500). Raise both when the effective
sample size is low or the marginals look noisy. Each step runs an inner Laplace, so
longer chains cost real wall-clock time.

```julia
hmc_laplace(model, y; n_samples = 2000, n_warmup = 1000)
```

### Reproducibility (`rng`) and gradients (`diff_strategy`)

Pass an `rng` for a chain you can reproduce. `diff_strategy` sets how the gradient
of the target is taken, as on the [TMB](@ref engine-tmb) page; the default
`ADStrategy()` suits `@latte` models with recognised GMRF latents.

## Reference

```@docs
hmc_laplace
```

## Limits

- **It is MCMC, so the marginals carry Monte Carlo noise.** That noise shrinks with
  the effective sample size, unlike the deterministic INLA and TMB. Check the ESS
  and run longer chains before making tail-sensitive claims.
- **It is the slowest engine.** Every leapfrog step is an inner Laplace solve, and
  NUTS takes many steps per draw. Reach for it when the faithfulness is worth the
  time, or when the hyperparameter posterior is too awkward for a grid.
- **Constrained or rank-deficient latents can stall it.** NUTS occasionally proposes
  a ``\theta`` where the inner solve is numerically singular. Those steps are
  rejected, which can hurt mixing.

## References

```@raw html
<div class="ref-grid-2">
<PaperCite
  tag="NUTS"
  title="The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo"
  authors="M. D. Hoffman & A. Gelman"
  venue="Journal of Machine Learning Research" year="2014"
  arxiv="1111.4246"
  url="https://arxiv.org/abs/1111.4246"
  abstract="The No-U-Turn Sampler, the adaptive Hamiltonian Monte Carlo algorithm Latte runs over the hyperparameters." />
<PaperCite
  tag="Embedded Laplace + HMC"
  title="Hamiltonian Monte Carlo using an Adjoint-differentiated Laplace Approximation"
  authors="C. C. Margossian, A. Vehtari, D. Simpson & R. Agrawal"
  venue="Advances in Neural Information Processing Systems (NeurIPS)" year="2020"
  arxiv="2004.12550"
  url="https://arxiv.org/abs/2004.12550"
  abstract="Hamiltonian Monte Carlo over the hyperparameters, with the latent Gaussian field marginalised by an embedded Laplace approximation and the gradient propagated through that inner solve. The method this engine implements." />
</div>
```

## See also

- [INLA](@ref engine-inla) and [TMB](@ref engine-tmb): the deterministic engines
  that approximate the same ``p(\theta \mid y)`` that HMC-Laplace samples.
- [Benchmarks](../benchmarks/index.md): where its extra cost shows up.
- [Validation](../validation/index.md): where its faithfulness pays off.
- [API reference](@ref api-reference): defining models and working with results.
