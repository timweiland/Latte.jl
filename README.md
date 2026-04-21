# Latte

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://timweiland.github.io/Latte.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://timweiland.github.io/Latte.jl/dev/)
[![Build Status](https://github.com/timweiland/Latte.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/timweiland/Latte.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/timweiland/Latte.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/timweiland/Latte.jl)

**Latte is a Bayesian inference toolkit for latent Gaussian models.** Write your
model once in [DynamicPPL](https://github.com/TuringLang/DynamicPPL.jl) and pick
which approximation to run: fast grid-based integration (INLA), point-estimate +
Gaussian covariance (TMB), or HMC on the Laplace marginal (tmbstan-style).

> ⚠️ **Early development.** API is not yet stable. Expect breaking changes until a 1.0.

## Example

```julia
using Latte, DynamicPPL, Distributions, LinearAlgebra

@model function hier_poisson(y, X, group)
    τ_u ~ Gamma(2, 1)
    β   ~ MvNormal(zeros(size(X, 2)), 100.0 * I)
    u   ~ MvNormal(zeros(maximum(group)), (1/τ_u) * I)
    for i in eachindex(y)
        y[i] ~ Poisson(exp(X[i, :] ⋅ β + u[group[i]]); check_args=false)
    end
end

# DSL → Latent Gaussian Model (one line)
lgm = latte_from_dppl(hier_poisson(y_obs, X, group); random = (:β, :u))

# Three inference methods, same model, shared result protocol
r_inla = inla(lgm, y_obs)          # grid/CCD integration over θ
r_tmb  = tmb(lgm, y_obs)           # MAP + Laplace covariance, TMB-style
r_hmc  = hmc_laplace(lgm, y_obs)   # NUTS on the Laplace marginal

# Uniform access for any result type
hyperparameter_mode(r_inla)        # → NaturalHyperparameters
latent_marginals(r_tmb)            # → Vector{<:Distribution}
rand(r_hmc, 500)                   # → PosteriorSamples

# How trustworthy is the Laplace approximation here?
diagnose(r_tmb)
# → (rel_ess = 0.87, pareto_k = 0.12, interpretation = :excellent, ...)
```

## What's in the box

- **Inference methods**:
  - [`inla`](https://timweiland.github.io/Latte.jl/dev/) — Integrated Nested Laplace Approximation (Rue et al. 2009)
  - [`tmb`](https://timweiland.github.io/Latte.jl/dev/) — TMB-style MAP + Laplace covariance (Monnahan & Kristensen 2018)
  - [`hmc_laplace`](https://timweiland.github.io/Latte.jl/dev/) — NUTS on the Laplace marginal, warm-started from TMB (tmbstan-style)
- **DSL front door**: `latte_from_dppl(m; random=...)` turns any `@model ... end`
  into a `LatentGaussianModel`. Auto-detects the latent DAG, falls back to
  sparse AD for non-linear priors, and pattern-matches common likelihoods
  (Poisson / Bernoulli / Normal) onto `GaussianMarkovRandomFields.jl`'s
  hand-coded `ExponentialFamily` observation models.
- **Shared protocol**: every inference result implements the same interface —
  `latent_marginals`, `hyperparameter_marginals`, `hyperparameter_mode`,
  `log_marginal_likelihood`, `rand`, etc. Method-agnostic post-processing
  (prediction, linear combinations, WAIC/CPO) composes from this.
- **PSIS-k̂ diagnostic**: `diagnose(r)` tells you when the inner Laplace
  approximation is trustworthy — works uniformly across all three methods.

## Architecture

Latte factors around the latent Gaussian model structure `p(θ) p(x|θ) p(y|x,θ)`
with `p(x|θ)` Gaussian. Inference methods share an inner Laplace for `x|θ` and
differ only on how they treat `θ`:

| method | θ treatment | best for |
|---|---|---|
| INLA | grid / CCD + spline interpolation | small θ, marginal densities |
| TMB | point MAP + Gaussian covariance | familiar sdreport-style output |
| HMC-Laplace | NUTS on L(θ) with TMB warm start | when Laplace is good but you want sampling |

See `src/LAYOUT.md` for the architectural reference.

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for the full dev workflow.

```bash
make setup      # install dependencies
make test       # run the test suite
make format     # format with Runic
```

## Related work

- [INLA](https://www.r-inla.org/) (R) — reference implementation of the INLA algorithm
- [TMB](https://kaskr.github.io/adcomp/_book/Introduction.html) (R / C++) — template model builder
- [tmbstan](https://github.com/kaskr/adcomp/wiki/tmbstan) — HMC on TMB models
- [ParetoSmooth.jl](https://github.com/TuringLang/ParetoSmooth.jl) — PSIS implementation (vendored for `diagnose`)

## Citing

If you use Latte in your work, please cite the package; a proper software
citation will appear here once a DOI is attached.
