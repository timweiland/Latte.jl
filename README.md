<div align="center">
  <img src="docs/src/assets/logo.svg" alt="Latte.jl logo" width="150"/>
  <h1>Latte.jl</h1>
  <p><strong>Probabilistic programming for latent Gaussian models</strong></p>
</div>

<div align="center">

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://timweiland.github.io/Latte.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://timweiland.github.io/Latte.jl/dev/)
[![Build Status](https://github.com/timweiland/Latte.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/timweiland/Latte.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/timweiland/Latte.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/timweiland/Latte.jl)

</div>

You have a latent Gaussian model — a spatial disease map, a temporal trend, a hierarchical GLMM, a Gaussian process on a mesh — and you want a posterior without a long MCMC run. These models have a fixed structure: a Gaussian Markov random field for the latent field, with a few hyperparameters on top. INLA and TMB exploit exactly that structure.

Latte lets you write the model once and choose how to handle it. Describe the model with the `@latte` macro using `~` statements, then pick an inference engine: grid-based integration over the hyperparameters (INLA), a point estimate with a Gaussian covariance (TMB), or NUTS run on the Laplace marginal (tmbstan-style). The model definition does not change when you switch engines, and every engine returns results through the same set of accessors.

> [!WARNING]
> **Early development.** Latte is `v0.1.0-DEV`. The API is not yet stable and breaking changes should be expected before a 1.0 release. It is not yet in the General registry; install from the GitHub URL below.

## Installation

```julia
] add https://github.com/timweiland/Latte.jl
```

Once Latte is registered in the General registry, `] add Latte` will work too.

## Example

A hierarchical Poisson GLMM — a fixed slope plus a per-group random intercept — written once and run through all three engines:

```julia
using Latte, Distributions, LinearAlgebra

@latte function hier_poisson(y, x, group, n_groups)
    τ_u ~ Gamma(2.0, 1.0)
    β   ~ MvNormal(zeros(2), 100.0 * I(2))
    u   ~ IIDModel(n_groups, constraint = :sumtozero)(τ = τ_u)
    for i in eachindex(y)
        y[i] ~ Poisson(exp(β[1] + β[2] * x[i] + u[group[i]]))
    end
end

# Calling the @latte function returns a LatentGaussianModel.
# (y, x are length-N vectors of counts/covariates; group maps each
#  observation to one of n_groups; n_groups is the group count.)
lgm = hier_poisson(y, x, group, n_groups)

# Same model, three engines, one shared result protocol.
r_inla = inla(lgm, y)                 # grid integration over θ (default)
r_tmb  = tmb(lgm, y)                  # MAP + Laplace covariance
r_hmc  = hmc_laplace(lgm, y)          # NUTS on the Laplace marginal

# Accessor functions work the same across all three engines.
β_marginals = latent_marginals(r_inla, :β)            # one marginal per coefficient
τ_marginal  = hyperparameter_marginals(r_inla, :τ_u)[1]
hyperparameter_mode(r_tmb)            # natural-space hyperparameter mode
converged(r_hmc)                      # did the chain converge?

samples = rand(r_inla, 200)           # → PosteriorSamples
diagnose(r_inla)                      # PSIS-k̂ check on the Laplace approximation
```

The `IIDModel` constructor and the rest of the latent-model layer are re-exported from [GaussianMarkovRandomFields.jl](https://github.com/timweiland/GaussianMarkovRandomFields.jl), so `using Latte` is enough to name them in an `@latte` body, with no separate import.

Marginals come back as [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) objects, so `mean`, `std`, `quantile`, `pdf`, and `rand` work on them directly. Hyperparameter marginals are returned in natural (prior-declared) space, so there is nothing to back-transform.

To learn the workflow end to end, start with the [tutorial gallery](https://timweiland.github.io/Latte.jl/dev/tutorials/), compare engines on real datasets in the [benchmarks against R-INLA](https://timweiland.github.io/Latte.jl/dev/benchmarks/), or read the [full documentation](https://timweiland.github.io/Latte.jl/dev/).

## What you can model

Latte targets the latent Gaussian model class, `p(θ) · p(x|θ) · p(y|x,θ)` with `x` a Gaussian Markov random field. You build the components directly in an `@latte` body from the re-exported constructors:

- IID and fixed-effects terms (`IIDModel`, `FixedEffectsModel`) for GLMMs
- ICAR/BYM2 disease mapping (`BesagModel`, `BYM2Model`) and Matérn SPDE fields on a mesh (`MaternModel`), with `BarrierModel` for coastlines and physical barriers
- Random-walk and autoregressive time terms (`RW1Model`, `RW2Model`, `AR1Model`, `ARModel`)
- Separable spatio-temporal fields (`SeparableModel`) and component sums (`CombinedModel`)
- Spline / GAM smoothers built from random walks

The tutorial gallery works through disease mapping, an SPDE log-Gaussian Cox process for earthquake intensity, temporal trends, separable spatio-temporal fields, spatial survival, GAM regression, and a barrier-coastline model.

Latent priors do not have to be Gaussian. When the prior itself is nonlinear in the latent variables — for example a survival recursion — `@latte` reads the `~` statements as a factor graph, recognizes the coupling and nonlinearity, and fits by iterated Laplace with no extra setup. The age-structured fisheries assessment (SAM) tutorial fits an 80-dimensional, two-field coupled latent field with an `exp(logF)` survival nonlinearity; `inla` and `tmb` posterior means agree closely, and both track a full-NUTS reference across the whole field to within Monte Carlo error.

One honest limit here: for non-Gaussian latent priors Latte reports Gaussian marginals. The iterated-Laplace posterior mean and precision are exact, but the higher-moment skew of each marginal is not yet corrected. For smooth fields this skew is typically small; it is not corrected regardless. Some post-processing (WAIC/CPO, posterior-predictive utilities) falls back to Monte Carlo or is not yet wired up for the nonlinear and composite-observation cases.

## Inference engines

All three engines share the same inner Laplace approximation for the latent field `x|θ`. They differ only in how they treat the hyperparameters `θ`, and they return results that implement the same interface.

| Engine | θ treatment | Good fit when |
|---|---|---|
| [`inla`](https://timweiland.github.io/Latte.jl/dev/engines/inla/) | grid / CCD integration over θ (default) | few hyperparameters; you want full marginal densities |
| [`tmb`](https://timweiland.github.io/Latte.jl/dev/engines/tmb/) | MAP plus a Gaussian (delta-method) covariance | sdreport-style point estimates suffice |
| [`hmc_laplace`](https://timweiland.github.io/Latte.jl/dev/engines/hmc_laplace/) | NUTS on the Laplace marginal `L(θ)`, warm-started from TMB | you want faithful θ sampling and can afford it |

Each engine has a regime where it fits and one where it does not:

- TMB fits a single Gaussian at the mode, so it cannot follow a skewed or curved hyperparameter posterior. It is the cheapest, and it is right when that posterior is close to Gaussian.
- INLA integrates a grid over θ and recovers non-Gaussian hyperparameter marginals, but the grid can be biased when the posterior is a curved, skewed ridge.
- HMC-Laplace samples θ directly, which is the most faithful option. It is also the slowest, and stays affordable only when there are few hyperparameters.

The `getting_started` tutorial runs one `surg_mortality` model through all three engines, changing only the entry-point function.

For the references behind each engine: `inla` follows Rue, Martino & Chopin (2009); `tmb` follows the TMB / sdreport approach (Kristensen et al. 2016); `hmc_laplace` follows the embedded-Laplace-plus-HMC method of Margossian et al. (2020).

## One shared result protocol

Every engine returns a result that implements the same accessors, so post-processing code does not care which engine produced it:

- `latent_marginals` and `hyperparameter_marginals` return marginals as Distributions.jl objects. Blocks are addressable by name, e.g. `latent_marginals(result, :u)`.
- `hyperparameter_mode`, `log_marginal_likelihood`, `converged`, `time_elapsed`
- `linear_predictor_marginals` and `observation_marginals` give the linear predictor and the response-scale marginals
- `rand(result, n)` draws a `PosteriorSamples` object
- `derived` computes marginals of nonlinear functionals of the latent field
- `diagnose` runs the PSIS check described below

Hyperparameter marginals come back in natural (prior-declared) space, so no back-transformation is needed. Prefer these accessor functions to the `result.…` fields: under the default compact latent parameterization the augmented blocks are not materialized (`result.linear_predictor_marginals` is `nothing`), and the functions derive or slice them transparently. The [results reference](https://timweiland.github.io/Latte.jl/dev/reference/results/) has the full protocol.

### Knowing when to trust the approximation

`diagnose(result)` runs a Pareto-smoothed importance-sampling (PSIS) check on the inner Laplace approximation `q(x|θ) ≈ p(x|y,θ)` at the hyperparameter mode:

```julia
d = diagnose(r_inla)
d.rel_ess          # relative effective sample size, in (0, 1]
d.pareto_k         # GPD shape parameter
d.interpretation   # :excellent / :acceptable / :unreliable
```

Because all three engines share the inner Laplace, the same diagnostic works uniformly across `inla`, `tmb`, and `hmc_laplace`. It is a guardrail on the inner latent approximation at the mode, not a global certificate that every downstream marginal is correct. The PSIS implementation is vendored from [ParetoSmooth.jl](https://github.com/TuringLang/ParetoSmooth.jl).

## How models compile

You write latent fields with ordinary array indexing — `logN[a, y]` over age and year, or `u[group[i]]` over groups — and `@latte` reads the loop and index structure to build the sparse gradient and Hessian by differentiating each small factor on its own. Work per Newton step then scales with the model's sparsity. Because AD only specialises on the small per-factor functions, per-model first-compile cost stays low. The generic assembly, Newton loop, and hyperparameter-gradient machinery compile once and are reused across models.

For common exponential-family likelihoods (Poisson, Binomial, Bernoulli, Normal, and more) the macro maps the observation block onto hand-coded `ExponentialFamily` observation models from GaussianMarkovRandomFields.jl, and falls back to sparse automatic differentiation for everything else. The fast structured path covers scalar indexed `~` sites inside (nested) loops with a single observed array, plus multivariate-block and element-wise-broadcast priors. Patterns outside that, such as multiple observed symbols, self-referential broadcast priors, or latents without a static array allocation, fall back to a monolithic path that is always correct but slower to compile, with a build-time warning pointing at the indexed-factor form. Correctness never depends on which path runs: Latte checks that the structured version reproduces the monolithic one before using it.

If you already have a Turing/DynamicPPL `@model`, hand it to `latte_from_dppl(m; random=...)` instead of rewriting it; the `@latte` macro is the front door for new models, and this is the handoff path.

## Cross-checking against MCMC

The same `@latte` function body is also forwarded to `DynamicPPL.@model`, exposed via `Latte.dppl_model(name)`, so you can sample the identical model with Turing's NUTS for validation without rewriting it. The SAM tutorial uses exactly this handoff to confirm the iterated-Laplace posterior matches a full-HMC reference across the latent field. This is for validation, not a fallback to MCMC for the actual inference.

## Performance

The [benchmarks page](https://timweiland.github.io/Latte.jl/dev/benchmarks/) compares Latte against R-INLA on the same model with an identical likelihood and prior. Accuracy is scored as the Kolmogorov–Smirnov distance between the two engines' marginals; speed is reported as warm-fit wall-clock.

On small and medium GLMMs the warm-fit multipliers are large — roughly 580x on the Crowder seeds binomial GLMM and 185x on the Scottish lip-cancer Besag model — but that gap is mostly R-INLA's fixed warm-fit overhead on small problems, and it shrinks toward parity as the latent field grows: about 1.4x on the Gaussian Matérn-SPDE "SPDEtoy" benchmark (1680-node shared mesh). Marginals agree closely throughout (max KS 0.034, 0.016, and 0.027 respectively).

Read those numbers with their caveats:

- The speedups are warm-fit only. Cold fits include Julia's first-call compilation, around 7–13 s across these cases.
- Timings are noisy run to run (about 2x).
- Some scenarios are weakly identified, which is a property of the model rather than a bug: the Paraná SPDE field KS of about 0.10 is a variance-limited floor, and the New Haven RW2 temperature trend (max KS 0.16) is the worst-agreeing model in the suite.

Every figure links to a runnable script under `benchmark/` with versions and hardware recorded, and the page invites corrections.

## Documentation

- [Tutorials](https://timweiland.github.io/Latte.jl/dev/tutorials/) — worked end-to-end models across spatial, temporal, and hierarchical settings
- [Benchmarks](https://timweiland.github.io/Latte.jl/dev/benchmarks/) — accuracy and speed against R-INLA on matched models, with the scripts that produced them
- [Reference](https://timweiland.github.io/Latte.jl/dev/reference/) — the `@latte` macro, the results protocol, and lower-level construction

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for the full workflow, and `src/LAYOUT.md` for the architectural reference.

```bash
make setup      # install dependencies
make test       # run the test suite
make format     # format with Runic
```

## Related work

- [DynamicPPL.jl](https://github.com/TuringLang/DynamicPPL.jl) — the probabilistic-programming layer the `@latte` macro builds on; the same model body is also exposed as a DynamicPPL `@model` for the Turing handoff
- [INLA](https://www.r-inla.org/) (R) — reference implementation of the INLA algorithm (Rue, Martino & Chopin 2009)
- [TMB](https://kaskr.github.io/adcomp/_book/Introduction.html) (R / C++) — Template Model Builder
- [tmbstan](https://github.com/kaskr/tmbstan) — HMC on TMB models
- [Stan](https://mc-stan.org/) — recently gained experimental embedded Laplace approximation support for latent Gaussian models (the `laplace_marginal` functions, after Margossian et al. 2020), in the same spirit as INLA and TMB
- [GaussianMarkovRandomFields.jl](https://github.com/timweiland/GaussianMarkovRandomFields.jl) — the latent-model layer Latte builds on
- [ParetoSmooth.jl](https://github.com/TuringLang/ParetoSmooth.jl) — PSIS implementation, vendored for `diagnose`

## Citing

If you use Latte in your work, please cite the package. A software citation with a DOI will appear here once one is attached.
