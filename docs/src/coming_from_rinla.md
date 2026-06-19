```@raw html
---
pageClass: latte-wide rinla-compare
---
```

# [Coming from R-INLA](@id coming-from-rinla)

This guide is for people who already fit latent Gaussian models with
[R-INLA](https://www.r-inla.org/) and want to do the same work in Julia with
Latte. R-INLA is the original, foundational implementation of Integrated Nested
Laplace Approximation, and the methodology here is the same: a latent Gaussian
field, a handful of hyperparameters, a Laplace approximation to the latent
posterior, and numerical integration over the hyperparameters. What changes is
the interface: how you describe the model, and how you read the results back
out.

## Mental model

In R-INLA you write a formula with `f(...)` terms for the structured random
effects, pick a `family`, and tune the fit through `control.*` arguments; the
`inla()` call ties those together into a single object.

Latte splits that into two steps. First you write an `@latte` model block, where
every prior is stated explicitly (the hyperparameter priors included), the
structured terms come from `GaussianMarkovRandomFields.jl`, and the likelihood
goes on the `~` line of a loop over the data. That block returns a
[latent Gaussian model](@ref latent-gaussian-models), which you then hand to an inference
engine. [`inla`](@ref) is the direct counterpart to R-INLA's `inla()`:

```julia
lgm = my_model(data...)        # the @latte block returns the model
result = inla(lgm, y)          # run INLA on it
```

The same `lgm` also runs under `tmb` or `hmc_laplace` (more on those later), so
defining a model and choosing an engine stay separate concerns.

## Worked examples

Each example puts the R-INLA formulation next to the Latte one on a dataset you
already know. The two share the same priors and likelihood, so the only
difference is the interface. The [translation table](#Translation-table) below
maps the pieces term by term.

### Scotland lip cancer: Besag disease mapping

Per-district lip-cancer counts over 56 Scottish districts: a Poisson rate with a
log-offset, one covariate, and a spatial Besag-ICAR effect. See also the
[spatial disease mapping tutorial](tutorials/disease_mapping_spatial.md).

```@raw html
<div class="rinla-pair">
```

```r
scot$x_scaled <- scot$X / 10

formula <- Counts ~ 1 + x_scaled + f(Region,
    model = "besag",
    graph = graph_path,
    scale.model = TRUE,
    hyper = list(prec = list(prior = "pc.prec",
                             param = c(params$pc_U, params$pc_alpha)))
)

result <- inla(formula,
    data = scot,
    family = "poisson",
    E = scot$E,
    control.fixed = list(prec = 1.0e-2, prec.intercept = 1.0e-2),
    control.inla = list(strategy = strategy, int.strategy = "grid"),
)
```

```julia
@latte function scotland_model(y, log_E, x_scaled, W, n_d)
    τ ~ PCPrior.Precision(1.0, α = 0.01)
    fixed ~ MvNormal(zeros(2), 100.0 * I(2))
    u ~ BesagModel(W)(τ = τ)
    for i in eachindex(y)
        η_i = log_E[i] + fixed[1] + fixed[2] * x_scaled[i] + u[i]
        y[i] ~ Poisson(exp(η_i); check_args = false)
    end
end

lgm = scotland_model(data.y, data.log_E, data.x_scaled, data.W, data.n)
result = inla(lgm, data.y; progress = false)
```

```@raw html
</div>
```

Note the `scale.model = TRUE` in the R formula: it matches Latte's Besag
default (see [Migration notes](#Migration-notes)).

### Epil: hierarchical Poisson GLMM with IID effects

Seizure counts for 59 patients over 4 visits (236 observations): a Poisson GLMM
with fixed effects plus subject-level and observation-level IID random effects.

```@raw html
<div class="rinla-pair">
```

```r
epil$log_base4 <- log(epil$Base / 4)
epil$trt_logbase4 <- epil$Trt * epil$log_base4
epil$log_age <- log(epil$Age)
epil$obs_id <- seq_len(n)

formula <- y ~ 1 + log_base4 + Trt + trt_logbase4 + log_age + V4 +
    f(Ind, model = "iid",
        hyper = list(prec = list(prior = "pc.prec",
                                 param = c(params$pc_U, params$pc_alpha)))) +
    f(obs_id, model = "iid",
        hyper = list(prec = list(prior = "pc.prec",
                                 param = c(params$pc_U, params$pc_alpha))))

result <- inla(formula,
    data = epil,
    family = "poisson",
    control.fixed = list(prec = 1.0e-2, prec.intercept = 1.0e-2),
    control.inla = list(strategy = strategy, int.strategy = "grid"),
)
```

```julia
@latte function epil_model(
        y, log_base4, trt, trt_logbase4, log_age, v4,
        ind, n_subject, n_obs,
    )
    τ_subj ~ PCPrior.Precision(1.0, α = 0.01)
    τ_obs ~ PCPrior.Precision(1.0, α = 0.01)
    fixed ~ MvNormal(zeros(6), 100.0 * I(6))
    b_subject ~ IIDModel(n_subject)(τ = τ_subj)
    b_obs ~ IIDModel(n_obs)(τ = τ_obs)
    for k in eachindex(y)
        η_k = fixed[1] + fixed[2] * log_base4[k] + fixed[3] * trt[k] +
            fixed[4] * trt_logbase4[k] + fixed[5] * log_age[k] + fixed[6] * v4[k] +
            b_subject[ind[k]] + b_obs[k]
        y[k] ~ Poisson(exp(η_k); check_args = false)
    end
end

lgm = epil_model(
    data.y, data.log_base4, data.trt, data.trt_logbase4, data.log_age, data.v4,
    data.ind, data.n_subject, data.n,
)
result = inla(lgm, data.y; progress = false)
```

```@raw html
</div>
```

The formula's two `f(..., model = "iid")` terms map onto the two `IIDModel`
blocks; each row indexes its subject effect by `ind[k]` and its
observation-level effect by `k`.

### SPDEtoy: Matérn SPDE geostatistics

A continuous spatial Gaussian field over 2D coordinates via a Matérn SPDE
(`alpha = 2`) with an intercept and Gaussian observation noise. The two engines
use the same mesh / FEM discretization, so only the inference differs. See the
[SPDE tutorial](tutorials/spatial_spde.md).

```@raw html
<div class="rinla-pair">
```

```r
spde <- inla.spde2.pcmatern(
    mesh, alpha = 2,
    prior.range = c(params$range_U, params$range_p),
    prior.sigma = c(params$sigma_field_U, params$sigma_field_p)
)

A <- inla.spde.make.A(mesh, loc = coords)
idx <- inla.spde.make.index("spatial", spde$n.spde)
stk <- inla.stack(
    data = list(y = df$y),
    A = list(A, 1),
    effects = list(idx, list(Intercept = rep(1, n))),
    tag = "est"
)

res <- inla(
    y ~ 0 + Intercept + f(spatial, model = spde),
    data = inla.stack.data(stk),
    family = "gaussian",
    control.predictor = list(A = inla.stack.A(stk)),
    control.family = list(hyper = list(prec = list(
        prior = "pc.prec", param = c(params$sigma_obs_U, params$sigma_obs_alpha)
    ))),
    control.fixed = list(prec.intercept = params$prec_intercept),
    control.inla = list(int.strategy = "grid")
)
```

```julia
@latte function spdetoy_model(y, base_matern, A_obs, p)
    σ ~ PCPrior.Sigma(p.sigma_obs_U; α = p.sigma_obs_alpha)
    τ_matern ~ PCPrior.Precision(p.sigma_field_U; α = p.sigma_field_p)
    range_matern ~ PCPrior.Range(p.range_U; p = p.range_p)
    β ~ MvNormal(zeros(1), (1 / p.prec_intercept) * I(1))
    field ~ base_matern(τ = τ_matern, range = range_matern)
    η = β[1] .+ A_obs * field
    for i in eachindex(y)
        y[i] ~ Normal(η[i], σ)
    end
end

A_obs = evaluation_matrix(disc, data.coords)
base_matern = MaternModel(disc; smoothness = 0)   # smoothness = 0 ⇒ ν = 1 ⇒ alpha = 2
lgm = spdetoy_model(data.y, base_matern, A_obs, p)
# Gaussian observations ⇒ the latent posterior is exactly Gaussian
result = inla(lgm, data.y; progress = false,
              latent_marginalization_method = GaussianMarginal())
```

```@raw html
</div>
```

The R-INLA `inla.stack` and projector matrix `A` correspond to building the
observation/evaluation matrix `A_obs` and applying it as `A_obs * field`.
Because the observations are Gaussian here, the latent posterior is exactly
Gaussian, so `GaussianMarginal()` is the natural choice.

### Tokyo rainfall: RW2 temporal smoothing

The number of rainy days per calendar day across 366 days: a binomial-logit
model with a second-order random walk (RW2) smooth over time.

```@raw html
<div class="rinla-pair">
```

```r
formula <- y ~ -1 + f(time,
    model = "rw2",
    scale.model = FALSE,
    hyper = list(prec = list(prior = "pc.prec",
                             param = c(params$pc_U, params$pc_alpha)))
)

result <- inla(formula,
    data = tokyo,
    family = "binomial",
    Ntrials = tokyo$n,
    control.inla = list(strategy = strategy, int.strategy = "grid"),
)
```

```julia
@latte function tokyo_model(y, n_trials, M)
    τ ~ PCPrior.Precision(1.0, α = 0.01)
    x ~ M(τ = τ)
    for t in eachindex(y)
        y[t] ~ Binomial(n_trials[t], 1 / (1 + exp(-x[t])); check_args = false)
    end
end

M = RW2SumOnly(data.n)   # RW2 with sum-to-zero only (matches R-INLA's rw2 default)
lgm = tokyo_model(data.y, data.n_trials, M)
result = inla(lgm, data.y; progress = false)
```

```@raw html
</div>
```

Here `RW2SumOnly` is a small local `LatentModel` wrapper that keeps only the
sum-to-zero constraint (dropping the slope null-space constraint), so its prior
matches R-INLA's `rw2`. The R formula's `-1` (no intercept) is reflected by the
model having no intercept term.

## Translation table

### Latent model terms (`f(s, model = ...)`)

In R-INLA these are `f(...)` terms in the formula. In Latte they are model
objects from `GaussianMarkovRandomFields.jl`, written as
`name ~ Model(...)(hyper = ...)` inside the `@latte` block. The model object is
callable: `IIDModel(n)(τ = ...)` returns the GMRF for the given
hyperparameters.

| R-INLA | Latte | Note |
|---|---|---|
| `f(i, model = "iid")` | `u ~ IIDModel(n)(τ = τ_u)` | `IIDModel(n; constraint = :sumtozero)` for identifiability alongside an intercept. |
| `f(t, model = "rw1")` | `f ~ RWModel{1}(n)(τ = τ)` (alias `RW1Model`) | Order-1 random walk. |
| `f(t, model = "rw2")` | `f ~ RWModel{2}(n)(τ = τ)` (alias `RW2Model`) | Order-2 random walk. |
| `f(t, model = "ar1")` | `f ~ AR1Model(n)(τ = τ, ρ = ρ)` | Two hyperparameters: precision `τ` and correlation `ρ` (PACF parameterization). |
| `f(s, model = "besag", graph = g)` | `u ~ BesagModel(W)(τ = τ)` | `W` is the adjacency matrix; intrinsic ICAR, sum-to-zero per connected component. |
| `f(s, model = "besagproper")` | `BesagModel(W; regularization = ...)` | The `regularization` diagonal term plays the role of the proper variant. |
| `f(s, model = "bym2", graph = g)` | `x ~ BYM2Model(W)(τ = τ, φ = φ)` | Riebler (2016) parameterization: `τ` total precision, `φ` mixing proportion. `φ` takes its own prior; `PCPrior.BYMProportion(Q_scaled, U; α)` is the PC prior for the mixing proportion, matching R-INLA's BYM2 `phi` `pc` prior. |
| separable space×time (`group =`, `control.group`) | `δ ~ SeparableModel(time_component, space_component)(...)` | Kronecker-separable; see the [spatio-temporal tutorial](tutorials/spatio_temporal_separable.md). |
| fixed effects / `f(..., model = "linear")` | `β ~ MvNormal(...)` directly, or `FixedEffectsModel` | Linear/fixed effects go straight into the latent Gaussian. |

Hyperparameters declared in the `@latte` block (`τ`, `ρ`, `φ`) live in natural
space, not the internal log/logit scale R-INLA reports for `theta`.

### Likelihood (`family =`, `control.family`)

The likelihood is written on the `~` line of the observation loop; there is no
separate `family` argument.

| R-INLA | Latte | Note |
|---|---|---|
| `family = "poisson"`, `E =` offset | `y[i] ~ Poisson(exp(η[i]))` or `Poisson(E[i] * exp(η[i]))` | The log link is explicit via `exp`; the exposure multiplies the rate. |
| `family = "binomial"`, `Ntrials =` | `r[i] ~ Binomial(n[i], logistic(η[i]))` | `logistic` (from StatsFuns) is the inverse-logit link. |
| `family = "gaussian"`, `control.family = list(hyper = list(prec = ...))` | `y[i] ~ Normal(η[i], σ)`, with `σ` (or a precision) declared as a hyperparameter | Observation noise becomes a model hyperparameter. |
| custom `inla.rgeneric` / nonstandard family | a `struct D <: ContinuousUnivariateDistribution` + `Distributions.logpdf(d::D, x)` | `@latte` only needs `logpdf`; see the [Tweedie tutorial](tutorials/tweedie_insurance.md). |
| `family = "weibullsurv"` + `inla.surv(time, event)` | Route 1: Poisson piecewise-exponential trick; Route 2: a custom `WeibullSurv` `logpdf` | See the [spatial survival tutorial](tutorials/spatial_survival_leukemia.md). |
| multiple families (stacked `family = c(...)`) | one observation loop per family, each `~` writing into the shared latent field | Latte groups them into a `CompositeObservationModel`. |

Offsets work a little differently. There is no `E =` argument, so you fold the
exposure straight into `η`, either as a multiplicative `Poisson(E[i] * exp(η[i]))`
or as the equivalent additive `log_E[i]` term inside `η`, as in the Scotland
example above.

### Priors

| R-INLA | Latte | Note |
|---|---|---|
| `hyper = list(prec = list(prior = "pc.prec", param = c(U, a)))` | `τ ~ PCPrior.Precision(U, α = a)` | Calibrates `P(σ > U) = α` (default `α = 0.05`); `PCPrior.Precision(λ)` gives the direct λ form. |
| `prior = "pc.range"` (SPDE) | `range ~ PCPrior.Range(ρ0, p = ..., dim = 2)` | Calibrates `P(ρ < ρ0) = p`. |
| fixed-effect prior `control.fixed = list(prec = ...)` | `β ~ MvNormal(zeros(k), Σ)` | e.g. `MvNormal(zeros(1), 100.0 * I(1))` for a vague intercept. |
| generic hyperparameter prior | any `Distributions.jl` prior on the natural-space hyperparameter | e.g. `τ ~ Gamma(2, 1)`, `α ~ LogNormal(0, 1)`. |

### SPDE

| R-INLA | Latte | Note |
|---|---|---|
| `inla.mesh.2d(...)` + `inla.spde2.pcmatern(...)` | `MaternModel(obs_points; smoothness = s)`, then `field ~ base_matern(τ = τ, range = range)` | Builds the FEM mesh from the points (loads `Ferrite`, `FerriteGmsh`, `Gmsh`, `LibGEOS`). |
| `alpha = 2` in `inla.spde2.pcmatern` | `smoothness = 0` | Convention: `smoothness = s` ⇒ `ν = s + 1`, `α = ν + d/2`; so in 2D `smoothness = 0` ⇒ `ν = 1` ⇒ `α = 2`. |
| barrier model (`inla.barrier.pcmatern`) | `BarrierModel(disc; range_fraction = ..., barrier_cells = ...)` | Non-stationary ν = 1 Matérn; with no barriers it reduces to `MaternModel(disc; smoothness = 0)`. |

### Inference, results, sampling, prediction

| R-INLA | Latte | Note |
|---|---|---|
| `r <- inla(formula, family, data, ...)` | `r = inla(lgm, y)` | `lgm` is the model returned by the `@latte` block. |
| `control.inla(int.strategy = "grid"/"ccd"/"eb")` | `inla(lgm, y; exploration_strategy = GridExplorationStrategy()` / `CCDExplorationStrategy()` / `INLAGridStrategy())` | How the hyperparameter posterior is integrated. The default `AutoExplorationStrategy()` uses a grid for `D ≤ 2` and CCD for `D ≥ 3`. |
| `control.inla(strategy = "gaussian"/"simplified.laplace"/"laplace")` | `inla(lgm, y; latent_marginalization_method = GaussianMarginal()` / `SimplifiedLaplace()` / default) | The latent posterior approximation. The default adapts per node; pass `GaussianMarginal()` when the observations are Gaussian. |
| `r$summary.fixed` | `latent_marginals(r, :β)` | Marginals for the named latent block. |
| `r$summary.random` | `latent_marginals(r, :u)` | Same accessor, keyed by the random-effect name. |
| `r$summary.hyperpar` | `hyperparameter_marginals(r)` or `hyperparameter_marginals(r, :τ)` | Reported on the **natural** declared scale. |
| `inla.zmarginal` / `$0.025quant` | `mean(m)`, `quantile(m, 0.025)`, `quantile(m, 0.975)` | On any marginal object. |
| `r$summary.linear.predictor` | `linear_predictor_marginals(r)` | Derived `η`. |
| `inla.make.lincombs(...)` / `r$summary.lincomb.derived` | `linear_combinations(r, A)`, or named `linear_combinations(r; β = ..., field = A)` | Affine functionals of the latent field (contrasts, sums, projections). |
| `inla.posterior.sample(n, r)` | `rand(r, n)` | `n` joint posterior draws; `rand(r, n; include_y = true)` also draws predictive `y`. |
| `NA` in the response | `missing` in `y` | `inla(lgm, y)` fits to the observed subset and treats `missing` as to-predict. |
| predicted response at `NA` rows | `predicted_marginals(r)` | Marginals at the `missing` positions. |
| `control.predictor(compute = TRUE, link = ...)` / fitted values | `linear_predictor_marginals(r)`, `observed_marginals(r)` / `observation_marginals(r)` | `η` marginals at observed rows; `observation_marginals` maps `η` through the inverse link onto the response scale. |
| `control.compute = list(dic, waic, cpo)` | `inla(lgm, y; accumulators = (DICStrategy(), MarginalLogLikelihoodStrategy(), WAICStrategy(), CPOStrategy()))` | This tuple is the default; results land in `r.accumulators`. |
| `r$mlik` | `MarginalLogLikelihoodStrategy()` / `log_marginal_likelihood(r)` | Marginal likelihood for model selection / BMA. |
| existing Turing `@model` | `latte_from_dppl(model; random = (:β, :u))` | Adapter to build an `lgm` from a plain DynamicPPL model. |

## Reading the results

R-INLA returns summary data frames. Latte instead hands back marginal
distribution objects that implement the `Distributions.jl` interface, so you call
`mean`, `median`, `std`, `quantile`, and `rand` on them directly, or
`summary_df(...)` for a table.

```julia
# R: result$summary.fixed
β = latent_marginals(result, :fixed)
mean(β[1])                      # posterior mean of the intercept
quantile.(β[2], (0.025, 0.975))  # 95% credible interval of a coefficient

# R: result$summary.random
u = latent_marginals(result, :u)

# R: result$summary.hyperpar  (Latte reports on the natural scale, e.g. τ itself)
hp = hyperparameter_marginals(result, :τ)
mean(hp)
median(hp)

# R: inla.posterior.sample(1000, result)
draws = rand(result, 1000)      # 1000 joint posterior draws (NamedTuple each)
```

Because the hyperparameters are reported on the natural declared scale (`τ`,
`ρ`, `φ`, `σ`, range), there is no back-transform step from an internal `theta`.

### Prediction

Two idioms cover most prediction needs. To predict new rows of an existing
model, put `missing` in `y` at those positions; `inla(lgm, y)` then fits to the
observed subset and exposes the predicted rows through `predicted_marginals(r)`,
the analogue of `NA` rows in an R-INLA response. To predict at new spatial or
temporal locations, build an evaluation matrix `A_pred` at the new coordinates
and push the field through it with `linear_combinations(r; field = A_pred)`, much
as you would project an SPDE field onto a prediction mesh with `inla.spde.make.A`.

## What's the same

A natural question when porting a model is whether the numbers line up. The
inference is the same INLA methodology you already rely on: a Laplace
approximation of the latent posterior, numerical integration over the
hyperparameters, and the same family of latent structures and PC priors. The
[benchmarks page](benchmarks/index.md) makes this concrete, running Latte and
R-INLA on identical model, likelihood, and prior specifications and reporting the
Kolmogorov–Smirnov distance between their marginals alongside wall-clock timings.

The [validation page](validation/index.md) is a separate check. It reports
Simulation-Based Calibration (SBC) for each engine, asking whether the inference
is correct on its own terms (PIT and rank uniformity against simultaneous ECDF
bands) rather than whether it matches R-INLA.

## What the Julia ecosystem adds

A few things come for free from building on Julia. None of them change how INLA
works; they are conveniences that happen to be easy here.

- Custom likelihoods are just a `logpdf`. Any distribution you can write as a
  `struct D <: ContinuousUnivariateDistribution` with a `Distributions.logpdf(d::D, x)`
  method can go on the `~` line, and `@latte` asks for nothing else. The
  [Tweedie](tutorials/tweedie_insurance.md) and
  [spatial survival](tutorials/spatial_survival_leukemia.md) tutorials work
  through examples.
- The same `lgm` runs under more than one engine: [`inla`](@ref engine-inla),
  the TMB-style Laplace MAP in [`tmb`](@ref engine-tmb), or HMC-on-Laplace in
  [`hmc_laplace`](@ref engine-hmc-laplace). That helps for cross-checking, or for
  reaching for a sampler when a grid struggles to integrate the hyperparameter
  posterior.
- The marginals are ordinary `Distributions.jl` objects, so they drop straight
  into plotting and downstream code without conversion.
- If you already keep a DynamicPPL `@model`, `latte_from_dppl(model; random = (...))`
  turns it into an `lgm`, and `diagnose(...)` confirms the latent/hyperparameter
  split came out as you intended (see the
  [Turing handoff tutorial](tutorials/turing_handoff.md)).

## Migration notes

A few defaults and conventions differ between the two interfaces, and they are
worth knowing when you port a model and want the numbers to line up.

- Variance scaling defaults differ. Latte's `BesagModel` and `BYM2Model`
  normalize their variance in the Sørbye–Rue sense by default
  (`normalize_var = true`), whereas R-INLA leaves `scale.model = FALSE`. A `τ`
  prior only means the same thing across the two engines once the scaling
  matches, so set R-INLA's `scale.model = TRUE` (as in the Scotland example);
  otherwise the precision marginal can drift noticeably even when the field
  itself agrees. The random-walk models go the other way, unscaled by default,
  so pass `RWModel{k}(n; scale_model = true)` to match a scaled R-INLA RW.
- The SPDE smoothness argument is offset from R-INLA's `alpha`.
  `MaternModel(...; smoothness = s)` means `ν = s + 1` (and `α = ν + d/2`), so
  R-INLA's common `alpha = 2` (ν = 1 in 2D) corresponds to `smoothness = 0`,
  not `smoothness = 2`.
- Hyperparameters are declared and reported on the natural scale (`τ`, `ρ`, `φ`,
  `σ`, range), not the internal log/logit scale R-INLA uses for `theta`, so there
  is no `summary.hyperpar` back-transform to do.
- Shared and multiple-likelihood effects are written out by hand. You share an
  effect by naming the same latent block in more than one `η` expression; there
  is no `copy` or `replicate` keyword, and a scaled shared effect (`βᵢ · u`) has
  no direct equivalent. For several likelihoods, write one observation loop per
  family over the shared field, and Latte composes them into a single
  `CompositeObservationModel`.
- The first `inla(...)` call in a fresh session pays a one-time compilation cost
  of a few seconds, after which fits in that session are fast; the cost is
  per-session, not per-fit. If you ship Latte inside an application, calling
  `Latte.warmup(model, y; random = ...)` from your package's precompile workload
  folds that cost into precompilation. SPDE models also need the FEM extension
  packages (`Ferrite`, `FerriteGmsh`, `Gmsh`, `LibGEOS`) loaded.
