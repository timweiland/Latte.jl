# # Spatial Survival: Leukemia Hazards Across Districts
#
# Survival analysis — modelling the *time until an event*, with some
# observations only **right-censored** (the patient was still alive when the
# study ended) — is one of the classic domains where INLA shines. This
# tutorial works the canonical example: the **Leukemia** dataset of Henderson
# et al. (2002), 1043 patients in North-West England, each with a survival
# time, a censoring indicator, four covariates, and the **district** they
# lived in. On top of a survival regression we add a **Besag spatial frailty**
# over the 24 districts, so the model can pick up residual geographic variation
# in hazard after adjusting for the covariates.
#
# What makes this a good Latte showcase is that we fit the *same* model **two
# different ways**, and they agree with each other and with R-INLA:
#
# 1. the **Poisson piecewise-exponential trick** — the textbook reformulation
#    of a survival model as Poisson regression, which rides Latte's fast
#    Poisson path with no custom code; and
# 2. a **hand-written Weibull survival likelihood** — a few lines of `logpdf`
#    that handle censoring directly, dropped straight into an `@latte` model.
#
# The first shows that a classic INLA workhorse is a one-liner here; the second
# shows that when you want a parametric likelihood that isn't built in, you
# just *write it down*.

using Latte
using GaussianMarkovRandomFields: BesagModel
using Distributions
using LinearAlgebra
using SparseArrays
using Statistics
using CSV
using DataFrames
using CairoMakie

# ## The data
#
# `status = 1` marks an observed death; `status = 0` a right-censored patient.
# The continuous covariates (age, white-blood-cell count `wbc`, Townsend
# deprivation index `tpi`) are standardized; `sex` is 0/1. The data and the
# district adjacency graph ship with the tutorial (originally from R-INLA's
# `Leuk` example).

data_dir = joinpath(@__DIR__, "data")
leuk = CSV.read(joinpath(data_dir, "leuk.csv"), DataFrame)
edges = CSV.read(joinpath(data_dir, "leuk_graph.csv"), DataFrame)
n_district = maximum(leuk.district)

## Symmetric 0/1 adjacency for the Besag prior.
Is, Js = Int[], Int[]
for r in eachrow(edges)
    push!(Is, r.i, r.j)
    push!(Js, r.j, r.i)
end
W = sparse(Is, Js, ones(length(Is)), n_district, n_district)

@info "Leukemia survival data" patients = nrow(leuk) districts = n_district events = sum(leuk.status) censored = sum(leuk.status .== 0)

# The covariate matrix used by both models (the spatial frailty is added
# separately, indexed by district).
const COVNAMES = ["age", "sex", "wbc", "tpi"]
covmat = hcat(leuk.age_z, Float64.(leuk.sex), leuk.wbc_z, leuk.tpi_z)
covmat[1:4, :]   # first four patients (standardized age, sex, wbc, tpi)

# ## Route 1: the Poisson piecewise-exponential trick
#
# A proportional-hazards survival model with hazard
# ``h_i(t) = h_0(t)\,\exp(\eta_i)`` can be fit *exactly* as a Poisson
# regression. Split each patient's follow-up into time intervals; within an
# interval the baseline hazard is treated as constant. For patient ``i`` in
# interval ``j`` we create one "person-period" record with
#
# * an **exposure** = the time the patient spent at risk in that interval, and
# * an event indicator ``y_{ij} = 1`` only in the interval where the death
#   actually happened (0 everywhere else, and for censored patients).
#
# Then ``y_{ij} \sim \mathrm{Poisson}(\text{exposure}_{ij}\cdot e^{\gamma_j +
# \eta_i})``, where ``\gamma_j`` is the log baseline hazard in interval ``j``.
# That is an ordinary Poisson GLM with a log-exposure offset — so it runs on
# Latte's hand-coded Poisson likelihood with no custom code at all.

## Interval cut points from the quantiles of the observed event times.
event_times = leuk.time[leuk.status .== 1]
K = 12
cuts = quantile(event_times, range(0, 1; length = K + 1))
cuts[1] = 0.0
cuts[end] = maximum(leuk.time) + 1.0

## Expand to person-period records.
pp_district = Int[]
pp_interval = Int[]
pp_exposure = Float64[]
pp_event = Int[]
pp_cov = Vector{Float64}[]
for r in eachrow(leuk)
    for j in 1:K
        lo, hi = cuts[j], cuts[j + 1]
        lo >= r.time && break                       # patient already left follow-up
        exposure = min(r.time, hi) - lo
        exposure <= 0 && continue
        push!(pp_district, r.district)
        push!(pp_interval, j)
        push!(pp_exposure, exposure)
        push!(pp_event, (r.status == 1 && lo < r.time <= hi) ? 1 : 0)
        push!(pp_cov, [r.age_z, Float64(r.sex), r.wbc_z, r.tpi_z])
    end
end
n_pp = length(pp_event)

## Design matrix: one indicator per interval (the piecewise baseline) followed
## by the four covariates. The interval indicators carry the intercept, so no
## separate intercept column is needed.
Xpp = zeros(n_pp, K + 4)
for r in 1:n_pp
    Xpp[r, pp_interval[r]] = 1.0
    Xpp[r, (K + 1):(K + 4)] = pp_cov[r]
end
log_exposure = log.(pp_exposure)
@info "Person-period dataset" rows = n_pp intervals = K

# The model is a plain Poisson regression with the log-exposure offset and a
# Besag frailty indexed by district.

@latte function leuk_poisson(y, X, offset, district, W)
    τ ~ PCPrior.Precision(1.0, α = 0.01)
    β ~ MvNormal(zeros(size(X, 2)), 100.0 * I(size(X, 2)))   # K baselines + 4 covariates
    u ~ BesagModel(W)(τ = τ)
    for r in eachindex(y)
        y[r] ~ Poisson(exp(offset[r] + X[r, :] ⋅ β + u[district[r]]))
    end
end

lgm_pois = leuk_poisson(pp_event, Xpp, log_exposure, pp_district, W)
result_pois = inla(lgm_pois, pp_event; progress = false)

## `latent_marginals(result, :name)` returns the marginals for a named latent
## block — here `:β`, whose last four entries are the covariate
## log-hazard-ratios (the first `K` are the piecewise baselines).
pois_β = latent_marginals(result_pois, :β)
pois_coef = [(mean(pois_β[K + k]), std(pois_β[K + k])) for k in 1:4]
for (nm, (m, s)) in zip(COVNAMES, pois_coef)
    println("  $nm: ", round(m, digits = 3), " ± ", round(s, digits = 3))
end

# ## Route 2: a hand-written Weibull survival likelihood
#
# The Poisson trick leaves the baseline hazard nonparametric. If instead we
# want a *parametric* Weibull baseline, there is no built-in survival
# observation model — so we write the likelihood ourselves. With a Weibull
# proportional-hazards model the cumulative hazard is ``H(t) = t^{\alpha}
# e^{\eta}`` and the hazard is ``h(t) = \alpha t^{\alpha-1} e^{\eta}``. The
# only thing censoring changes is *which* term each observation contributes:
#
# * an observed death contributes the log-density ``\log h(t) - H(t)``;
# * a censored patient contributes the log-survival ``\log S(t) = -H(t)``.
#
# We encode that as a tiny `Distribution` whose `logpdf` branches on the event
# indicator. The Weibull shape ``\alpha`` is an unknown we infer — it becomes a
# hyperparameter via the `log_α` term in the model below.

struct WeibullSurv{T <: Real} <: ContinuousUnivariateDistribution
    η::T
    α::T
    event::Int
end
## Promote so η (a latent-derived dual number) and α (a Float64 hyperparameter)
## share a type when Latte differentiates the likelihood.
function WeibullSurv(η::Real, α::Real, event::Integer)
    ηp, αp = promote(η, α)
    return WeibullSurv{typeof(ηp)}(ηp, αp, Int(event))
end
function Distributions.logpdf(d::WeibullSurv, t::Real)
    logt = log(t)
    log_cumhaz = d.α * logt + d.η
    log_haz = log(d.α) + (d.α - 1) * logt + d.η
    return d.event == 1 ? log_haz - exp(log_cumhaz) : -exp(log_cumhaz)
end
Distributions.minimum(::WeibullSurv) = 0.0
Distributions.maximum(::WeibullSurv) = Inf
Distributions.insupport(::WeibullSurv, t::Real) = t > 0

# The `@latte` model carries an intercept plus the four covariates in `β`, the
# same Besag frailty, and the Weibull shape as a hyperparameter. `@latte` reads
# `log_α` as a hyperparameter and hoists `α = exp(log_α)`, exactly as it would
# for any other likelihood parameter.

@latte function leuk_weibull(t, X, district, status, W)
    log_α ~ Normal(0.0, 1.0)
    τ ~ PCPrior.Precision(1.0, α = 0.01)
    β ~ MvNormal(zeros(size(X, 2)), 100.0 * I(size(X, 2)))   # intercept + 4 covariates
    u ~ BesagModel(W)(τ = τ)
    α = exp(log_α)
    for i in eachindex(t)
        η = X[i, :] ⋅ β + u[district[i]]
        t[i] ~ WeibullSurv(η, α, status[i])
    end
end

Xw = hcat(ones(nrow(leuk)), covmat)   # intercept + covariates
lgm_weib = leuk_weibull(leuk.time, Xw, leuk.district, leuk.status, W)

# This model runs under the default derivative strategy without any extra knobs.
# Here we pass `FiniteDiffStrategy()` purely for speed: with only two
# hyperparameters the finite-difference inner Laplace step is roughly twice as
# fast as the autodiff default and gives identical results. (It's the same knob
# the Tweedie tutorial uses, where the sparse custom-Hessian path needs it.)
result_weib = inla(lgm_weib, leuk.time; diff_strategy = FiniteDiffStrategy(), progress = false)

## `:β` is intercept + 4 covariates, so the covariate effects are entries 2:5.
weib_β = latent_marginals(result_weib, :β)
weib_coef = [(mean(weib_β[1 + k]), std(weib_β[1 + k])) for k in 1:4]   # skip intercept
α_post = result_weib.hyperparameter_marginals.log_α
α_mean = exp(mean(α_post))
for (nm, (m, s)) in zip(COVNAMES, weib_coef)
    println("  $nm: ", round(m, digits = 3), " ± ", round(s, digits = 3))
end
println("  Weibull shape α ≈ ", round(α_mean, digits = 3))

# ## The two routes agree — and match R-INLA
#
# Both formulations recover the same covariate log-hazard-ratios. As an
# external check, the table also shows R-INLA's `weibullsurv` fit (variant 0,
# the proportional-hazards parameterization) on the identical data and priors.
# The numbers line up to two decimals, including the posterior standard
# deviations.

## R-INLA weibullsurv (variant 0) reference: (mean, sd).
rinla_coef = [(0.592, 0.04), (0.079, 0.069), (0.22, 0.033), (0.094, 0.035)]

println("\ncovariate │  Poisson (route 1) │  Weibull (route 2) │  R-INLA")
for (k, nm) in enumerate(COVNAMES)
    a, b, c = pois_coef[k], weib_coef[k], rinla_coef[k]
    println(
        rpad(nm, 9), "│ ", rpad(string(round(a[1], digits = 3), " ± ", round(a[2], digits = 3)), 18),
        "│ ", rpad(string(round(b[1], digits = 3), " ± ", round(b[2], digits = 3)), 18),
        "│ ", round(c[1], digits = 3), " ± ", round(c[2], digits = 3)
    )
end

# A coefficient plot makes the agreement easy to see: each covariate's
# posterior mean ± one standard deviation, for all three fits.

let
    fig = Figure(size = (680, 360))
    ax = Axis(
        fig[1, 1], xlabel = "log hazard ratio", yticks = (1:4, COVNAMES),
        title = "Covariate effects: two Latte routes vs. R-INLA"
    )
    series = [
        ("Poisson trick", pois_coef, :dodgerblue, -0.18),
        ("Weibull likelihood", weib_coef, :firebrick, 0.0),
        ("R-INLA", rinla_coef, :black, 0.18),
    ]
    for (lab, coef, col, dy) in series
        ys = (1:4) .+ dy
        ms = [c[1] for c in coef]
        ss = [c[2] for c in coef]
        errorbars!(ax, ms, ys, ss; direction = :x, color = col, whiskerwidth = 8)
        scatter!(ax, ms, ys; color = col, markersize = 11, label = lab)
    end
    vlines!(ax, 0.0; color = (:gray, 0.5), linestyle = :dash)
    axislegend(ax; position = :rt, framevisible = false)
    fig
end

# All three agree: higher age, higher white-blood-cell count, and greater
# deprivation each raise the hazard; the sex effect is small and its interval
# covers zero. The Weibull route additionally estimates a shape ``\alpha
# \approx 0.59 < 1``, i.e. a hazard that *decreases* over time — survivors of
# the early high-risk period face a steadily lower rate.

# ## The spatial frailty
#
# After the covariates, what does geography add? The Besag frailty ``u`` is a
# log-hazard adjustment per district; ``e^{u}`` is the multiplicative effect on
# the hazard relative to the regional average. We read it off the Weibull fit
# and draw it on the North-West England map.

## The spatial frailty is its own named block `:u`, one entry per district.
weib_u = latent_marginals(result_weib, :u)
frailty = exp.([mean(weib_u[d]) for d in 1:n_district])

mapdf = CSV.read(joinpath(data_dir, "leuk_map.csv"), DataFrame)
polys = Vector{Point2f}[]
poly_region = Int[]
for g in groupby(mapdf, :poly)              # one ring per `poly`; some districts have several
    push!(polys, Point2f.(g.x, g.y))
    push!(poly_region, first(g.region))
end

let
    crange = (minimum(frailty), maximum(frailty))
    fig = Figure(size = (560, 620))
    ax = Axis(
        fig[1, 1], aspect = DataAspect(),
        title = "Posterior spatial frailty  exp(u)  (hazard multiplier)"
    )
    hidedecorations!(ax)
    hidespines!(ax)
    poly!(
        ax, polys; color = frailty[poly_region], colorrange = crange,
        colormap = :balance, strokewidth = 0.5, strokecolor = (:black, 0.4)
    )
    Colorbar(
        fig[1, 2]; colorrange = crange, colormap = :balance,
        label = "relative hazard  exp(u)"
    )
    fig
end

# The frailty is mild but real: a few districts carry a residual excess hazard
# (and others a deficit) that the covariates do not explain — exactly the kind
# of structure a spatial random effect is there to absorb, and which would
# otherwise leak into biased covariate estimates or overconfident intervals.

# ## Takeaway
#
# * **Survival analysis is first-class in Latte.** The Poisson
#   piecewise-exponential trick turns a proportional-hazards model into a Poisson
#   GLM that runs on the fast built-in likelihood — no custom code.
# * **When you want a parametric likelihood that isn't built in, write it.** A
#   five-line `WeibullSurv` `logpdf`, handling censoring by branching between
#   the density and the survival function, drops straight into `@latte` with
#   full posterior uncertainty over the shape, the coefficients, and the field.
# * **Both routes agree with each other and with R-INLA** to two decimals, and
#   the spatial frailty rides for free on the same GMRF machinery used
#   throughout the rest of these tutorials.
#
# ## References
#
# - Henderson, R., Shimakura, S. & Gorst, D. (2002). Modeling spatial
#   variation in leukemia survival data. *JASA* 97(460), 965–972.
# - Rue, H., Martino, S. & Chopin, N. (2009). Approximate Bayesian inference
#   for latent Gaussian models using INLA. *JRSS-B* 71(2), 319–392.
