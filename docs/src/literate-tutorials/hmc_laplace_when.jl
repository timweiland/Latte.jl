# # When to sample the hyperparameters: HMC-Laplace vs INLA
#
# INLA's speed comes from a deliberate shortcut. The latent field is
# integrated out with a Laplace (Gaussian) approximation, and the
# *hyperparameters* are integrated out with a small, **deterministic** set
# of design points — a grid for one or two hyperparameters, a Central
# Composite Design (CCD) for three or more. That design is laid out on an
# ellipsoid scaled by the curvature (Hessian) of the posterior at its mode.
#
# This is exact when the hyperparameter posterior is approximately Gaussian
# in the transformed (working) space — which it very often is, and then INLA
# is both fast and accurate. But the assumption can fail. When the
# hyperparameter posterior is a **curved, skewed ridge**, a symmetric
# ellipsoidal design can't follow the curve, and the resulting marginals are
# biased — typically by truncating the skewed tails.
#
# `hmc_laplace` keeps the inner Laplace approximation for the latent field
# but replaces the deterministic hyperparameter design with **NUTS**: it
# samples the hyperparameter posterior directly. It is more expensive than
# INLA (though far cheaper than full MCMC, since the latent field is still
# integrated out analytically), and in exchange it tracks whatever shape the
# hyperparameter posterior actually has.
#
# This tutorial builds a model whose hyperparameter posterior is genuinely
# non-Gaussian, shows that INLA-CCD is biased there, confirms HMC-Laplace is
# not — against a validated gold-standard MCMC reference — and finishes with
# a decision guide.

# ## A model with a curved hyperparameter posterior
#
# An AR(1) latent process is a clean way to manufacture this geometry. It
# has two hyperparameters that trade off against each other:
#
# - `τ_ar` — the precision (inverse variance) of the innovations, and
# - `ρ` — the lag-1 correlation, with `0 < ρ < 1`.
#
# On short or noisy series the data constrain the *overall smoothness* of
# the path well, but not the *split* between "high precision, high
# correlation" and "low precision, low correlation". The result is a curved
# ridge in `(τ_ar, ρ)` space, made worse by `ρ` piling up against its upper
# boundary at 1. We add an IID observation-level effect (a third
# hyperparameter `τ_iid`) so that INLA uses its CCD design rather than a
# dense grid — CCD is where the ellipsoidal-design assumption bites.

using Latte
using Distributions
using GaussianMarkovRandomFields: AR1Model, IIDModel
using Turing
using LinearAlgebra
using Random, Statistics, Printf
using DataFrames
using CairoMakie

# The `@latte` model below is what `inla` and `hmc_laplace` consume. The PC
# prior on `ρ` shrinks toward the base model `ρ = 0`; the PC priors on the
# precisions shrink toward infinite precision (zero variance).
@latte function ar1_counts(y, n)
    τ_ar ~ PCPrior.Precision(1.0, α = 0.01)
    ρ ~ PCPrior.AR1Correlation(0.7; α = 0.1, positive_only = true)
    τ_iid ~ PCPrior.Precision(1.0, α = 0.01)
    f ~ AR1Model(n)(τ = τ_ar, ρ = ρ)
    u ~ IIDModel(n)(τ = τ_iid)
    for i in eachindex(y)
        y[i] ~ Poisson(exp(f[i] + u[i]); check_args = false)
    end
end

# Simulate a moderately persistent series with relatively few counts, so the
# hyperparameter posterior is genuinely uncertain.
Random.seed!(20260606)
n = 40
ρ_true, σ_ar = 0.88, 0.5
f_true = zeros(n)
f_true[1] = randn() * σ_ar / sqrt(1 - ρ_true^2)
for i in 2:n
    f_true[i] = ρ_true * f_true[i - 1] + randn() * σ_ar
end
η = f_true .+ randn(n) * 0.2 .- mean(f_true) .+ 1.0
y = rand.(Poisson.(exp.(η)))
@printf("simulated %d time points, total count %d, max %d\n", n, sum(y), maximum(y))

lgm = ar1_counts(y, n)

# ## Two ways to integrate out the hyperparameters
#
# Same model object, two engines. `inla` uses the deterministic CCD design;
# `hmc_laplace` samples the hyperparameters with NUTS, warm-started from the
# Laplace approximation at the mode.
hp = (:τ_ar, :ρ, :τ_iid)

# A warm-up pass compiles both engines so the runtimes below are a fair
# comparison rather than a measurement of first-call compilation.
inla(lgm, y; progress = false)
hmc_laplace(lgm, y; n_samples = 10, n_warmup = 10, diff_strategy = FiniteDiffStrategy(), rng = MersenneTwister(0))

t_inla = @elapsed result_inla = inla(lgm, y; progress = false)
t_hmc = @elapsed result_hmc = hmc_laplace(
    lgm, y; n_samples = 2000, n_warmup = 1000,
    diff_strategy = FiniteDiffStrategy(), rng = MersenneTwister(1),
)
@printf("INLA used %d CCD design points\n", length(result_inla.exploration.grid_points))
@printf("HMC-Laplace: converged=%s, divergences=%d\n", converged(result_hmc), divergences(result_hmc))
@printf("runtime — INLA: %.2f s     HMC-Laplace: %.1f s   (≈%.0f× INLA)\n", t_inla, t_hmc, t_hmc / t_inla)

# The two engines expose their hyperparameter posteriors differently. INLA
# returns a *distribution object* per hyperparameter — a spline fit to the
# marginal, with an honest `pdf` / `cdf` / `quantile` — so we read it directly
# instead of sampling it. HMC-Laplace returns draws.
inla_marginal(nm) = getproperty(result_inla.hyperparameter_marginals, nm)
hmc_chain = chain(result_hmc)
hmc_draws = Dict(nm => vec(hmc_chain[nm].data) for nm in hp)

# Already the two disagree on `ρ`, the correlation — most visibly in the
# upper tail, i.e. *how persistent the process could plausibly be*:
for nm in hp
    mi = inla_marginal(nm)
    @printf(
        "%-6s  INLA median=%7.3g (q97.5=%8.3g)   HMC median=%7.3g (q97.5=%8.3g)\n",
        nm, quantile(mi, 0.5), quantile(mi, 0.975),
        median(hmc_draws[nm]), quantile(hmc_draws[nm], 0.975),
    )
end

# ## Who is right? A gold-standard reference
#
# To adjudicate we need the *exact* hyperparameter posterior, from MCMC. An
# `@latte` model is itself a DynamicPPL model — `Latte.dppl_model` hands it
# straight to Turing, no rewriting. For *this* model there is a sampling
# subtlety worth knowing, though: a **centered** hierarchical scale (drawing
# the latent directly at precision `τ`) gives NUTS a funnel geometry that mixes
# poorly for a weakly-identified variance component — you would see it as a low
# ESS and `R̂ > 1`. So for a pristine reference we write the model in
# **non-centered** form (unit-normal innovations scaled by `σ = 1/√τ`): the
# same statistical model, in a geometry NUTS samples cleanly. INLA and
# HMC-Laplace need no such care — they marginalize the latent analytically
# rather than sampling it.
@model function ar1_gold(y, n)
    τ_ar ~ PCPrior.Precision(1.0, α = 0.01)
    ρ ~ PCPrior.AR1Correlation(0.7; α = 0.1, positive_only = true)
    τ_iid ~ PCPrior.Precision(1.0, α = 0.01)
    σ, σu = 1 / sqrt(τ_ar), 1 / sqrt(τ_iid)
    z ~ filldist(Normal(), n)
    zu ~ filldist(Normal(), n)
    f = Vector{typeof(σ * ρ)}(undef, n)
    f[1] = z[1] * σ / sqrt(1 - ρ^2)
    for i in 2:n
        f[i] = ρ * f[i - 1] + z[i] * σ
    end
    u = zu .* σu
    for i in 1:n
        y[i] ~ Poisson(exp(f[i] + u[i]); check_args = false)
    end
    return (f = f,)
end

gold_model = ar1_gold(y, n)
Random.seed!(7)
gold_chain = sample(gold_model, NUTS(1000, 0.95), MCMCThreads(), 1500, 2; progress = false)
gold_draws = Dict(nm => vec(Array(gold_chain[nm])) for nm in hp)

# **Validate the reference before trusting it** — healthy `R̂` / ESS, plus a
# coupling check: the sampled innovation scale `σ_ar = 1/√τ_ar` should track the
# realized spread of the sampled latent path `f`. A chain that mixed poorly
# would show this correlation near zero.
gold_stats = DataFrame(summarystats(gold_chain))
for nm in hp
    idx = findfirst(==(nm), gold_stats.parameters)
    @printf("gold %-6s  R̂=%.3f  ESS=%6.0f\n", nm, gold_stats.rhat[idx], gold_stats.ess_bulk[idx])
end
gold_f = [g.f for g in vec(generated_quantities(gold_model, MCMCChains.get_sections(gold_chain, :parameters)))]
coupling = cor(1 ./ sqrt.(gold_draws[:τ_ar]), [std(fp) for fp in gold_f])
@printf("gold coupling cor(σ_ar, sd(f)) = %.3f\n", coupling)

# ## The verdict
#
# With a trustworthy reference we can score both engines. The
# Kolmogorov–Smirnov distance to the gold measures how far each engine's
# hyperparameter marginal is from the truth.
## KS distance to the gold, evaluated at the gold draws. INLA contributes its
## analytic spline `cdf` directly (no sampling); HMC its empirical CDF.
function ks_to_gold(engine_cdf, gold)
    g = sort(gold)
    m = length(g)
    return maximum(abs(engine_cdf(g[i]) - i / m) for i in 1:m)
end
ecdf_of(s) = let ss = sort(s), n = length(s)
    x -> count(<=(x), ss) / n
end

println("\nKolmogorov–Smirnov distance to the gold standard (smaller is better):")
for nm in hp
    ks_inla = ks_to_gold(x -> cdf(inla_marginal(nm), x), gold_draws[nm])
    ks_hmc = ks_to_gold(ecdf_of(hmc_draws[nm]), gold_draws[nm])
    @printf("  %-6s  INLA-CCD = %.3f      HMC-Laplace = %.3f\n", nm, ks_inla, ks_hmc)
end

# Across all three hyperparameters HMC-Laplace is several times closer to the
# truth. The reason is geometric, and worth seeing directly.

# ## Seeing the banana
#
# Plot the joint posterior of `(log τ_ar, ρ)`. The gold-standard samples
# trace a **curved, skewed ridge**: low precision goes with high correlation
# and vice versa, and the ridge bends. Overlaid are INLA's CCD design
# points — a small, symmetric ellipsoidal cloud centered at the mode. It
# simply cannot reach up the curved arm of the ridge toward `ρ → 1`, which is
# why INLA truncates the upper tail of `ρ` (and of the precisions).
ccd_points = [
    convert(NamedTuple, convert(NaturalHyperparameters, p.θ))
        for p in result_inla.exploration.grid_points
]

fig = Figure(size = (760, 340))
ax1 = Axis(
    fig[1, 1], xlabel = "log τ_ar  (innovation precision)", ylabel = "ρ  (correlation)",
    title = "Joint hyperparameter posterior",
)
scatter!(
    ax1, log.(gold_draws[:τ_ar]), gold_draws[:ρ];
    color = (:steelblue, 0.18), markersize = 3, label = "gold (NUTS)"
)
scatter!(
    ax1, [log(p.τ_ar) for p in ccd_points], [p.ρ for p in ccd_points];
    color = :firebrick, markersize = 11, marker = :xcross, label = "INLA CCD design"
)
axislegend(ax1; position = :lt, framevisible = false)

# And the marginal that matters most — the correlation `ρ`. INLA's curve
# falls away too early on the right; HMC-Laplace matches the gold's reach
# toward strong persistence.
ax2 = Axis(fig[1, 2], xlabel = "ρ  (correlation)", ylabel = "density", title = "Marginal posterior of ρ")
density!(ax2, gold_draws[:ρ]; color = (:steelblue, 0.25), strokecolor = :steelblue, strokewidth = 2, label = "gold (NUTS)")
density!(ax2, hmc_draws[:ρ]; color = (:seagreen, 0.0), strokecolor = :seagreen, strokewidth = 2, label = "HMC-Laplace")
## INLA's marginal is an analytic spline, not samples — plot its actual density.
mρ = inla_marginal(:ρ)
ρ_grid = range(quantile(mρ, 1.0e-3), quantile(mρ, 1 - 1.0e-3); length = 400)
lines!(
    ax2, ρ_grid, pdf.(Ref(mρ), ρ_grid);
    color = :firebrick, linewidth = 2, linestyle = :dash, label = "INLA-CCD (spline)"
)
axislegend(ax2; position = :lt, framevisible = false)
fig

# ## When to reach for HMC-Laplace
#
# INLA is the right default. It is fast, and on the large majority of latent
# Gaussian models the hyperparameter posterior is well-behaved enough that
# the deterministic design is accurate. Reach past it only when you have a
# specific reason to.
#
# **Prefer `hmc_laplace` when:**
# - The hyperparameter posterior is **curved or strongly skewed** — scale /
#   correlation trade-offs (AR and Matérn ranges are classic), or parameters
#   pinned against a **boundary** (`ρ → 1`, a variance heading to zero).
# - You need **faithful tails / credible intervals** for the
#   hyperparameters themselves, not just their modes — e.g. "how persistent
#   *could* this process be?".
# - You have **few hyperparameters** (a handful), so sampling is affordable.
#
# **Stick with `inla` when:**
# - The hyperparameter posterior is roughly Gaussian in the working space
#   (the common case), where CCD is both fast and accurate.
# - You have **many hyperparameters**, where MCMC mixing gets hard and the
#   deterministic design scales better.
# - **Speed** matters and you mainly need posterior means / modes.
#
# **On cost.** HMC-Laplace is more expensive than INLA — it runs a NUTS
# chain instead of evaluating a fixed handful of design points. But it is
# still dramatically cheaper than full MCMC on the joint model: the latent
# field (often thousands of dimensions) is integrated out by the inner
# Laplace approximation at every step, so NUTS only ever explores the
# low-dimensional hyperparameter space. It is the middle rung between INLA
# and full HMC — and, as here, sometimes the right one.
