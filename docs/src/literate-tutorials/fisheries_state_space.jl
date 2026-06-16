# # State-space stock assessment with TMB-style inference
#
# Fisheries science has been a steady adopter of Laplace-based Bayesian
# inference over the last decade. A typical modern stock assessment finds
# the hyperparameter MAP, integrates the random effects out with an inner
# Laplace approximation, and reports standard errors from the outer
# Hessian. Three features make these models a natural fit for that style:
#
# - The hyperparameter space is large. Population dynamics, process noise,
#   and several observation series each with their own catchability and
#   noise add up quickly, so 10 to 20 hyperparameters is common. A MAP point
#   estimate of the hyperparameters (empirical Bayes) is the usual approach at
#   that scale.
# - The latent state is moderate in size. Twenty to fifty years of biomass,
#   log-recruitment, or process-noise increments integrate out cleanly
#   through the inner Laplace.
# - The dynamics are mechanistic. Biomass evolves under a continuous-time
#   ODE driven by removals, and the discretised dynamics sit inside the
#   likelihood as ordinary Julia code that AD differentiates through.
#
# Latte's `tmb()` targets this shape, mirroring the marginal-likelihood
# Laplace approximation popularised in fisheries by
# [Kristensen et al. (2016)](#ref-tmb). The tutorial works through a
# state-space surplus production model with three observation series, the
# "Schaefer with multi-fleet CPUE" setup used in stock assessments, and
# shows how a calibrated research survey resolves the surplus-production
# identifiability problem.
#
# ## The model
#
# *Schaefer logistic biomass dynamics* ([Schaefer 1954](#ref-schaefer)):
#
# ```math
# \frac{\mathrm{d}B}{\mathrm{d}t} = r B \left(1 - \frac{B}{K}\right) - F(t) B
# ```
#
# where `r` is the intrinsic growth rate, `K` the carrying capacity, and
# `F(t) = C(t)/B(t)` the instantaneous fishing mortality reconstructed from
# observed annual catches `C(t)`. We integrate `dB/dt` between annual
# reporting times with an adaptive Runge-Kutta solver, in pure Julia code
# that ForwardDiff can differentiate through.
#
# *State-space form:*
#
# ```math
# \log B_{t+1} = \widehat{\log B}_{t+1}(B_t, r, K, C_t) + \sigma_{\text{proc}} \, \varepsilon_t,
# \qquad \varepsilon_t \stackrel{\mathrm{iid}}{\sim} \mathcal{N}(0, 1)
# ```
#
# *Three observation series* (commercial CPUE, research survey,
# recreational CPUE):
#
# ```math
# \log I_{t,j} = \log q_j + \log B_t + \sigma_{\text{obs},j}\, \eta_{t,j},
# \qquad \eta_{t,j} \stackrel{\mathrm{iid}}{\sim} \mathcal{N}(0, 1)
# ```
#
# The `\varepsilon_t` increments are the latent field that Latte integrates
# out via the inner Laplace; everything else is a hyperparameter. With three
# observation series there are nine hyperparameters: `(r, K, σ_proc)` plus
# `(q_j, σ_obs,j)` for each `j ∈ {com, sur, rec}`.
#
# ## Continuous-time dynamics with `OrdinaryDiffEq`
#
# We integrate `dB/dt` annually with `Tsit5`, Tsitouras's adaptive 5(4)
# Runge-Kutta solver from `OrdinaryDiffEq`, a common default for non-stiff
# ODEs. What matters for inference is that `Tsit5` threads `ForwardDiff.Dual`
# numbers through the solve, so when Latte's outer hyperparameter optimiser
# propagates `Dual`s through the likelihood the ODE solve runs under AD and
# the Hessian used for the standard errors is exact to Laplace precision.
#
# Working in `log B` keeps biomass positive without branching. The wrapper
# below takes `(log_B0, r, K, F)` and returns `log B(t = dt)`; it is the
# single primitive the model relies on.
using Latte
using DynamicPPL: @model
using Distributions
using GaussianMarkovRandomFields
using GaussianMarkovRandomFields: IIDModel
using LinearAlgebra
using Random
using OrdinaryDiffEqTsit5
using SciMLBase: ODEProblem

## d(log B)/dt = r·(1 − B/K) − F      (Schaefer logistic with constant F)
schaefer_rhs(log_B, p, t) = p[1] * (1 - exp(log_B) / p[2]) - p[3]

function schaefer_step(log_B0, r, K, F, dt)
    prob = ODEProblem(schaefer_rhs, log_B0, (0.0, dt), (r, K, F))
    sol = solve(
        prob, Tsit5();
        reltol = 1.0e-8, abstol = 1.0e-10, save_everystep = false,
    )
    return sol.u[end]
end

# A standalone forward simulator for generating "truth" data and for
# reconstructing the posterior biomass at the MAP.
function simulate_biomass(log_K, ε, catches, r, K, σ_proc)
    T = length(catches) + 1
    log_B = Vector{Float64}(undef, T)
    log_B[1] = log_K
    for t in 1:(T - 1)
        F_t = min(catches[t] / exp(log_B[t]), r - 1.0e-3)
        log_B[t + 1] = schaefer_step(log_B[t], r, K, F_t, 1.0) +
            σ_proc * ε[t]
    end
    return log_B
end

# ## Simulating a 25-year fishery
#
# True parameters chosen to mimic a moderately productive demersal stock:
# `r = 0.30`, `K = 1000` (units arbitrary, say tonnes of biomass). Three
# fleets observe the stock with very different `(q, σ)`:
#
# - The commercial CPUE has `q ≈ 0.001` and `σ_obs ≈ 0.20`, with a loose
#   prior on `q` because commercial efficiency drifts with effort changes
#   that the model does not capture.
# - The research bottom-trawl survey has `q ≈ 0.5` and `σ_obs ≈ 0.10`. Its
#   `q` prior is tight: the gear is calibrated and the survey design is
#   standardised. This is what breaks the surplus-production
#   identifiability, as the next section explains.
# - The recreational CPUE has `q ≈ 0.0005` and `σ_obs ≈ 0.30`, with a loose
#   prior and high noise to reflect the variability of angler reports.
Random.seed!(20260502)

const T = 25
true_r = 0.3
true_K = 1000.0
true_σ_proc = 0.1
true_q_com = 0.001;
true_σ_com = 0.2
true_q_sur = 0.5;
true_σ_sur = 0.1
true_q_rec = 0.0005;
true_σ_rec = 0.3

## Catch trajectory: development → peak → decline (typical fishery history)
true_catches = [50.0 + 250.0 * exp(-0.5 * ((t - 12) / 5.0)^2) for t in 1:(T - 1)]

true_ε = randn(T - 1)
true_log_B = simulate_biomass(
    log(true_K), true_ε, true_catches, true_r, true_K, true_σ_proc,
)

## Three observation series, each with their own (q, σ) calibration
true_log_I_com = log(true_q_com) .+ true_log_B .+ true_σ_com .* randn(T)
true_log_I_sur = log(true_q_sur) .+ true_log_B .+ true_σ_sur .* randn(T)
true_log_I_rec = log(true_q_rec) .+ true_log_B .+ true_σ_rec .* randn(T)

# The simulated fishery, biomass and catches on the left, the three CPUE
# series on the right:
using AlgebraOfGraphics, CairoMakie

fig = Figure(size = (1100, 360))
ax_b = Axis(
    fig[1, 1], title = "Biomass trajectory + catches",
    xlabel = "year", ylabel = "B (tonnes)",
)
lines!(ax_b, 1:T, exp.(true_log_B); color = :black, linewidth = 2, label = "B")
barplot!(
    ax_b, 1:(T - 1), true_catches; color = (:firebrick, 0.4),
    label = "catch",
)
axislegend(ax_b; position = :rt, framevisible = false)
ax_i = Axis(
    fig[1, 2], title = "Three CPUE indices (log-scale)",
    xlabel = "year", ylabel = "log I",
)
scatter!(ax_i, 1:T, true_log_I_com; color = :steelblue, label = "commercial")
scatter!(ax_i, 1:T, true_log_I_sur; color = :forestgreen, label = "survey")
scatter!(ax_i, 1:T, true_log_I_rec; color = :goldenrod, label = "recreational")
axislegend(ax_i; position = :rb, framevisible = false)
fig

# ## Why a calibrated survey?
#
# The Schaefer model is hard to identify from CPUE data alone. A model with
# `(2r, K/2, q*sqrt(2))` produces nearly the same CPUE trajectory as
# `(r, K, q)`: `r` and `K` compensate, and only their product `r·K`
# (proportional to MSY) is well-determined. Stock assessment scientists
# call this the "one-way trip" problem.
#
# Adding a research survey with a known catchability breaks the symmetry.
# The absolute level of `B(t)` is anchored by the survey's tight `q` prior,
# and the commercial and recreational indices then contribute information
# about trends. The survey gets a `Normal(log(0.5), 0.05)` prior, roughly a
# 5% standard error in `q`, while the other two get loose `Normal(_, 1.0)`
# priors.
#
# ## The DPPL model
#
# The model has the same shape as any other Latte regression: hyperparameter
# priors, an IID-Gaussian latent prior, and one observation `~` statement
# per series.
@model function multifleet_schaefer(
        log_I_com, log_I_sur, log_I_rec, catches, T,
    )
    log_r ~ Normal(log(0.3), 0.7)
    log_K ~ Normal(log(1000.0), 1.0)
    log_σ_proc ~ Normal(log(0.1), 0.5)
    ## Loose priors for the commercial and recreational fleets...
    log_q_com ~ Normal(log(0.001), 1.0)
    log_σ_com ~ Normal(log(0.2), 0.5)
    ## ...tight prior on the calibrated research survey:
    log_q_sur ~ Normal(log(0.5), 0.05)
    log_σ_sur ~ Normal(log(0.1), 0.4)
    log_q_rec ~ Normal(log(0.0005), 1.5)
    log_σ_rec ~ Normal(log(0.3), 0.5)

    r = exp(log_r)
    K = exp(log_K)
    σ_proc = exp(log_σ_proc)

    ## Latent: T-1 process-noise increments, IID standard normal under the
    ## prior. σ_proc is folded into the dynamics rather than the prior, the
    ## non-centered parameterisation that keeps the latent prior independent
    ## of the hyperparameters and the LGM Gaussian.
    ε ~ IIDModel(T - 1)(τ = 1.0)

    ## Forward-simulate biomass with the ODE step above. The buffer eltype
    ## must promote across ε (latent) and the closure-captured Dual hyperparameters.
    Tp = promote_type(eltype(ε), typeof(log_K), typeof(σ_proc))
    log_B = Vector{Tp}(undef, T)
    log_B[1] = log_K
    for t in 1:(T - 1)
        F_t = min(catches[t] / exp(log_B[t]), r - 1.0e-3)
        log_B[t + 1] = schaefer_step(log_B[t], r, K, F_t, 1.0) +
            σ_proc * ε[t]
    end

    ## Three independent observation likelihoods, each LogNormal(q·B, σ).
    σ_com = exp(log_σ_com)
    σ_sur = exp(log_σ_sur)
    σ_rec = exp(log_σ_rec)
    for t in 1:T
        log_I_com[t] ~ Normal(log_q_com + log_B[t], σ_com)
        log_I_sur[t] ~ Normal(log_q_sur + log_B[t], σ_sur)
        log_I_rec[t] ~ Normal(log_q_rec + log_B[t], σ_rec)
    end
end

# Build the LGM. The likelihood involves an iterative ODE solve, which
# `SparseConnectivityTracer` cannot trace through, so we pass `:dense` and
# the Hessian is built without sparsity-pattern detection. At 24 latent
# dimensions the difference is negligible.
dppl = multifleet_schaefer(
    true_log_I_com, true_log_I_sur, true_log_I_rec, true_catches, T,
)
lgm = latte_from_dppl(
    dppl; random = :ε, likelihood_hessian_pattern = :dense,
)

# ## Fitting with `tmb()`
#
# `tmb()` runs three steps:
# 1. the outer hyperparameter MAP optimisation (BFGS in working space),
# 2. the inner Laplace at each outer evaluation, integrating ε out, and
# 3. a Hessian of the outer objective at the MAP, giving working-space
#    standard errors that transform back to natural space for reporting.
#
# We pass `FiniteDiffStrategy()` for the outer Hessian. With nine
# hyperparameters and an inner Newton-plus-Cholesky chain, finite
# differences on the AD gradient are more numerically stable here than
# nested AD; TMB uses a similar scheme.
y_joint = vcat(true_log_I_com, true_log_I_sur, true_log_I_rec)
result = tmb(lgm, y_joint; diff_strategy = Latte.FiniteDiffStrategy())

# ## Posteriors
#
# The nine hyperparameters, summarised against truth. The model declares
# each one on the log scale (`log_r`, `log_K`, …), so the natural-scale MAP
# is the exponential of the marginal's median, and the standard error is the
# marginal's working-scale standard deviation.
using DataFrames
truth_natural = (
    r = true_r, K = true_K, σ_proc = true_σ_proc,
    q_com = true_q_com, σ_com = true_σ_com,
    q_sur = true_q_sur, σ_sur = true_σ_sur,
    q_rec = true_q_rec, σ_rec = true_σ_rec,
)
hp_keys = collect(keys(lgm.hyperparameter_spec.free))
hp_marg = hyperparameter_marginals(result)
summary_df_tmb = DataFrame(
    parameter = [Symbol(string(k)[5:end]) for k in hp_keys],
    truth = [getproperty(truth_natural, Symbol(string(k)[5:end])) for k in hp_keys],
    map = [exp(median(hp_marg[i])) for i in eachindex(hp_keys)],
    se_working = [std(hp_marg[i]) for i in eachindex(hp_keys)],
)

# All five `(q, σ)` parameters land within a few percent of truth, including
# the loose-prior commercial and recreational catchabilities. The survey
# anchors the level, and the relative trends in the other indices fall into
# place. `r` and `K` are slightly biased, a known feature of the Schaefer
# model even with multi-fleet data, but their MAP product is about right and
# the recovered biomass trajectory tracks truth closely.
#
# To reconstruct biomass at the MAP we combine the hyperparameter MAP with
# the latent posterior mean. The process-noise increments come from the
# named latent group `:ε`, and the dynamics hyperparameters from their
# marginals' medians on the log scale:
ε_map = mean.(latent_marginals(result, :ε))
r_map = exp(median(hyperparameter_marginals(result, :log_r)[1]))
K_map = exp(median(hyperparameter_marginals(result, :log_K)[1]))
σ_proc_map = exp(median(hyperparameter_marginals(result, :log_σ_proc)[1]))
log_B_map = simulate_biomass(
    log(K_map), ε_map, true_catches, r_map, K_map, σ_proc_map,
)

biomass_df = DataFrame(
    year = repeat(1:T, 2),
    B = vcat(exp.(true_log_B), exp.(log_B_map)),
    series = repeat(["truth", "MAP"]; inner = T),
)

data(biomass_df) *
    mapping(:year => "year", :B => "B (tonnes)", color = :series => "") *
    visual(Lines) |> draw(;
    axis = (title = "Reconstructed biomass at the MAP",)
)

# The inner Laplace gives every `ε[t]` a Gaussian posterior. A summary of
# the first three increments:
summary_df(latent_marginals(result, :ε)[1:3])

# The marginal log-likelihood from the Laplace approximation is available
# for comparing alternative model formulations:
log_marginal_likelihood(result)

# ## What this demonstrates
#
# A few threads run through the example. The hyperparameter MAP is
# nine-dimensional; `tmb()` estimates it with one BFGS run plus a single
# Hessian evaluation, reporting standard errors from that Hessian rather than
# integrating over the hyperparameter posterior. The
# dynamics live inside the likelihood: the Schaefer ODE is integrated with
# `Tsit5` from `OrdinaryDiffEq`, ForwardDiff's `Dual`s thread through the
# solve, and the Laplace approximation gets exact Hessians without any custom
# adjoint code. Swapping `schaefer_step` for a stiff solver (`Rodas5`), an
# event-driven solver (catch closures, marine protected areas), or a
# stochastic DE leaves the inference machinery unchanged. Each fleet
# contributes its own `~` statement, and the per-fleet `(q, σ)`
# hyperparameters are estimated jointly; a fourth fleet is two more lines.
# The tight `q` prior on the calibrated survey is the device stock
# assessments use to break the surplus-production "one-way trip", shown here
# working end to end on simulated data.
#
# A few directions to extend this:
#
# - Time-varying productivity: replace `r` with a random walk in `log r(t)`,
#   parameterised by `(σ_r, ρ_r)`, adding two hyperparameters and `T-1`
#   latent dimensions. State-space assessment models in the style of
#   [Nielsen & Berg (2014)](#ref-sam) do this for recruitment.
# - Stock-recruitment forms: Beverton-Holt, Ricker, and hockey-stick are
#   Julia functions of the hyperparameters and the latent recruitment
#   deviations.
# - Further SciML interop: sensitivities, parameter screening, and
#   reaction-network DSLs compose with this pipeline, since they share the
#   same Julia and AD interfaces.
#
# For more on the inference protocol shared across `inla`, `tmb`, and
# `hmc_laplace`, see the [Main Interface](../main_interface.md) reference.
#
# ## References
#
# ```@raw html
# <div class="ref-grid-2">
# <PaperCite
#   tag="TMB"
#   title="TMB: Automatic Differentiation and Laplace Approximation"
#   authors="K. Kristensen, A. Nielsen, C. W. Berg, H. Skaug & B. M. Bell"
#   venue="Journal of Statistical Software" year="2016"
#   doi="10.18637/jss.v070.i05"
#   url="https://doi.org/10.18637/jss.v070.i05"
#   abstract="The TMB R package: fast Laplace approximation of the marginal likelihood for latent-variable models, with automatic differentiation for the gradients and Hessians." />
# <PaperCite
#   tag="Schaefer"
#   title="Some Aspects of the Dynamics of Populations Important to the Management of the Commercial Marine Fisheries"
#   authors="M. B. Schaefer"
#   venue="Bull. Inter-American Tropical Tuna Commission (repr. Bull. Math. Biol.)" year="1954"
#   url="https://doi.org/10.1007/BF02464432"
#   abstract="The logistic surplus-production model of fish-stock dynamics used here, relating biomass growth to intrinsic rate r and carrying capacity K. Linked via the 1991 Bulletin of Mathematical Biology reprint." />
# <PaperCite
#   tag="SAM"
#   title="Estimation of Time-Varying Selectivity in Stock Assessments Using State-Space Models"
#   authors="A. Nielsen & C. W. Berg"
#   venue="Fisheries Research" year="2014"
#   doi="10.1016/j.fishres.2014.01.014"
#   url="https://doi.org/10.1016/j.fishres.2014.01.014"
#   abstract="The state-space assessment model (SAM), which treats recruitment and other quantities as latent random walks integrated out by a Laplace approximation." />
# </div>
# ```
