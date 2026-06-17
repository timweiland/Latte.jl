# # Age-structured stock assessment with a nonlinear latent field
#
# The [state-space surplus-production tutorial](fisheries_state_space.md) tracked a
# single biomass through time. Its latent field — the process-noise increments — was
# Gaussian, so the latent prior was an ordinary GMRF. Age-structured assessment models
# add a second axis, age, and a survival process that is *nonlinear* in the latent
# variables: this year's numbers-at-age depend on `exp` of last year's fishing
# mortality. That nonlinearity makes the latent prior **non-Gaussian**.
#
# This is the state-space assessment model (SAM) of
# [Nielsen & Berg (2014)](#ref-sam), the workhorse for many ICES stocks. Three things
# make it a good fit for Latte:
#
# - The latent field is two-dimensional: log-numbers `logN[a, y]` and log-fishing-
#   mortality `logF[a, y]` over age `a` and year `y`. Both are random fields with their
#   own process noise.
# - The survival recursion is nonlinear. Numbers carry over as
#   `logN[a, y] = logN[a-1, y-1] - exp(logF[a-1, y-1]) - M`, and the catch is the
#   nonlinear Baranov equation. A Gaussian prior cannot represent the curvature the
#   `exp` introduces.
# - Despite that, the joint is still a latent-Gaussian-*shaped* model: a structured,
#   sparse latent field observed through a smooth likelihood. `@latte` recognises the
#   nonlinear coupling automatically and fits it by iterated Laplace — no manual setup,
#   and the same model runs through `inla` and `tmb`.
#
# ## The model
#
# *Fishing mortality* follows an independent random walk in time for each age:
#
# ```math
# \log F_{a, y} = \log F_{a, y-1} + \sigma_F\, \xi_{a, y},
# \qquad \xi_{a, y} \stackrel{\mathrm{iid}}{\sim} \mathcal{N}(0, 1).
# ```
#
# *Numbers-at-age* combine recruitment (a random walk at age 1) with a survival
# recursion for older ages. With natural mortality `M` fixed, total mortality is
# `Z_{a, y} = F_{a, y} + M`, and survivors age forward:
#
# ```math
# \log N_{1, y} = \log N_{1, y-1} + \sigma_N\, \varepsilon_{1, y},
# \qquad
# \log N_{a, y} = \log N_{a-1, y-1} - \exp(\log F_{a-1, y-1}) - M + \sigma_N\, \varepsilon_{a, y}.
# ```
#
# The `\exp(\log F)` term is the nonlinearity: survival depends on the *level* of
# fishing mortality, not its logarithm. This is what `@latte` detects and what pushes
# the latent prior out of the Gaussian family.
#
# *Catch-at-age* is observed through the Baranov catch equation
# ([Quinn & Deriso 1999](#ref-quinn)) on the log scale:
#
# ```math
# \log C_{a, y} = \log N_{a, y} + \log F_{a, y} - \log Z_{a, y}
#                 + \log\!\bigl(1 - e^{-Z_{a, y}}\bigr) + \sigma_c\, \eta_{a, y}.
# ```
#
# The three process and observation standard deviations `(\sigma_N, \sigma_F, \sigma_c)`
# are the hyperparameters; everything inside the loops is the latent field.
#
# ## Simulating a fishery
#
# We use four age classes over ten years — small enough to read off, with the same
# structure a real assessment would have. Natural mortality is fixed at `M = 0.2`, the
# usual default when there is no information to estimate it.
using Latte
using Distributions
using Random
using Statistics: mean, std, median

const nA, nY = 4, 10
const M = 0.2
fl(a, y) = (y - 1) * nA + a   # flat index of (age a, year y) into the catch series

# Simulate truth: an initial age structure, then the random-walk dynamics forward.
Random.seed!(20260617)
σN_true, σF_true, σc_true = 0.1, 0.1, 0.1

logN_t = zeros(nA, nY)
logF_t = zeros(nA, nY)
for a in 1:nA
    logN_t[a, 1] = 8.0 + 0.1 * randn()
    logF_t[a, 1] = -1.5 + 0.1 * randn()
end
for y in 2:nY
    for a in 1:nA
        logF_t[a, y] = logF_t[a, y - 1] + σF_true * randn()
    end
    logN_t[1, y] = logN_t[1, y - 1] + σN_true * randn()
    for a in 2:nA
        logN_t[a, y] = logN_t[a - 1, y - 1] - exp(logF_t[a - 1, y - 1]) - M + σN_true * randn()
    end
end

## Observed log-catch-at-age, stored as a flat series (the natural layout of a
## catch-at-age table read row by row).
logC = [
    let Z = exp(logF_t[a, y]) + M
            logN_t[a, y] + logF_t[a, y] - log(Z) + log1p(-exp(-Z)) + σc_true * randn()
    end for y in 1:nY for a in 1:nA
]

# The latent fishing mortality we want to recover (left) and the noisy catch-at-age we
# actually observe (right), one line per age class:
using CairoMakie

fig = Figure(size = (1000, 380))
ax_f = Axis(fig[1, 1], title = "True fishing mortality", xlabel = "year", ylabel = "F")
ax_c = Axis(fig[1, 2], title = "Observed log-catch-at-age", xlabel = "year", ylabel = "log C")
agecols = [:steelblue, :forestgreen, :goldenrod, :firebrick]
for a in 1:nA
    lines!(ax_f, 1:nY, exp.(logF_t[a, :]); color = agecols[a], linewidth = 2, label = "age $a")
    scatter!(ax_c, 1:nY, [logC[fl(a, y)] for y in 1:nY]; color = agecols[a], label = "age $a")
end
axislegend(ax_f; position = :rt, framevisible = false)
fig

# ## The `@latte` model
#
# The model reads like the generative process above. Latent age-year fields are declared
# as matrices and indexed `logN[a, y]` — the natural notation for a state-space
# assessment. `@latte` reads the `~` statements, recognises that `logN` and `logF` form
# one coupled latent field, and detects the `exp(logF)` nonlinearity.
@latte function sam(logC, nA, nY)
    log_σN ~ Normal(-2.0, 0.5)
    log_σF ~ Normal(-2.0, 0.5)
    log_σc ~ Normal(-2.0, 0.5)
    σN = exp(log_σN)
    σF = exp(log_σF)
    σc = exp(log_σc)
    M = 0.2

    logN = Matrix{Real}(undef, nA, nY)
    logF = Matrix{Real}(undef, nA, nY)

    ## Year 1: weakly informative priors on the initial age structure.
    for a in 1:nA
        logN[a, 1] ~ Normal(8.0, 0.5)
        logF[a, 1] ~ Normal(-1.5, 0.5)
    end

    ## Process dynamics for the remaining years.
    for y in 2:nY
        for a in 1:nA
            logF[a, y] ~ Normal(logF[a, y - 1], σF)          # F random walk in time
        end
        logN[1, y] ~ Normal(logN[1, y - 1], σN)              # recruitment random walk
        for a in 2:nA                                        # survival (nonlinear in logF)
            logN[a, y] ~ Normal(logN[a - 1, y - 1] - exp(logF[a - 1, y - 1]) - M, σN)
        end
    end

    ## Baranov catch-at-age likelihood.
    for y in 1:nY, a in 1:nA
        Z = exp(logF[a, y]) + M
        predC = logN[a, y] + logF[a, y] - log(Z) + log1p(-exp(-Z))
        logC[(y - 1) * nA + a] ~ Normal(predC, σc)
    end
end

# Building the model surfaces the recognition: the latent prior is a
# `NonGaussianLatentPrior` over the stacked `[logN; logF]` field, `2 · nA · nY = 80`
# dimensions.
lgm = sam(logC, nA, nY)
lgm.latent_prior

# ## Inference
#
# `inla` explores the three-dimensional hyperparameter posterior and integrates the
# latent field out at each point with an iterated Laplace approximation — re-linearising
# the nonlinear survival recursion at every Newton step rather than once. We keep the
# marginal log-likelihood accumulator for model comparison and skip the pointwise
# predictive metrics, which fall back to Monte Carlo for the nonlinear catch likelihood.
result = inla(lgm, logC; progress = false, accumulators = (MarginalLogLikelihoodStrategy(),))

# A note Latte prints on the first call: a non-Gaussian latent prior is reported through
# Gaussian marginals. The posterior mean and precision from the iterated Laplace are
# exact; only the higher-moment skew of each marginal is not yet corrected for
# non-Gaussian priors. For the smooth fields here that is a small effect.
#
# ## Recovering the latent field
#
# The posterior means reshape back to the age-year grid. Both fields are recovered
# closely despite only catch being observed:
mN = reshape(mean.(latent_marginals(result, :logN)), nA, nY)
mF = reshape(mean.(latent_marginals(result, :logF)), nA, nY)
sF = reshape(std.(latent_marginals(result, :logF)), nA, nY)

(logN_maxerr = maximum(abs.(mN .- logN_t)), logF_maxerr = maximum(abs.(mF .- logF_t)))

# Fishing mortality per age, with the posterior mean, a 95% credible band, and the truth.
# The latent field tracks the simulated dynamics across all four ages:
fig2 = Figure(size = (1000, 640))
for a in 1:nA
    row, col = fldmod1(a, 2)
    ax = Axis(fig2[row, col], title = "log F — age $a", xlabel = "year", ylabel = "log F")
    band!(
        ax, 1:nY, mF[a, :] .- 1.96 .* sF[a, :], mF[a, :] .+ 1.96 .* sF[a, :];
        color = (:steelblue, 0.25),
    )
    lines!(ax, 1:nY, mF[a, :]; color = :steelblue, linewidth = 2, label = "posterior mean")
    lines!(ax, 1:nY, logF_t[a, :]; color = :black, linestyle = :dash, label = "truth")
    a == 1 && axislegend(ax; position = :rb, framevisible = false)
end
fig2

# ## The hyperparameter posterior
#
# The three standard deviations are reported on their natural scale. Each was declared on
# the log scale, so the natural-scale point estimate is the exponential of the marginal's
# median:
using DataFrames

hp_keys = collect(keys(lgm.hyperparameter_spec.free))
truth = (σN = σN_true, σF = σF_true, σc = σc_true)
summary_hp = DataFrame(
    parameter = [Symbol(string(k)[5:end]) for k in hp_keys],
    truth = [getproperty(truth, Symbol(string(k)[5:end])) for k in hp_keys],
    posterior_median = [exp(median(hyperparameter_marginals(result, k)[1])) for k in hp_keys],
)

# ## The same model through `tmb`
#
# Swapping the engine needs no change to the model. `tmb` finds the hyperparameter MAP,
# takes the outer Hessian for standard errors, and reconstructs the inner Laplace at the
# MAP — the workflow most age-structured assessments use in practice. Its latent
# reconstruction matches `inla`'s to plotting precision:
result_tmb = tmb(lgm, logC)
mF_tmb = reshape(mean.(latent_marginals(result_tmb, :logF)), nA, nY)

(inla_vs_tmb_maxdiff = maximum(abs.(mF .- mF_tmb)),)

# ## What this demonstrates
#
# The nonlinear survival recursion makes the latent prior non-Gaussian, and `@latte`
# handles it from the model definition alone: it recognises that `logN` and `logF` form
# one coupled field, detects the `exp(logF)` curvature, and routes the fit to an iterated
# Laplace approximation. The same `sam` model runs unchanged through `inla` (full
# hyperparameter posterior) and `tmb` (MAP with standard errors), so the choice of engine
# is a runtime decision, not a modelling one.
#
# The age-year latent fields are written with natural matrix indexing, `logN[a, y]`,
# rather than hand-flattened vectors, and the sparse structure of the survival recursion
# is exploited automatically — the per-iterate factorisation stays banded and the cost
# grows linearly in the number of years.
#
# A few directions to extend this:
#
# - A plus group at the oldest age, where survivors accumulate rather than age out.
# - Selectivity-at-age, separating `F_{a, y}` into a year effect and an age-selectivity
#   curve, as in the original SAM formulation.
# - A second observation series, such as a research survey index, added as one more
#   `~` block with its own catchability and noise — the device the
#   [surplus-production tutorial](fisheries_state_space.md) used to anchor absolute
#   stock size.
# - Management quantities — spawning-stock biomass and average fishing mortality are
#   nonlinear functions of the recovered latent field, recoverable with uncertainty by
#   pushing posterior draws through them.
#
# For the inference protocol shared across `inla`, `tmb`, and `hmc_laplace`, see the
# [Main Interface](../main_interface.md) reference.
#
# ## References
#
# ```@raw html
# <div class="ref-grid-2">
# <PaperCite
#   tag="SAM"
#   title="Estimation of Time-Varying Selectivity in Stock Assessments Using State-Space Models"
#   authors="A. Nielsen & C. W. Berg"
#   venue="Fisheries Research" year="2014"
#   doi="10.1016/j.fishres.2014.01.014"
#   url="https://doi.org/10.1016/j.fishres.2014.01.014"
#   abstract="The state-space assessment model (SAM): numbers-at-age and fishing mortality as coupled latent random walks, integrated out by a Laplace approximation." />
# <PaperCite
#   tag="Quinn"
#   title="Quantitative Fish Dynamics"
#   authors="T. J. Quinn II & R. B. Deriso"
#   venue="Oxford University Press" year="1999"
#   url="https://global.oup.com/academic/product/quantitative-fish-dynamics-9780195076318"
#   abstract="Standard reference for age-structured population dynamics, including the Baranov catch equation relating catch to numbers-at-age and instantaneous mortality rates." />
# </div>
# ```
