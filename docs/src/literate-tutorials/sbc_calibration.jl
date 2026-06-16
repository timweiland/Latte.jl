# # Simulation-based calibration
#
# How do you know your inference is *correct*? Not "does the posterior
# look reasonable on this dataset", which is posterior-predictive
# checking — but: given the very generative process the model
# describes, does your inference method recover the truth on average?
#
# Simulation-Based Calibration ([Talts et al. 2018](#ref-sbc)) answers
# this with a short protocol:
#
# 1. Sample `θ_true ∼ p(θ)` from the prior.
# 2. Simulate `y ∼ p(y | θ_true)` from the generative model.
# 3. Run inference on `y` to get a posterior.
# 4. Draw `L` samples from the posterior. Compute the rank of `θ_true`
#    among those samples.
# 5. Repeat.
#
# Across many replicates, the ranks of a *calibrated* inference
# procedure are uniformly distributed on `{0, 1, …, L}`. Non-uniform
# rank histograms diagnose specific failure modes:
#
# - ∪-shape (pile-up at ends): posterior is too narrow (over-confident).
# - ∩-shape (pile-up in middle): posterior is too wide (under-confident).
# - Skew left/right: posterior is biased.
#
# Latte exposes SBC via `sbc_run`, which works uniformly over the
# three engines (`:inla`, `:tmb`, `:hmc_laplace`).
#
# ## Setup
using Latte
using Distributions
using GaussianMarkovRandomFields: IIDModel
using Random

# A small latent-Gaussian model: IID effects under a PC prior on
# precision, with a Poisson likelihood. It is written as an `@latte`
# model. `sbc_run` accepts the resulting `LatentGaussianModel` factory
# directly: it draws priors via `rand` and infers on the LGM, so there is
# no `random` kwarg to pass.
@latte function smoke_model(y, n)
    τ ~ PCPrior.Precision(1.0, α = 0.01)
    x ~ IIDModel(n)(τ = τ)
    for i in eachindex(y)
        y[i] ~ Poisson(exp(x[i]); check_args = false)
    end
end

n = 10
build_model = y -> smoke_model(y, n)
y_proto = Vector{Missing}(missing, n)

# ## A smoke-test SBC run
#
# `n_attempted = 40` is only enough for a smoke test. The point here is
# to exercise the pipeline, not to make a calibration claim; for that,
# aim for `n_attempted >= 1000`.
r = sbc_run(
    build_model, y_proto;
    n_attempted = 40,
    n_posterior = 200,
    engine = :inla,
    base_seed = UInt64(0x05bc_abc),
    progress = false,
)

# The `show` method prints per-target mean quantile positions (should
# be near 0.5 if calibrated) and coverage of 50 / 80 / 95 % credible
# intervals (should match the nominal levels):
r

# ## Interpreting the summary
#
# For the `τ` hyperparameter in this model, a calibrated run shows two
# things. The mean quantile position sits close to 0.5, which says there
# is no systematic bias. And the empirical coverage tracks the nominal
# levels: roughly 0.5 at the 50% interval, 0.8 at the 80%, and 0.95 at
# the 95%, which says the credible intervals are faithful.
#
# At `n_attempted = 40` these numbers carry real sampling noise, and the
# yellow warning in the summary banner is there to remind you. On 40
# replicates a value outside a ±0.15 envelope around nominal is only
# mildly suspicious; on 1000 replicates, outside ±0.05 would be a genuine
# miscalibration signal.
#
# ## Reading the raw data
#
# Ranks and truths live on the result as dense matrices, one row per
# successful replicate and one column per ranked target:
size(r.ranks), size(r.truths)

# Column `j` corresponds to `r.targets[j]`. Each target carries a label;
# here there is just the one hyperparameter:
[d.label for d in r.targets]

# The per-replicate quantile positions (a calibrated procedure has
# these uniform on `(0, 1)`):
using Statistics
q = sbc_quantile_position(r, 1)
round(mean(q), digits = 3), round(var(q), digits = 3)  # target ≈ (0.5, 1/12 ≈ 0.083)

# Empirical credible-interval coverage:
sbc_coverage(r, 1)

# ## Failure handling
#
# PC priors are heavy-tailed; some prior draws produce datasets where
# inference breaks (numerical failures in INLA's Cholesky, optimiser
# divergence in TMB). `sbc_run` catches these and records them per
# stage:
r.n_failures, r.status

# The default policy flags the run as `:invalid` if more than 5% of
# replicates fail — a failure rate that high usually means the prior
# is producing pathological data, and a blind rank histogram from the
# survivors would silently be biased.
#
# ## Determinism
#
# SBC is reproducible via `base_seed`: replicate `i` always sees the
# same random stream regardless of the executor or scheduling. This
# makes rank matrices directly comparable across engines:
r_tmb = sbc_run(
    build_model, y_proto;
    n_attempted = 40,
    n_posterior = 200,
    engine = :tmb,
    base_seed = UInt64(0x05bc_abc),      # same seed → same prior draws
    progress = false,
)
r_tmb

# For the same draws, `:tmb`'s Gaussian-at-MAP approximation will
# typically under-cover heavy hyperparameter tails compared to
# `:inla`'s full nested-Laplace — a difference you can quantify by
# comparing `sbc_coverage` outputs across engines.
#
# ## What SBC can and can't tell you
#
# SBC validates the inference procedure against the model's own prior. It
# is the right tool for questions about the inference: is the Laplace
# approximation biased, does the MCMC mix, does the approximation degrade
# in the tails. It is the wrong tool for asking whether the model fits the
# data, which is what posterior predictive checks are for.
#
# If your prior generates scientifically absurd datasets (a common outcome
# with vaguely specified PC priors), SBC still reports faithfully against
# *that* prior. Scenario-restricted SBC, which samples from a truth-filter
# rather than the full prior, is a natural follow-up but is not part of the
# current implementation.
#
# ## Rough rule of thumb for `n_attempted`
#
# - 40–100: development smoke test.
# - 200: quick sanity check.
# - 1000–2000: narrow calibration claim for one model/scenario.
# - 5000+: tail behaviour claims, cross-engine comparisons.
#
# Each replicate runs full inference, so SBC is expensive by design.
# For 1000-replicate runs use `executor = ThreadedExecutor()` — the
# replicates are embarrassingly parallel and the result is bitwise
# identical to the sequential one.

# ## References
#
# ```@raw html
# <div class="ref-grid-2">
# <PaperCite
#   tag="SBC"
#   title="Validating Bayesian Inference Algorithms with Simulation-Based Calibration"
#   authors="S. Talts, M. Betancourt, D. Simpson, A. Vehtari & A. Gelman"
#   venue="arXiv preprint" year="2018"
#   arxiv="1804.06788"
#   url="https://arxiv.org/abs/1804.06788"
#   abstract="Introduces SBC: under exact inference, the rank of each prior-drawn parameter within its posterior is uniform, so non-uniform ranks diagnose miscalibration of any Bayesian computation." />
# </div>
# ```
