# # Simulation-based calibration
#
# How do you know your inference is *correct*? Not "does the posterior
# look reasonable on this dataset", which is posterior-predictive
# checking — but: given the very generative process the model
# describes, does your inference method recover the truth on average?
#
# Simulation-Based Calibration (Talts, Betancourt, Simpson, Vehtari,
# Gelman 2020) answers this with one clean protocol:
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

# A simple latent-Gaussian model: IID Normal-ish effects under a
# PC prior on precision, Poisson likelihood. Written as an `@latte`
# model — `sbc_run` accepts the resulting `LatentGaussianModel` factory
# directly (it draws priors via `rand` and infers on the LGM), so no
# `random` kwarg is needed.
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
# `n_attempted = 40` is only enough for a smoke test — we want to
# exercise the pipeline, not make calibration claims. For real
# calibration claims aim for `n_attempted >= 1000`.
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
# For the `τ` hyperparameter in this model, you should see:
#
# - Mean quantile position close to 0.5 — no systematic bias.
# - 50% coverage ≈ 0.5, 80% coverage ≈ 0.8, 95% coverage ≈ 0.95 —
#   credible intervals are faithful.
#
# With `n_attempted = 40`, there's real sampling noise in these
# numbers; the yellow warning in the summary banner is your
# reminder of that. Values outside a ±0.15 envelope around nominal
# on 40 replicates is only mildly suspicious; outside ±0.05 on 1000
# replicates would be a real miscalibration signal.
#
# ## Reading the raw data
#
# Ranks and truths live on the result as dense matrices:
size(r.ranks), size(r.truths)

# Column `j` corresponds to `r.targets[j]`:
r.targets

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
# SBC validates the **inference procedure** against **the model's own
# prior**. It is:
#
# - **The right tool** for "is my Laplace approximation biased?",
#   "does my MCMC mix?", "does my approximation degrade in tails?".
# - **Not the right tool** for "does my model fit my data?" — that's
#   what posterior predictive checks are for.
#
# If your prior generates scientifically absurd datasets (a common
# outcome with vaguely-specified PC priors), SBC will still report
# faithfully against *that* prior. Scenario-restricted SBC — sampling
# from a truth-filter rather than the full prior — is a natural
# follow-up but explicitly not included in the MVP.
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
