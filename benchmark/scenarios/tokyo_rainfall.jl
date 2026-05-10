# Tokyo rainfall scenario — classic R-INLA example.
#
# 366 days of rainfall observations from 1983-1984 Tokyo. Each day t
# has y_t out of n_t Bernoulli trials (n_t = 2 for most days, 1 for
# Feb 29 in non-leap years).
#
# Model:
#   τ ~ PCPrior.Precision(1.0, α = 0.01)
#   x ~ RW2Model(366)(τ = τ)
#   y[t] ~ Binomial(n[t], sigmoid(x[t]))
#
# Real data lives in `benchmark/external/rinla/tokyo/tokyo_data.csv`,
# emitted by `R-INLA::data(Tokyo)`. `generate_data` loads it (the `n`
# argument is ignored — the dataset is fixed).

using DynamicPPL: @model
using Distributions
using GaussianMarkovRandomFields: RW2Model
using CSV
using DataFrames

const SCENARIO_ID = "tokyo_rainfall"
const RANDOM_SYMS = (:x,)
const HP_SYMS = (:τ,)

const _TOKYO_CSV = joinpath(
    @__DIR__, "..", "external", "rinla", "tokyo", "tokyo_data.csv",
)

"""
    scenario() -> Scenario

Tokyo rainfall: 366-day RW2 + binomial logit. Posterior on the daily
logit-probability x_t is what we compare against R-INLA.
"""
function scenario()
    return Scenario(
        id = SCENARIO_ID,
        title = "Tokyo Rainfall",
        description = "Tokyo 1983-1984 rainfall, RW2 latent + binomial logit. Classic R-INLA reference dataset.",
        target = "Posterior marginals of x[1..366] (latent logit-probability) and τ.",
        engines = [:latte_inla, :latte_tmb, :latte_hmc_laplace, :nuts_reference],
        quick_n = 366,
        full_n = 366,
        timeout_seconds = 600.0,
        repetitions = 3,
        nuts_full_samples = 2_000,
        nuts_full_warmup = 1_000,
        nuts_full_chains = 4,
        nuts_target_accept = 0.95,
        notes = ["Tokyo rainfall (R-INLA classic). 366 obs, n_t binomial trials per day."],
    )
end

# ─── Model + data ─────────────────────────────────────────────────────

@model function tokyo_model(y, n, n_trials)
    τ ~ PCPrior.Precision(1.0, α = 0.01)
    x ~ RW2Model(n)(τ = τ)
    for t in eachindex(y)
        y[t] ~ Binomial(n_trials[t], 1 / (1 + exp(-x[t])); check_args = false)
    end
end

"""
    generate_data(n; seed) -> NamedTuple

Loads the fixed Tokyo dataset; `n` and `seed` are ignored.
"""
function generate_data(n::Int; seed::UInt64 = UInt64(0))
    df = CSV.read(_TOKYO_CSV, DataFrame)
    return (;
        n = nrow(df),
        y = Vector{Int}(df.y),
        n_trials = Vector{Int}(df.n),
        time = Vector{Int}(df.time),
    )
end

build_model(data) = tokyo_model(data.y, data.n, data.n_trials)
