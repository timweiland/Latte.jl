# Crowder seeds GLMM scenario — classic R-INLA example.
#
# 21 plates of seeds. y_i out of n_i germinate; covariates are root
# extract (`x1`: 0=bean, 1=cucumber) and seed type (`x2`: 0=O.a.75,
# 1=O.a.73). The model fits the standard 2×2 design plus an iid plate-
# level random effect.
#
# Model:
#   τ ~ PCPrior.Precision(1.0, α = 0.01)
#   α, β1, β2, β12 ~ Normal(0, 100)
#   b ~ IIDModel(21)(τ = τ)
#   logit(p_i) = α + β1·x1[i] + β2·x2[i] + β12·x1[i]·x2[i] + b[i]
#   y[i] ~ Binomial(n[i], p[i])

using DynamicPPL: @model
using Distributions
using GaussianMarkovRandomFields: IIDModel
using LinearAlgebra
using CSV
using DataFrames

const SCENARIO_ID = "seeds"
const RANDOM_SYMS = (:fixed, :b)
const HP_SYMS = (:τ,)

const _SEEDS_CSV = joinpath(
    @__DIR__, "..", "external", "rinla", "seeds", "seeds_data.csv",
)

"""
    scenario() -> Scenario

Crowder seeds: 21-plate Binomial GLMM with 2×2 fixed design and an iid
plate random effect. Targets are the four fixed-effect marginals plus
τ.
"""
function scenario()
    return Scenario(
        id = SCENARIO_ID,
        title = "Crowder Seeds",
        description = "Crowder seeds Binomial GLMM. 21 plates, 2×2 fixed design (root × seed) + iid plate random effect. Classic R-INLA reference.",
        target = "Posterior marginals of (α, β1, β2, β12), b[1..21], and τ.",
        engines = [:latte_inla, :latte_tmb, :latte_hmc_laplace, :nuts_reference],
        quick_n = 21,
        full_n = 21,
        timeout_seconds = 300.0,
        repetitions = 3,
        nuts_full_samples = 4_000,
        nuts_full_warmup = 2_000,
        nuts_full_chains = 4,
        nuts_target_accept = 0.95,
        notes = ["Crowder seeds (R-INLA classic). 21 obs, Binomial(n_i, p_i) with 2×2 logit + iid RE."],
    )
end

# ─── Model + data ─────────────────────────────────────────────────────

@model function seeds_model(y, n_trials, x1, x2, n_plate)
    τ ~ PCPrior.Precision(1.0, α = 0.01)
    fixed ~ MvNormal(zeros(4), 100.0 * I(4))
    b ~ IIDModel(n_plate)(τ = τ)
    for i in eachindex(y)
        η_i = fixed[1] + fixed[2] * x1[i] + fixed[3] * x2[i] +
            fixed[4] * x1[i] * x2[i] + b[i]
        y[i] ~ Binomial(n_trials[i], 1 / (1 + exp(-η_i)); check_args = false)
    end
end

"""
    generate_data(n; seed) -> NamedTuple

Loads the fixed seeds dataset; `n` and `seed` are ignored.
"""
function generate_data(n::Int; seed::UInt64 = UInt64(0))
    df = CSV.read(_SEEDS_CSV, DataFrame)
    return (;
        n = nrow(df),
        y = Vector{Int}(df.r),
        n_trials = Vector{Int}(df.n),
        x1 = Vector{Float64}(df.x1),
        x2 = Vector{Float64}(df.x2),
    )
end

build_model(data) = seeds_model(data.y, data.n_trials, data.x1, data.x2, data.n)
