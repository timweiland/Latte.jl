# BUGS / R-INLA `Epil` scenario — hierarchical Poisson regression.
#
# 59 epileptic patients × 4 visits = 236 observations of seizure counts.
# Covariates: pre-randomisation baseline `Base`, treatment indicator
# `Trt` (0=placebo, 1=progabide), age, and a 4th-visit dummy `V4`.
# Subject random effect `b_subject[i]` plus observation-level dispersion
# `b_obs[k]` (the BUGS-style two-level random structure).
#
# Model:
#   τ_subj ~ PCPrior.Precision(1.0, α = 0.01)
#   τ_obs  ~ PCPrior.Precision(1.0, α = 0.01)
#   fixed  ~ MvNormal(zeros(5), 100·I)
#   b_subject ~ IIDModel(59)(τ = τ_subj)
#   b_obs     ~ IIDModel(236)(τ = τ_obs)
#   log λ_k = fixed[1] + fixed[2]·log(Base_k/4) + fixed[3]·Trt_k +
#             fixed[4]·Trt_k·log(Base_k/4) + fixed[5]·log(Age_k) + fixed[6]·V4_k +
#             b_subject[Ind_k] + b_obs[k]
#   y_k ~ Poisson(λ_k)

using DynamicPPL: @model
using Distributions
using GaussianMarkovRandomFields: IIDModel
using LinearAlgebra
using CSV
using DataFrames

const SCENARIO_ID = "epil"
const RANDOM_SYMS = (:fixed, :b_subject, :b_obs)
const HP_SYMS = (:τ_subj, :τ_obs)

const _EPIL_CSV = joinpath(
    @__DIR__, "..", "external", "rinla", "epil", "epil_data.csv",
)

"""
    scenario() -> Scenario

BUGS Epil dataset: 59 subjects × 4 visits, hierarchical Poisson with
subject + observation-level RE.
"""
function scenario()
    return Scenario(
        id = SCENARIO_ID,
        title = "Epil (BUGS)",
        description = "Hierarchical Poisson regression of seizure counts. 59 subjects × 4 visits, two-level RE structure (subject + observation), classic BUGS / R-INLA reference.",
        target = "Posterior marginals of fixed effects, both τs, and the per-subject random effects.",
        engines = [:latte_inla, :latte_tmb, :latte_hmc_laplace, :nuts_reference],
        quick_n = 236,
        full_n = 236,
        timeout_seconds = 600.0,
        repetitions = 3,
        nuts_full_samples = 4_000,
        nuts_full_warmup = 2_000,
        nuts_full_chains = 4,
        nuts_target_accept = 0.95,
        notes = ["BUGS Epil. 236 obs, Poisson + log link with subject + observation IID REs."],
    )
end

# ─── Model ────────────────────────────────────────────────────────────

@model function epil_model(
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

"""
    generate_data(n; seed) -> NamedTuple

Loads the fixed Epil dataset; `n` and `seed` are ignored.
"""
function generate_data(n::Int; seed::UInt64 = UInt64(0))
    df = CSV.read(_EPIL_CSV, DataFrame)
    n_obs = nrow(df)
    log_base4 = log.(Vector{Float64}(df.Base) ./ 4)
    trt = Vector{Float64}(df.Trt)
    log_age = log.(Vector{Float64}(df.Age))
    v4 = Vector{Float64}(df.V4)
    ind = Vector{Int}(df.Ind)
    n_subject = maximum(ind)
    return (;
        n = n_obs,
        n_subject = n_subject,
        y = Vector{Int}(df.y),
        log_base4 = log_base4,
        trt = trt,
        trt_logbase4 = trt .* log_base4,
        log_age = log_age,
        v4 = v4,
        ind = ind,
    )
end

build_model(data) = epil_model(
    data.y, data.log_base4, data.trt, data.trt_logbase4, data.log_age, data.v4,
    data.ind, data.n_subject, data.n,
)
