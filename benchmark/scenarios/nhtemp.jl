# New Haven temperature scenario — Normal + RW2 smoothing.
#
# 60 years (1912-1971) of annual mean temperatures from Yale's
# New Haven station (R's built-in `nhtemp`). The model fits an
# RW2 smooth trend over time plus a Gaussian observation
# variance. Targets are the smooth field and both hyperparameters.
#
# Model:
#   τ_x   ~ PCPrior.Precision(1.0, α = 0.01)   # RW2 smoothness
#   τ_obs ~ PCPrior.Precision(1.0, α = 0.01)   # observation precision
#   x ~ RW2Model(60)(τ = τ_x)
#   y[t] ~ Normal(x[t], 1/√τ_obs)

using DynamicPPL: @model
using Distributions
using GaussianMarkovRandomFields: RWModel, LatentModel
using GaussianMarkovRandomFields
using LinearAlgebra
using CSV
using DataFrames

const SCENARIO_ID = "nhtemp"
const RANDOM_SYMS = (:fixed, :x)
const HP_SYMS = (:τ_x, :σ)

const _NHTEMP_CSV = joinpath(
    @__DIR__, "..", "external", "rinla", "nhtemp", "nhtemp_data.csv",
)

# RW2SumOnly drops the linear-trend null-space constraint that
# `RW2Model` imposes by default. R-INLA's stock `rw2` only constrains
# sum-to-zero, so without this wrapper we'd be benchmarking different
# priors. Same wrapper used in tokyo_compare.jl.
struct RW2SumOnly{Inner <: RWModel{2}} <: LatentModel
    inner::Inner
end
RW2SumOnly(n::Int; kwargs...) = RW2SumOnly(RWModel{2}(n; kwargs...))
GaussianMarkovRandomFields.precision_matrix(m::RW2SumOnly; kwargs...) =
    GaussianMarkovRandomFields.precision_matrix(m.inner; kwargs...)
GaussianMarkovRandomFields.mean(m::RW2SumOnly; kwargs...) =
    GaussianMarkovRandomFields.mean(m.inner; kwargs...)
function GaussianMarkovRandomFields.constraints(m::RW2SumOnly; kwargs...)
    n = m.inner.n
    return (ones(1, n), zeros(1))
end
Base.length(m::RW2SumOnly) = length(m.inner)
function Base.getproperty(m::RW2SumOnly, s::Symbol)
    s === :inner && return getfield(m, :inner)
    s === :n && return m.inner.n
    s === :alg && return m.inner.alg
    return getfield(m, s)
end

"""
    scenario() -> Scenario

Yale New Haven temperature: 60-year RW2 + Gaussian observation noise.
"""
function scenario()
    return Scenario(
        id = SCENARIO_ID,
        title = "New Haven Temperature",
        description = "New Haven mean annual temperature 1912-1971 (R's `nhtemp`). RW2 smooth + Gaussian likelihood. Tests Normal+RW2 surface end-to-end against R-INLA.",
        target = "Posterior marginals of x[1..60] (smooth trend), τ_x (smoothness), and τ_obs (obs precision).",
        engines = [:latte_inla, :latte_tmb, :latte_hmc_laplace, :nuts_reference],
        quick_n = 60,
        full_n = 60,
        timeout_seconds = 300.0,
        repetitions = 3,
        nuts_full_samples = 4_000,
        nuts_full_warmup = 2_000,
        nuts_full_chains = 4,
        nuts_target_accept = 0.95,
        notes = ["Yale `nhtemp` (R built-in). 60 obs, Normal + RW2 smooth."],
    )
end

# ─── Model + data ─────────────────────────────────────────────────────

@model function nhtemp_model(y, n, M)
    τ_x ~ PCPrior.Precision(1.0, α = 0.01)
    σ ~ PCPrior.Sigma(1.0, α = 0.01)
    fixed ~ MvNormal(zeros(1), 100.0 * I(1))
    x ~ M(τ = τ_x)
    for t in eachindex(y)
        y[t] ~ Normal(fixed[1] + x[t], σ)
    end
end

"""
    generate_data(n; seed) -> NamedTuple

Loads the fixed nhtemp dataset; `n` and `seed` are ignored.
"""
function generate_data(n::Int; seed::UInt64 = UInt64(0))
    df = CSV.read(_NHTEMP_CSV, DataFrame)
    return (;
        n = nrow(df),
        y = Vector{Float64}(df.temp),
        year = Vector{Int}(df.year),
    )
end

build_model(data) = nhtemp_model(data.y, data.n, RW2SumOnly(data.n))
