# Scottish lip cancer scenario — classic R-INLA spatial example.
#
# 56 districts, observed lip-cancer counts `y` against expected counts
# `E` (treated as log-offset). Covariate `X` is the percentage of
# agricultural / forestry / fishing workers in the district. Districts
# share a Besag (ICAR) spatial random effect on the log-rate.
#
# Model:
#   τ ~ PCPrior.Precision(1.0, α = 0.01)
#   α, β ~ Normal(0, 100)
#   u ~ BesagModel(W)(τ = τ)            # sum-to-zero ICAR
#   y[i] ~ Poisson(exp(log(E[i]) + α + β · X[i]/10 + u[i]))

using DynamicPPL: @model
using Distributions
using GaussianMarkovRandomFields: BesagModel
using SparseArrays
using LinearAlgebra
using CSV
using DataFrames

const SCENARIO_ID = "scotland"
const RANDOM_SYMS = (:fixed, :u)
const HP_SYMS = (:τ,)

const _SCOT_DATA_CSV = joinpath(
    @__DIR__, "..", "external", "rinla", "scotland", "scotland_data.csv",
)
const _SCOT_EDGES_CSV = joinpath(
    @__DIR__, "..", "external", "rinla", "scotland", "scotland_edges.csv",
)

"""
    scenario() -> Scenario

Scottish lip cancer: 56-district Besag-ICAR Poisson with log-offset
and one fixed covariate. Targets are the fixed-effect marginals plus τ
plus the spatial field u.
"""
function scenario()
    return Scenario(
        id = SCENARIO_ID,
        title = "Scottish Lip Cancer",
        description = "Scottish lip cancer Besag/ICAR spatial Poisson. 56 districts, log-expected offset, X = % agricultural workers. Classic R-INLA reference.",
        target = "Posterior marginals of (α, β), u[1..56], and τ.",
        engines = [:latte_inla, :latte_tmb, :latte_hmc_laplace, :nuts_reference],
        quick_n = 56,
        full_n = 56,
        timeout_seconds = 600.0,
        repetitions = 3,
        nuts_full_samples = 4_000,
        nuts_full_warmup = 2_000,
        nuts_full_chains = 4,
        nuts_target_accept = 0.95,
        notes = ["Scottish lip cancer (R-INLA classic). 56 districts, Poisson + log-offset + Besag spatial RE."],
    )
end

# ─── Data + adjacency ─────────────────────────────────────────────────

"""
    _adjacency_matrix(n, edge_df) -> SparseMatrixCSC{Float64}

Symmetric 0/1 adjacency. Each undirected edge (i,j) sets W[i,j] = W[j,i] = 1.
"""
function _adjacency_matrix(n::Int, edge_df::DataFrame)
    Is = Int[]
    Js = Int[]
    for row in eachrow(edge_df)
        i, j = Int(row.i), Int(row.j)
        push!(Is, i)
        push!(Js, j)
        push!(Is, j)
        push!(Js, i)
    end
    Vs = ones(Float64, length(Is))
    return sparse(Is, Js, Vs, n, n)
end

# ─── Model ────────────────────────────────────────────────────────────

@model function scotland_model(y, log_E, x_scaled, W, n_district)
    τ ~ PCPrior.Precision(1.0, α = 0.01)
    fixed ~ MvNormal(zeros(2), 100.0 * I(2))
    u ~ BesagModel(W)(τ = τ)
    for i in eachindex(y)
        η_i = log_E[i] + fixed[1] + fixed[2] * x_scaled[i] + u[i]
        y[i] ~ Poisson(exp(η_i); check_args = false)
    end
end

"""
    generate_data(n; seed) -> NamedTuple

Loads the fixed Scotland dataset; `n` and `seed` are ignored.
"""
function generate_data(n::Int; seed::UInt64 = UInt64(0))
    df = CSV.read(_SCOT_DATA_CSV, DataFrame)
    edges = CSV.read(_SCOT_EDGES_CSV, DataFrame)
    n_d = nrow(df)
    return (;
        n = n_d,
        y = Vector{Int}(df.Counts),
        log_E = log.(Vector{Float64}(df.E)),
        x_scaled = Vector{Float64}(df.X) ./ 10,
        W = _adjacency_matrix(n_d, edges),
    )
end

build_model(data) = scotland_model(data.y, data.log_E, data.x_scaled, data.W, data.n)
