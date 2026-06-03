# End-to-end benchmark: the two inference speedups, on the BUGS Epil scenario
# (236 obs, 2 hyperparameters, MvNormal fixed effects + two IID random effects
# + Poisson). Epil exercises both at once:
#
#   1. Recognition — `@latte` recognizes `fixed ~ MvNormal(...)` (FixedEffectsModel)
#      and the `IIDModel` random effects, so the latent prior rides the structured
#      fast path; `latte_from_dppl` type-erases into the DAG path.
#   2. Grid coarsening — the default `GridExplorationStrategy` evaluates a coarse
#      `(dz=1.0, max_log_drop=2.5)` grid; the old default walked a fine
#      `(0.75, 6.0)` grid (~3x the points).
#
# Three configs isolate each win:
#   DAG + fine grid          — pre-change baseline
#   recognized + fine grid   — recognition win alone
#   recognized + coarse grid — recognition + grid win (current default)
#
# Run:  julia --project benchmark/micro/grid_and_recognition_epil.jl [--json out.json]

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)

using Latte
using DynamicPPL
using DynamicPPL: @model
using Distributions
using GaussianMarkovRandomFields: IIDModel
using LinearAlgebra, Random, Statistics

const FINE = GridExplorationStrategy(integration_step_z = 0.75, max_log_drop = 6.0)
const COARSE = GridExplorationStrategy()   # current default: (1.0, 2.5)

# Synthetic data of the Epil shape (59 subjects × 4 visits = 236 obs). Relative
# speedups don't depend on exact values; this keeps the benchmark self-contained.
Random.seed!(20260530)
n_subject = 59
ind = repeat(1:n_subject, inner = 4)
n_obs = length(ind)
log_base4 = randn(n_obs) .* 0.5
trt = Float64.(rand(0:1, n_obs))
log_age = randn(n_obs) .* 0.15 .+ 3.3
v4 = Float64.(repeat([0, 0, 0, 1], n_subject))
b_subj = randn(n_subject) .* 0.3
η_gen = 1.2 .+ 0.6 .* log_base4 .- 0.3 .* trt .+ b_subj[ind]
y = [rand(Poisson(exp(clamp(η_gen[k], -5.0, 5.0)); check_args = false)) for k in 1:n_obs]
data = (;
    y, log_base4, trt, trt_logbase4 = trt .* log_base4,
    log_age, v4, ind, n_subject, n = n_obs,
)

@model function epil_dag(y, log_base4, trt, trt_logbase4, log_age, v4, ind, n_subject, n_obs)
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

@latte function epil_rec(y, log_base4, trt, trt_logbase4, log_age, v4, ind, n_subject, n_obs)
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

args = (data.y, data.log_base4, data.trt, data.trt_logbase4, data.log_age, data.v4, data.ind, data.n_subject, data.n)
lgm_dag = latte_from_dppl(epil_dag(args...); random = (:fixed, :b_subject, :b_obs))
lgm_rec = epil_rec(args...)
println("recognized latent_components: ", latent_components(lgm_rec) === nothing ? "nothing(DAG)" : collect(keys(latent_components(lgm_rec))))

best(f) = minimum(@elapsed(f()) for _ in 1:3)
runinla(lgm, strat) = inla(lgm, data.y; progress = false, exploration_strategy = strat)

# warmups
runinla(lgm_dag, FINE); runinla(lgm_rec, FINE); runinla(lgm_rec, COARSE)

t_dag_fine = best(() -> runinla(lgm_dag, FINE))
t_rec_fine = best(() -> runinla(lgm_rec, FINE))
t_rec_coarse = best(() -> runinla(lgm_rec, COARSE))

println("\n─── Epil (236 obs, 2 hp) — warm inla, best of 3 ──────────────")
println("DAG        + fine grid   : ", round(t_dag_fine, digits = 2), " s")
println("recognized + fine grid   : ", round(t_rec_fine, digits = 2), " s   (recognition ×", round(t_dag_fine / t_rec_fine, digits = 2), ")")
println("recognized + coarse grid : ", round(t_rec_coarse, digits = 2), " s   (grid ×", round(t_rec_fine / t_rec_coarse, digits = 2), ", combined ×", round(t_dag_fine / t_rec_coarse, digits = 2), ")")

function _json_arg(a)
    for (i, x) in enumerate(a)
        x == "--json" && i < length(a) && return a[i + 1]
    end
    return nothing
end
jp = _json_arg(ARGS)
if jp !== nothing
    open(jp, "w") do io
        print(io, "{\"dag_fine_s\":", t_dag_fine, ",\"rec_fine_s\":", t_rec_fine, ",\"rec_coarse_s\":", t_rec_coarse, ",\"julia\":\"", VERSION, "\"}\n")
    end
    println("wrote $jp")
end
