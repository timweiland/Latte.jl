# Micro-benchmark: per-θ (μ, Q) build cost — recognition path vs DAG path.
#
# Validates the core motivation of the concrete-LatentModel recognition work
# (tasks/recognize-concrete-gmrf-priors.org): when the user's latent prior is
# a concrete `LatentModel` (here `RW1Model(n)`), preserving it lets inference
# call `precision_matrix(::RWModel{1}; τ)` directly — a single SymTridiagonal
# build — instead of the DAG path's per-call sparse re-assembly from the DPPL
# conditional.
#
# Both paths are exercised on the *same* canonical model (RW1 + Poisson) at
# n=1000. We unwrap the auto-augmentation wrapper and benchmark the BASE prior
# so the comparison isolates the latent (μ, Q) build, not the obs layer.
#
# Run as:
#
#     julia --project benchmark/micro/recognized_vs_dag.jl
#
# Or, to write JSON for diffing:
#
#     julia --project benchmark/micro/recognized_vs_dag.jl --json /tmp/rec_vs_dag.json
#
# Unlike the main suite (benchmark/utils/timing.jl, which avoids BenchmarkTools
# for seconds-scale end-to-end fits), this is a true microbenchmark — the
# regime BenchmarkTools is built for.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)

using Latte
using DynamicPPL: @model
import DynamicPPL
using Distributions
using GaussianMarkovRandomFields
using LinearAlgebra
using Random
using BenchmarkTools
using Statistics: median

Random.seed!(20260530)

const N = 1000
const ΤVAL = 1.7

# Canonical RW1 + Poisson data.
x_true = 1.0 .+ cumsum(randn(N)) .* 0.1
y_obs = [rand(Poisson(exp(xi); check_args = false)) for xi in x_true]

# ─── Recognition path: macro preserves RW1Model as a RoutedLatentModel ───────
@latte function rw_poisson_rec(y)
    τ ~ Gamma(2.0, 1.0)
    x ~ RW1Model(length(y))(; τ = τ)
    for i in eachindex(y)
        y[i] ~ Poisson(exp(x[i]); check_args = false)
    end
end
lgm_rec = rw_poisson_rec(y_obs)
lat_rec = lgm_rec.latent_prior isa Latte.AugmentedLatentModel ?
    lgm_rec.latent_prior.base_model : lgm_rec.latent_prior

# ─── DAG path: bare adapter type-erases into a CachedDAGLatentModel ──────────
@model function rw_poisson_dag(y)
    τ ~ Gamma(2.0, 1.0)
    x ~ RW1Model(length(y))(; τ = τ)
    for i in eachindex(y)
        y[i] ~ Poisson(exp(x[i]); check_args = false)
    end
end
lgm_dag = latte_from_dppl(rw_poisson_dag(y_obs); random = :x)
lat_dag = lgm_dag.latent_prior isa Latte.AugmentedLatentModel ?
    lgm_dag.latent_prior.base_model : lgm_dag.latent_prior

@assert lat_rec isa Latte.RoutedLatentModel "recognition path did not produce a RoutedLatentModel"
println("recognition latent : ", typeof(lat_rec).name.name, " → ", model_name(lat_rec))
println("DAG latent         : ", typeof(lat_dag).name.name, " → ", model_name(lat_dag))

# Sanity: both paths must agree on the actual precision at this θ, else the
# speed comparison is meaningless.
Q_rec = precision_matrix(lat_rec; τ = ΤVAL)
Q_dag = precision_matrix(lat_dag; τ = ΤVAL)
@assert Matrix(Q_rec) ≈ Matrix(Q_dag) "recognition and DAG precision matrices disagree"
println(
    "precision agreement: ✓ (‖Q_rec - Q_dag‖∞ = ",
    round(maximum(abs.(Matrix(Q_rec) - Matrix(Q_dag))), sigdigits = 2), ")"
)

# ─── Benchmarks ──────────────────────────────────────────────────────────────
println("\n─── precision_matrix(latent; τ) — n=$N ──────────────────────")
b_pm_rec = @benchmark precision_matrix($lat_rec; τ = $ΤVAL)
b_pm_dag = @benchmark precision_matrix($lat_dag; τ = $ΤVAL)

# Full cold (μ, Q) GMRF build — what a fresh per-θ materialisation costs.
println("─── latent(; τ) — full (μ, Q) GMRF build ────────────────────")
b_full_rec = @benchmark $lat_rec(; τ = $ΤVAL)
b_full_dag = @benchmark $lat_dag(; τ = $ΤVAL)

ns(b) = median(b).time            # nanoseconds
us(b) = ns(b) / 1.0e3

function report(label, b_rec, b_dag)
    r, d = us(b_rec), us(b_dag)
    return println(
        rpad(label, 22),
        "rec ", lpad(string(round(r, digits = 2)), 9), " µs   ",
        "dag ", lpad(string(round(d, digits = 2)), 9), " µs   ",
        "speedup ×", round(d / r, digits = 1)
    )
end

println("\n─── results (median) ────────────────────────────────────────")
report("precision_matrix", b_pm_rec, b_pm_dag)
report("full (μ,Q) build", b_full_rec, b_full_dag)
println()

# ─── Optional JSON for diffing ───────────────────────────────────────────────
function _parse_json_arg(args)
    for (i, a) in enumerate(args)
        a == "--json" && i < length(args) && return args[i + 1]
    end
    return nothing
end
json_path = _parse_json_arg(ARGS)
if json_path !== nothing
    open(json_path, "w") do io
        print(io, "{")
        print(io, "\"n\":", N, ",")
        print(io, "\"precision_matrix_rec_ns\":", ns(b_pm_rec), ",")
        print(io, "\"precision_matrix_dag_ns\":", ns(b_pm_dag), ",")
        print(io, "\"full_build_rec_ns\":", ns(b_full_rec), ",")
        print(io, "\"full_build_dag_ns\":", ns(b_full_dag), ",")
        print(io, "\"julia_version\":\"", VERSION, "\",")
        print(io, "\"timestamp\":", round(Int, time()))
        print(io, "}\n")
    end
    println("wrote $json_path")
end
