# TTFX (time-to-first-execution) measurement harness.
#
# Single-number harness for cold-start UX: spawns a fresh Julia, loads Latte,
# constructs an LGM from a tiny DPPL model, runs `inla(lgm, y)` once. Reports:
#
#   load_s          time of `using Latte` + companion deps
#   adapter_s       time of `latte_from_dppl(...)` (mostly DPPL/SCT specialisation)
#   first_inla_s    time of first `inla(lgm, y)`
#   warm_inla_s     median of 3 subsequent calls (for sanity)
#   total_cold_s    load_s + adapter_s + first_inla_s     ← the number to drive down
#
# Run as:
#
#     julia --project benchmark/ttfx/measure_ttfx.jl
#
# Or, to write JSON for diffing:
#
#     julia --project benchmark/ttfx/measure_ttfx.jl --json /tmp/ttfx.json
#
# Pure timing — no inference correctness checks. Pair with the existing
# benchmark/external/rinla/* scenarios for accuracy validation.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)

# Hide load-side __init__ noise from output for clean parsing.
const _STDERR_DEVNULL = devnull

t_load = @elapsed (using Latte)
t_load_dppl = @elapsed (using DynamicPPL: @model)
t_load_dist = @elapsed (using Distributions)
t_load_gmrf = @elapsed (using GaussianMarkovRandomFields)
t_total_load = t_load + t_load_dppl + t_load_dist + t_load_gmrf

# Also need Statistics for the median; counted in load to keep things honest.
t_load_stat = @elapsed (using Statistics)
t_total_load += t_load_stat

using Random
Random.seed!(0)

# Representative tiny scenario: IID-Poisson, n=8, single hyperparameter.
# This hits the Poisson{LogLink} fast path (the most common benchmark family).
@model function _ttfx_iid_poisson(y, n)
    τ ~ LogNormal(0.0, 1.0)
    x ~ GaussianMarkovRandomFields.IIDModel(n)(τ = τ)
    for i in eachindex(y)
        y[i] ~ Poisson(exp(x[i]); check_args = false)
    end
end

const _N = 8
const _Y = [3, 0, 4, 1, 2, 1, 5, 0]

dppl = _ttfx_iid_poisson(_Y, _N)

t_adapter = @elapsed lgm = latte_from_dppl(dppl; random = :x)

t_first_inla = @elapsed result = inla(
    lgm, _Y;
    latent_marginalization_method = SimplifiedLaplace(),
    progress = false,
)

# Warm: median of 3 subsequent calls. Doesn't enter total_cold but we
# report it so we can spot regressions in steady-state perf too.
warm_times = Float64[]
for _ in 1:3
    push!(
        warm_times, @elapsed inla(
            lgm, _Y;
            latent_marginalization_method = SimplifiedLaplace(),
            progress = false,
        )
    )
end
t_warm_inla = sort(warm_times)[2]

total_cold = t_total_load + t_adapter + t_first_inla

println()
println("─── TTFX (cold) ────────────────────────────────────────────")
println("  using Latte           ", round(t_total_load, digits = 2), " s")
println("  latte_from_dppl       ", round(t_adapter, digits = 2), " s")
println("  inla(lgm, y) first    ", round(t_first_inla, digits = 2), " s")
println("  ──────────────────────────")
println("  TOTAL COLD            ", round(total_cold, digits = 2), " s")
println()
println("─── warm ───────────────────────────────────────────────────")
println("  inla warm (median 3)  ", round(t_warm_inla * 1000, digits = 1), " ms")
println()

# Optional JSON output for diffing.
function _parse_json_arg(args)
    for (i, a) in enumerate(args)
        if a == "--json" && i < length(args)
            return args[i + 1]
        end
    end
    return nothing
end
json_path = _parse_json_arg(ARGS)
if json_path !== nothing
    # Hand-rolled JSON to avoid pulling JSON3 into the load measurement.
    open(json_path, "w") do io
        print(io, "{")
        print(io, "\"load_s\":", t_total_load, ",")
        print(io, "\"adapter_s\":", t_adapter, ",")
        print(io, "\"first_inla_s\":", t_first_inla, ",")
        print(io, "\"warm_inla_s\":", t_warm_inla, ",")
        print(io, "\"total_cold_s\":", total_cold, ",")
        print(io, "\"julia_version\":\"", VERSION, "\",")
        print(io, "\"timestamp\":", round(Int, time()))
        print(io, "}\n")
    end
    println("wrote $json_path")
end
