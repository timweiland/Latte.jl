# Merge the two scaling-curve runs (Latte: serial + threaded; R-INLA: serial +
# parallel) into docs/src/data/scaling_curve.json, consumed by the BenchScaling
# component (a log-log wall-clock-vs-n figure on the Benchmarks page).
#   julia --project=benchmark benchmark/scaling/render_scaling.jl
using JSON3, Dates

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const WORKDIR = joinpath(@__DIR__, "_workdir")
const OUT = joinpath(REPO_ROOT, "docs", "src", "data", "scaling_curve.json")

latte = JSON3.read(read(joinpath(WORKDIR, "latte_curve.json"), String))
rinla = JSON3.read(read(joinpath(WORKDIR, "rinla_curve.json"), String))

# Index the R-INLA levels by mesh level so we can join on the shared mesh.
rmap = Dict(r.level => r for r in rinla.levels)

levels = map(latte.levels) do l
    r = rmap[l.level]
    Dict(
        "level" => l.level,
        "n" => l.n,                       # latent dimension (mesh nodes + fixed)
        "m" => l.m,                       # observations
        "latte_serial" => l.serial_s,
        "latte_threaded" => l.threaded_s,
        "rinla_serial" => r.serial_s,
        "rinla_parallel" => r.parallel_s,
    )
end

payload = Dict(
    "generated_at" => string(now()),
    "inla_version" => rinla.inla_version,
    "julia_threads" => latte.threads,
    # Honest scope: one machine, one benchmark family, R-INLA's PARDISO license
    # expired so its inner solves fall back to serial taucs and its "10:1"
    # threading only parallelizes the outer independent evaluations.
    "caveat" => "One workstation, one benchmark family. Latte threaded = $(latte.threads) Julia threads; " *
        "R-INLA parallel = num.threads \"10:1\". PARDISO unavailable on this machine, so R-INLA's " *
        "inner sparse solves run serial taucs — with PARDISO its parallel times could tighten at large n.",
    "levels" => levels,
)

open(OUT, "w") do io
    JSON3.pretty(io, payload)
end
println("wrote $OUT  ($(length(levels)) levels)")
