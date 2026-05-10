# Cold / warm / phase timing helpers.
#
# We deliberately avoid BenchmarkTools.@benchmark for the main suite.
# That tool is calibrated for microbenchmarks (median over many trials);
# here a single fit is seconds-to-minutes and the user cares about
# end-to-end wall time, not statistical micro-precision. Hand-rolled
# `@elapsed` over N reps is more honest.

using Statistics: median, quantile

"""
    cold_time(f) -> seconds::Float64

Run `f()` once, returning the wall-clock seconds it took. By construction
"cold" means "this is the first run" — Julia compilation and any package
initialisation are included. Caller is responsible for making sure
nothing about the workload has been previously executed in this session.
"""
cold_time(f) = @elapsed f()

"""
    warm_times(f; reps) -> Vector{Float64}

Run `f()` `reps` times and return per-run wall-clock seconds. Caller
should have already done a cold run before calling this — use
`cold_then_warm` for the common pattern.
"""
function warm_times(f; reps::Int = 5)
    times = Vector{Float64}(undef, reps)
    for i in 1:reps
        times[i] = @elapsed f()
    end
    return times
end

"""
    cold_then_warm(f; reps) -> (cold::Float64, warm::Vector{Float64})

The standard benchmarking pattern: one cold run (to absorb compilation
costs) followed by `reps` warm runs.

`f` should be idempotent (or at least produce comparable timings on
repeated calls).
"""
function cold_then_warm(f; reps::Int = 5)
    c = cold_time(f)
    w = warm_times(f; reps = reps)
    return (c, w)
end

"""
    summarize(times::Vector{Float64}) -> NamedTuple

Median + IQR summary of a vector of timings. We report median + IQR
rather than mean + sd because timings are right-tailed.
"""
function summarize(times::AbstractVector{<:Real})
    isempty(times) && return (median = NaN, iqr_lo = NaN, iqr_hi = NaN, n = 0)
    return (
        median = median(times),
        iqr_lo = quantile(times, 0.25),
        iqr_hi = quantile(times, 0.75),
        n = length(times),
    )
end

"""
    timed_phase(f; phase_name::Symbol)

Convenience: time `f()`, push the result into the supplied
`PhaseTimings` mutator, return whatever `f()` returned. Used by engines
that want phase-level timing.

Caller pattern:

```julia
phases = Ref(PhaseTimings(total = 0.0))
result = timed_phase(() -> build_model(...), phases, :model_construction)
result = timed_phase(() -> fit(model), phases, :sampling)
```

The `Ref` wrapping is awkward; engines typically just track phase
timings in local Float64s and assemble the `PhaseTimings` at the end.
This helper is here for cases where the phase set is dynamic.
"""
function timed_phase(f, phases::Ref{PhaseTimings}, phase_name::Symbol)
    t = @elapsed result = f()
    p = phases[]
    new_p = if phase_name === :model_construction
        PhaseTimings(
            p.model_construction === nothing ? t : p.model_construction + t,
            p.compilation, p.optimisation, p.sampling, p.posterior_summary, p.total + t
        )
    elseif phase_name === :compilation
        PhaseTimings(
            p.model_construction,
            p.compilation === nothing ? t : p.compilation + t,
            p.optimisation, p.sampling, p.posterior_summary, p.total + t
        )
    elseif phase_name === :optimisation
        PhaseTimings(
            p.model_construction, p.compilation,
            p.optimisation === nothing ? t : p.optimisation + t,
            p.sampling, p.posterior_summary, p.total + t
        )
    elseif phase_name === :sampling
        PhaseTimings(
            p.model_construction, p.compilation, p.optimisation,
            p.sampling === nothing ? t : p.sampling + t,
            p.posterior_summary, p.total + t
        )
    elseif phase_name === :posterior_summary
        PhaseTimings(
            p.model_construction, p.compilation, p.optimisation, p.sampling,
            p.posterior_summary === nothing ? t : p.posterior_summary + t,
            p.total + t
        )
    else
        error("unknown phase $(phase_name)")
    end
    phases[] = new_p
    return result
end
