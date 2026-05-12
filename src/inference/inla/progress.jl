using ProgressMeter

export ProgressState, initialize_progress!, update_progress!, advance_phase!, finish_progress!

# Tick budgets for the 3 top-level INLA phases. Rough heuristics: mode
# finding is fast, exploration is the heavy lifter, marginalisation is
# moderate. Cumulative => (0, 10, 80, 100), giving 10/70/20 splits.
const _PHASE_BUDGETS = (10, 70, 20)
const _PHASE_CUMULATIVE = cumsum(collect(_PHASE_BUDGETS))  # (10, 80, 100)
const _TOTAL_TICKS = _PHASE_CUMULATIVE[end]

"""
    ProgressState

Container for INLA progress tracking. `phase_index` is 1-based:
1 → mode-finding, 2 → exploration, 3 → marginalisation. The bar moves
smoothly within each phase when callbacks pass a `progress` fraction
(0 ≤ p ≤ 1); without it, the bar holds at the phase's start position
and only `showvalues` updates until `advance_phase!` is called.
"""
mutable struct ProgressState
    enabled::Bool
    meter::Union{Nothing, Progress}
    phase_index::Int
end

function initialize_progress!(enabled::Bool)
    if enabled
        meter = Progress(_TOTAL_TICKS; dt = 0.1, desc = "INLA Inference: ", color = :blue, barlen = 50)
        return ProgressState(true, meter, 1)
    end
    return ProgressState(false, nothing, 1)
end

# Start and end tick positions for the current phase.
_phase_start(state::ProgressState) =
    state.phase_index <= 1 ? 0 : _PHASE_CUMULATIVE[state.phase_index - 1]
_phase_end(state::ProgressState) =
    _PHASE_CUMULATIVE[min(state.phase_index, length(_PHASE_CUMULATIVE))]

# Build the (phase, kvs...) tuple that ProgressMeter renders below the bar.
function _showvalues(phase::String, phase_info::NamedTuple)
    showvalues = Any[(:phase, phase)]
    for (key, value) in pairs(phase_info)
        push!(showvalues, (key, repr(value)))
    end
    return showvalues
end

"""
    update_progress!(state, phase, phase_info=NamedTuple(); fraction=nothing)

Refresh the progress display. If `fraction` is given (0..1), the bar
advances smoothly within the current phase between its start/end ticks;
otherwise only the `showvalues` strip refreshes.
"""
function update_progress!(
        state::ProgressState, phase::String,
        phase_info::NamedTuple = NamedTuple(); fraction = nothing,
    )
    (state.enabled && state.meter !== nothing) || return nothing
    showvalues = _showvalues(phase, phase_info)
    tick = if fraction === nothing
        state.meter.counter
    else
        start = _phase_start(state)
        finish = _phase_end(state)
        start + clamp(round(Int, fraction * (finish - start)), 0, finish - start)
    end
    update!(state.meter, tick; showvalues = showvalues)
    return nothing
end

"""
    advance_phase!(state, phase, phase_info=NamedTuple())

Finish the current phase (bar jumps to its end tick) and step
`phase_index` forward; subsequent `update_progress!` calls fill in the
next phase's budget.
"""
function advance_phase!(
        state::ProgressState, phase::String, phase_info::NamedTuple = NamedTuple(),
    )
    (state.enabled && state.meter !== nothing) || return nothing
    end_tick = _phase_end(state)
    update!(state.meter, end_tick; showvalues = _showvalues(phase, phase_info))
    state.phase_index += 1
    return nothing
end

function finish_progress!(state::ProgressState)
    (state.enabled && state.meter !== nothing) || return nothing
    finish!(state.meter)
    return nothing
end

"""
    create_progress_callback(state, phase_name)

Wrap `state` and `phase_name` into a per-phase callback `f(; kwargs...)`
that downstream code calls during work. The callback recognises one
distinguished kwarg, `progress` (a 0..1 fraction within the current
phase), and forwards it to the meter; all other kwargs render as
`showvalues`.

For long parallel evaluations, callers should pass `progress = done /
total` from an `on_complete` hook so the bar moves continuously.
"""
function create_progress_callback(state::ProgressState, phase_name::String)
    return function (; kwargs...)
        kw = NamedTuple(kwargs)
        # Split out `progress` so it doesn't show up twice in the
        # rendered showvalues.
        fraction = get(kw, :progress, nothing)
        rest = NamedTuple{filter(!=(:progress), keys(kw))}(kw)
        return update_progress!(state, phase_name, rest; fraction = fraction)
    end
end
