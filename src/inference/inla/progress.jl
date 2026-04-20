using ProgressMeter

export ProgressState, initialize_progress!, update_progress!, advance_phase!, finish_progress!

"""
    ProgressState

Container for progress tracking state.

# Fields
- `enabled::Bool`: Whether progress tracking is enabled
- `meter::Union{Nothing, Progress}`: ProgressMeter object if enabled
"""
mutable struct ProgressState
    enabled::Bool
    meter::Union{Nothing, Progress}
end

"""
    initialize_progress!(enabled::Bool)

Initialize progress tracking for INLA inference.

# Arguments
- `enabled::Bool`: Whether to enable progress tracking

# Returns
- `ProgressState`: Progress tracking state
"""
function initialize_progress!(enabled::Bool)
    if enabled
        # 3 phases: Mode Finding (33%), Exploration (66%), Interpolation (100%)
        meter = Progress(3, dt = 0.1, desc = "INLA Inference: ", color = :blue, barlen = 50)
        return ProgressState(true, meter)
    else
        return ProgressState(false, nothing)
    end
end

"""
    update_progress!(state::ProgressState, phase::String, phase_info::NamedTuple=NamedTuple())

Update progress display information without advancing the meter.

# Arguments
- `state::ProgressState`: Progress tracking state
- `phase::String`: Description of current phase
- `phase_info::NamedTuple`: Additional information to display (optional)
"""
function update_progress!(state::ProgressState, phase::String, phase_info::NamedTuple = NamedTuple())
    return if state.enabled && state.meter !== nothing
        # Convert phase_info to vector format for showvalues
        showvalues = [(:phase, phase)]
        for (key, value) in pairs(phase_info)
            push!(showvalues, (key, repr(value)))
        end

        # Update display without advancing - use update! with current value
        current_val = state.meter.counter
        update!(state.meter, current_val, showvalues = showvalues)
    end
end

"""
    advance_phase!(state::ProgressState, phase::String, phase_info::NamedTuple=NamedTuple())

Advance to the next phase of progress.

# Arguments
- `state::ProgressState`: Progress tracking state
- `phase::String`: Description of current phase
- `phase_info::NamedTuple`: Additional information to display (optional)
"""
function advance_phase!(state::ProgressState, phase::String, phase_info::NamedTuple = NamedTuple())
    return if state.enabled && state.meter !== nothing
        # Convert phase_info to vector format for showvalues
        showvalues = [(:phase, phase)]
        for (key, value) in pairs(phase_info)
            push!(showvalues, (key, repr(value)))
        end

        # Advance to next phase
        next!(state.meter, showvalues = showvalues)
    end
end

"""
    finish_progress!(state::ProgressState)

Finish progress tracking.

# Arguments
- `state::ProgressState`: Progress tracking state
"""
function finish_progress!(state::ProgressState)
    return if state.enabled && state.meter !== nothing
        finish!(state.meter)
    end
end

"""
    create_progress_callback(state::ProgressState, phase_name::String)

Create a progress callback function for a specific phase.

# Arguments
- `state::ProgressState`: Main progress tracking state
- `phase_name::String`: Name of the current phase

# Returns
- `Function`: Callback function that accepts phase-specific information as keyword arguments
"""
function create_progress_callback(state::ProgressState, phase_name::String)
    return function (; kwargs...)
        phase_info = NamedTuple(kwargs)
        return update_progress!(state, phase_name, phase_info)
    end
end
