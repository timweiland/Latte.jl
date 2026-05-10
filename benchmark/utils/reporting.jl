# Result + Scenario types and JSON serialization.
#
# The Result schema is the single source of truth for what a benchmark
# run produces. Every engine — Latte INLA, TMB, HMC-Laplace, NUTS, and
# eventually R-INLA / brms / glmmTMB — emits one of these.

using JSON3
using OrderedCollections: OrderedDict

# ─── Engine-side types ────────────────────────────────────────────────

"""
    PhaseTimings

Per-phase wall-clock timings (seconds). Engines fill in what they have;
absent phases are nothing. Rationale: a single "runtime" number always
gets disputed for INLA/TMB/HMC comparisons.
"""
Base.@kwdef struct PhaseTimings
    model_construction::Union{Nothing, Float64} = nothing
    compilation::Union{Nothing, Float64} = nothing
    optimisation::Union{Nothing, Float64} = nothing
    sampling::Union{Nothing, Float64} = nothing
    posterior_summary::Union{Nothing, Float64} = nothing
    total::Float64 = 0.0
end

"""
    AccuracyMetrics

Posterior-agreement metrics against a reference. All optional; only
populated when the engine has a reference summary loaded for the
scenario.
"""
Base.@kwdef struct AccuracyMetrics
    # KS distance: sup_x |F_engine(x) - F_ref(x)|. Bounded in [0, 1],
    # always defined (no moment requirements), interpretable on a
    # universal scale across parameters and scenarios.
    #
    # Aggregates: `*_max` is the worst across the relevant parameter
    # block (hyperparameters or latents), `*_signed_at_argmax` is the
    # signed CDF gap at that worst point — positive = engine CDF runs
    # ahead of reference (mass shifted left); negative = engine CDF
    # lags (mass shifted right).
    posterior_ks_max::Union{Nothing, Float64} = nothing
    posterior_ks_signed_at_argmax::Union{Nothing, Float64} = nothing
    latent_ks_max::Union{Nothing, Float64} = nothing
    latent_ks_signed_at_argmax::Union{Nothing, Float64} = nothing

    # Conservative 95% KS noise floor for the reference, computed as
    # `1.36 / √(min ESS in block)`. Only populated when the reference
    # is a NUTS run (n_chains > 0). KS values at or below this floor
    # are indistinguishable from MC error of the reference itself.
    posterior_ks_mc_floor::Union{Nothing, Float64} = nothing
    latent_ks_mc_floor::Union{Nothing, Float64} = nothing

    # Reference credible-interval mass error: how much engine mass
    # falls inside the reference's central α-CI? Reported as the worst
    # absolute |P_engine(θ ∈ I_α) − α| across parameters in the block.
    posterior_ci50_mass_error::Union{Nothing, Float64} = nothing
    posterior_ci90_mass_error::Union{Nothing, Float64} = nothing
    posterior_ci95_mass_error::Union{Nothing, Float64} = nothing
    latent_ci50_mass_error::Union{Nothing, Float64} = nothing
    latent_ci90_mass_error::Union{Nothing, Float64} = nothing
    latent_ci95_mass_error::Union{Nothing, Float64} = nothing

    # Headline composite: worst KS distance across hp + latent blocks,
    # mapped to a four-tier accuracy band (descriptive label only —
    # the KS number is the primary diagnostic):
    #   green   ≤ 0.02  practically indistinguishable
    #   yellow  ≤ 0.05  small approximation bias
    #   orange  ≤ 0.15  noticeable on a CDF/density plot
    #   red     >  0.15 clearly different distribution
    worst_ks::Union{Nothing, Float64} = nothing
    accuracy_band::Union{Nothing, Symbol} = nothing

    notes::Vector{String} = String[]
end

"""
    EngineDiagnostics

Engine-specific diagnostic output. A loose dict keyed by short name;
documented per engine.
"""
const EngineDiagnostics = OrderedDict{Symbol, Any}

"""
    Result

What every benchmark run produces. Serializes to JSON via
`write_result(result, path)`.
"""
Base.@kwdef struct Result
    schema_version::Int = 1

    # Identity
    scenario_id::String
    scenario_version::String              # hash of scenario config + data
    engine_id::String
    engine_version::String                # version string of the package providing the engine
    run_id::String                         # unique per invocation
    seed::UInt64
    data_id::String                        # hash of data for this run
    mode::Symbol                           # :cold | :warm | :quick | :full | :reference | :scaling

    # Outcome
    status::Symbol                         # :success | :failed | :timeout | :invalid | :skipped
    error_type::Union{Nothing, String} = nothing
    error_message::Union{Nothing, String} = nothing
    skip_reason::Union{Nothing, String} = nothing

    # Timing
    timings::PhaseTimings
    repetitions::Int
    timeout_seconds::Union{Nothing, Float64} = nothing

    # Accuracy (only when reference is available)
    accuracy::Union{Nothing, AccuracyMetrics} = nothing

    # Engine-specific
    diagnostics::EngineDiagnostics = EngineDiagnostics()

    # Comparability (for external runs)
    comparability::Union{Nothing, String} = nothing
    comparability_notes::Union{Nothing, String} = nothing
    external_target::Union{Nothing, String} = nothing
    process_startup_seconds::Union{Nothing, Float64} = nothing
    script_path::Union{Nothing, String} = nothing

    # Environment (filled by runner)
    environment::Dict{Symbol, Any}

    # When
    timestamp_iso::String
    git_sha::Union{Nothing, String} = nothing

    # Free-form
    notes::Vector{String} = String[]
end

# ─── Scenario type ────────────────────────────────────────────────────

"""
    Scenario

What a scenario file declares. The runner pairs it with engines.

Engines is a list of engine ids that *can* run this scenario. The runner
respects this list when scheduling.
"""
Base.@kwdef struct Scenario
    id::String
    title::String
    description::String
    target::String                          # what posterior quantities are compared
    engines::Vector{Symbol}                  # eligible engine ids
    quick_n::Int = 50                       # data size for the `quick` mode
    full_n::Int = 1000
    timeout_seconds::Float64 = 300.0
    repetitions::Int = 5                    # warm-mode reps

    # NUTS reference knobs (full mode). Defaults are good enough for
    # mid-sized scenarios; scenarios with heavy-tailed posteriors should
    # bump samples and target_accept until the reference's ESS / Δq975
    # are stable.
    nuts_full_samples::Int = 2000
    nuts_full_warmup::Int = 1000
    nuts_full_chains::Int = 4
    nuts_target_accept::Float64 = 0.8

    external_implementations::NamedTuple = NamedTuple()
    notes::Vector{String} = String[]
end

# ─── JSON ────────────────────────────────────────────────────────────

# Custom serialization: Symbol → String, Nothing → null. JSON3 handles
# most of this; we just need to predeclare the StructType for our
# structs. Order matters: declare leaf types first.

JSON3.StructType(::Type{PhaseTimings}) = JSON3.Struct()
JSON3.StructType(::Type{AccuracyMetrics}) = JSON3.Struct()
JSON3.StructType(::Type{Result}) = JSON3.Struct()

"""
    write_result(result::Result, path)

Serialize a `Result` to JSON at `path`. Pretty-printed for diff-friendly
review.
"""
function write_result(result::Result, path::AbstractString)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON3.pretty(io, result)
    end
    return path
end

"""
    read_result(path) -> JSON3.Object

Read a result JSON file. Returns the parsed object — caller is
responsible for shaping it back into a Result if needed (rare; the JSON
is the canonical form).
"""
read_result(path::AbstractString) = JSON3.read(read(path, String))
