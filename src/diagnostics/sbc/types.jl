using OrderedCollections: OrderedDict

export SBCTarget, Hyperparameters, NamedScalars,
    TargetDescriptor, SBCReplicate, SBCFailure,
    SBCFailurePolicy, ReplicateDiagnostics, SBCResult

# ─── Targets (what to rank) ───────────────────────────────────────────

"""
    SBCTarget

Abstract type describing which scalar quantities an SBC run should rank
`θ_true` against. Concrete subtypes:

- `Hyperparameters()` — every scalar component of every free
  hyperparameter in the LGM.
- `NamedScalars(syms)` — rank only the listed syms (each must resolve
  to a scalar hyperparameter for MVP).

Target resolution happens up front via `resolve_targets(target, lgm)`,
before any replicate runs. Ranking latent field components is
deliberately out of scope for MVP (see task file).
"""
abstract type SBCTarget end

struct Hyperparameters <: SBCTarget end

struct NamedScalars <: SBCTarget
    symbols::Vector{Symbol}
end
NamedScalars(syms::Symbol...) = NamedScalars(collect(syms))

"""
    TargetDescriptor

Concrete scalar target. `label` is the user-facing name (e.g. `:τ` for
a scalar prior, `Symbol("β[1]")` for a vector component). The functions
extract the corresponding scalar from either the prior-drawn `truth`
NamedTuple or from the posterior-draw θ matrix.
"""
struct TargetDescriptor
    label::Symbol
    sym::Symbol              # root DPPL sym
    index::Union{Nothing, Int}  # `nothing` scalar, `i` for vector component
    extract_truth::Function   # truth_nt -> Float64
    extract_posterior::Function  # θ_mat (n × n_hp) -> Vector{Float64}
end

# ─── Per-replicate records ────────────────────────────────────────────

"""
    SBCReplicate

Structured record returned by `_prior_simulate`. Carries the prior
draw for one replicate.
"""
struct SBCReplicate{TruthNT, Y}
    replicate_id::Int
    truth::TruthNT            # NamedTuple over non-observation syms
    y::Y                      # simulated observations (type preserved)
end

"""
    ReplicateDiagnostics

Per-successful-replicate diagnostic bundle. Engines populate the fields
they know; others remain at the default. `misc` is an escape hatch for
engine-specific extras.
"""
Base.@kwdef struct ReplicateDiagnostics
    replicate_id::Int
    inference_time::Float64 = 0.0
    convergence_ok::Union{Nothing, Bool} = nothing
    engine_warnings::Vector{String} = String[]
    posterior_notes::Vector{String} = String[]
    misc::Dict{Symbol, Any} = Dict{Symbol, Any}()
end

"""
    SBCFailure

Record of a failed replicate. `truth` is preserved when available —
crucial for diagnosing failure clusters (e.g. "inference blows up
whenever τ_true > 100"). `stage` is one of `:prior_simulate`,
`:model_build`, `:inference`, `:posterior_sample`, `:rank`.
"""
struct SBCFailure
    replicate_id::Int
    stage::Symbol
    error_type::String
    message::String
    truth::Union{Nothing, NamedTuple}
end

# ─── Policies ──────────────────────────────────────────────────────────

"""
    SBCFailurePolicy(; on_failure = :record, max_failure_rate = 0.05)

`on_failure = :record` catches, logs, and continues. `:error`
propagates. `max_failure_rate` marks the final `SBCResult.status` as
`:invalid` when the proportion of failures exceeds the threshold.
"""
Base.@kwdef struct SBCFailurePolicy
    on_failure::Symbol = :record
    max_failure_rate::Float64 = 0.05
end

# ─── Run result ────────────────────────────────────────────────────────

"""
    SBCResult

Experiment record produced by `sbc_run`. Reproducible: stores
`base_seed` and `engine_kwargs` so the run can be replayed.

`ranks[k, j]` = rank of the true value for target `j` in posterior
draw `k`. Equivalently `count(posterior_draws .< θ_true) + tie_break`
per replicate. Ranks are integers in `{0, …, n_posterior}`.

`truths[k, j]` = the true value (in natural space) for that replicate
and target.

`status`:
- `:valid` — failure rate at or below threshold and no policy violation.
- `:completed_with_failures` — failures exist but within threshold.
- `:invalid` — failure rate exceeded threshold.
"""
struct SBCResult
    targets::Vector{TargetDescriptor}
    ranks::Matrix{Int}
    truths::Matrix{Float64}
    n_posterior::Int
    n_attempted::Int
    n_success::Int
    n_failures::Int
    failures::Vector{SBCFailure}
    diagnostics::Vector{ReplicateDiagnostics}
    status::Symbol
    base_seed::UInt64
    engine::Symbol
    engine_kwargs::NamedTuple
    elapsed::Float64
end
