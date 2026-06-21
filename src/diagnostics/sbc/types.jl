using OrderedCollections: OrderedDict

export SBCTarget, Hyperparameters, NamedScalars, DataDependentQuantity,
    SBCFailurePolicy, SBCResult

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

"""
    Hyperparameters()

Default `SBCTarget` for `sbc_run`. Ranks every scalar component of every
free hyperparameter in the LGM.
"""
struct Hyperparameters <: SBCTarget end

"""
    NamedScalars(syms::Vector{Symbol})
    NamedScalars(syms::Symbol...)

`SBCTarget` that ranks only the named scalar quantities passed to it. Each
sym must resolve to a free scalar hyperparameter of the LGM.
"""
struct NamedScalars <: SBCTarget
    symbols::Vector{Symbol}
end
NamedScalars(syms::Symbol...) = NamedScalars(collect(syms))

"""
    DataDependentQuantity(; quantity = :loglik)
    DataDependentQuantity(f; label = :derived)

Rank a single scalar functional of the *whole* latent state `(θ, x)` that
also depends on the data `y` — the SBC analogue of a posterior predictive
test quantity. Marginal (per-hyperparameter) SBC can miss miscalibration
that a joint functional exposes.

Built-in quantities (evaluated consistently in natural space for both the
true draw and every posterior draw):

- `:loglik` → the observation log-likelihood `log p(y | x, θ)`. This is the
  exact density the prior-predictive `y` was drawn from, so it reproduces
  the data-generating process. Labelled `:loglik`.
- `:complete` → the complete-data log-likelihood `log p(y, x | θ) =
  log p(y | x, θ) + log p(x | θ)`, adding the latent GMRF log-prior.
  Labelled `:log_complete`.

A custom functional `f(lgm, θ_nt, x, y) -> Real` may be supplied directly
(`θ_nt` is the full natural-space hyperparameter NamedTuple, `x` the joint
latent vector, `y` the observations).

Requires the latent truth, which is recorded only on the LGM SBC path
(`build_model` returning a `LatentGaussianModel`). DPPL-path latent
assembly is future work; using it there raises a clear error.
"""
struct DataDependentQuantity{F} <: SBCTarget
    label::Symbol
    f::F
end

function DataDependentQuantity(; quantity::Symbol = :loglik)
    quantity === :loglik && return DataDependentQuantity(:loglik, _sbc_loglik)
    quantity === :complete && return DataDependentQuantity(:log_complete, _sbc_complete)
    throw(
        ArgumentError(
            "DataDependentQuantity: unknown quantity :$(quantity). " *
                "Use :loglik or :complete, or pass a custom f(lgm, θ_nt, x, y)."
        )
    )
end

DataDependentQuantity(f; label::Symbol = :derived) = DataDependentQuantity(label, f)

"""
    AbstractTargetDescriptor

Supertype for the concrete per-replicate scalar targets an SBC run ranks.
Two flavours exist: [`TargetDescriptor`](@ref) (a scalar pulled directly
from the truth NamedTuple / θ matrix) and [`DerivedTargetDescriptor`](@ref)
(a functional of the full per-replicate context).
"""
abstract type AbstractTargetDescriptor end

"""
    TargetDescriptor

Concrete scalar target. `label` is the user-facing name (e.g. `:τ` for
a scalar prior, `Symbol("β[1]")` for a vector component). The functions
extract the corresponding scalar from either the prior-drawn `truth`
NamedTuple or from the posterior-draw θ matrix.
"""
struct TargetDescriptor <: AbstractTargetDescriptor
    label::Symbol
    sym::Symbol              # root DPPL sym
    index::Union{Nothing, Int}  # `nothing` scalar, `i` for vector component
    extract_truth::Function   # truth_nt -> Float64
    extract_posterior::Function  # θ_mat (n × n_hp) -> Vector{Float64}
end

"""
    DerivedTargetDescriptor

Target whose scalar is a functional of the whole per-replicate context
(latent draws, observations, model), not a single column of the θ matrix.
Both extractors take the context NamedTuple assembled in the SBC run loop:
`extract_truth(ctx) -> Float64` and `extract_posterior(ctx) ->
Vector{Float64}` (one entry per posterior draw). Produced by
[`DataDependentQuantity`](@ref).
"""
struct DerivedTargetDescriptor <: AbstractTargetDescriptor
    label::Symbol
    extract_truth::Function      # ctx -> Float64
    extract_posterior::Function  # ctx -> Vector{Float64}
end

# ─── Per-replicate records ────────────────────────────────────────────

"""
    SBCReplicate

Structured record returned by `_prior_simulate`. Carries the prior
draw for one replicate. `latent_truth` is the prior-drawn joint latent
vector `x` when available (LGM path), or `nothing` (DPPL path, where the
latent lives component-wise inside `truth`); it is what
[`DataDependentQuantity`](@ref) ranks against.
"""
struct SBCReplicate{TruthNT, LT, Y}
    replicate_id::Int
    truth::TruthNT            # NamedTuple over non-observation syms
    latent_truth::LT          # joint latent vector x, or nothing
    y::Y                      # simulated observations (type preserved)
end

SBCReplicate(replicate_id::Int, truth, y) =
    SBCReplicate(replicate_id, truth, nothing, y)

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
    targets::Vector{<:AbstractTargetDescriptor}
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
