# Reference posterior summaries: per-scenario "ground truth" produced
# by either a deterministic quadrature oracle or a long NUTS run.
# CDF-centric schema (no mean/SD) so heavy-tailed parameters with
# undefined moments are first-class citizens.

using JSON3
using OrderedCollections: OrderedDict

"""
    ReferenceSummary

What `references/<scenario_id>/<data_id>.json` deserializes into.
Every parameter is described by its CDF (a sorted `x` grid plus
matching cumulative probabilities) and a handful of quantiles. The
accuracy code uses the CDF for the headline KS metric and the
quantiles for descriptive output and CI-mass-error checks.
"""
Base.@kwdef struct ReferenceSummary
    schema_version::Int = 1
    scenario_id::String
    data_id::String                                # hash of y; ties reference to its dataset
    parameter_names::Vector{String}                # canonical names; engines must align

    # Per-parameter CDFs: `cdf_grids[i]` sorted ascending,
    # `cdf_values[i]` is the matching cumulative probability vector.
    # Lengths can differ across parameters.
    posterior_cdf_grids::Vector{Vector{Float64}}
    posterior_cdf_values::Vector{Vector{Float64}}

    # Always-finite quantiles. Boundary points for the CI-mass-error
    # metrics; also used for descriptive reporting.
    posterior_q025::Vector{Float64}
    posterior_q25::Vector{Float64}
    posterior_median::Vector{Float64}
    posterior_q75::Vector{Float64}
    posterior_q975::Vector{Float64}
    posterior_q99::Vector{Float64}

    # NUTS diagnostics; ignore when `n_chains == 0` (oracle reference).
    ess::Vector{Float64}
    rhat::Vector{Float64}
    n_chains::Int
    n_samples_per_chain::Int
    n_warmup::Int

    seed::UInt64
    timestamp_iso::String
    notes::Vector{String} = String[]
end

JSON3.StructType(::Type{ReferenceSummary}) = JSON3.Struct()

"""
    reference_path(scenario_id, data_id) -> String

Where the reference summary for a `(scenario, dataset)` pair lives.
References are keyed by `data_id = hash(y)` so a reference is only
ever applied to the dataset it was produced from. Changing `n`, the
seed, or the data-generation code invalidates the reference
automatically (because `data_id` changes).
"""
function reference_path(scenario_id::AbstractString, data_id::AbstractString)
    return joinpath(@__DIR__, "..", "references", scenario_id, "$(data_id).json")
end

"""
    has_reference(scenario_id, data_id) -> Bool

True iff a committed reference summary exists for this
`(scenario, dataset)` pair.
"""
has_reference(scenario_id::AbstractString, data_id::AbstractString) =
    isfile(reference_path(scenario_id, data_id))

"""
    load_reference(scenario_id, data_id) -> Union{Nothing, ReferenceSummary}

Returns the parsed reference summary, or `nothing` if none exists.
"""
function load_reference(scenario_id::AbstractString, data_id::AbstractString)
    p = reference_path(scenario_id, data_id)
    isfile(p) || return nothing
    json = JSON3.read(read(p, String))
    return ReferenceSummary(
        schema_version = json.schema_version,
        scenario_id = json.scenario_id,
        data_id = json.data_id,
        parameter_names = collect(json.parameter_names),
        posterior_cdf_grids = [collect(Float64, g) for g in json.posterior_cdf_grids],
        posterior_cdf_values = [collect(Float64, v) for v in json.posterior_cdf_values],
        posterior_q025 = collect(json.posterior_q025),
        posterior_q25 = collect(json.posterior_q25),
        posterior_median = collect(json.posterior_median),
        posterior_q75 = collect(json.posterior_q75),
        posterior_q975 = collect(json.posterior_q975),
        posterior_q99 = collect(json.posterior_q99),
        ess = collect(json.ess),
        rhat = collect(json.rhat),
        n_chains = json.n_chains,
        n_samples_per_chain = json.n_samples_per_chain,
        n_warmup = json.n_warmup,
        seed = json.seed,
        timestamp_iso = json.timestamp_iso,
        notes = collect(json.notes),
    )
end

"""
    save_reference(ref::ReferenceSummary)

Write a reference summary to `references/<scenario_id>/<data_id>.json`.
Pretty-printed JSON for diff-friendly review.
"""
function save_reference(ref::ReferenceSummary)
    path = reference_path(ref.scenario_id, ref.data_id)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON3.pretty(io, ref)
    end
    return path
end
