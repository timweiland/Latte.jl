# InferenceResult protocol implementations for INLAResult.
#
# Wraps the existing INLAResult fields with the shared Tier 1 accessors defined
# in posterior/result_protocol.jl. Field access (r.latent_marginals etc.)
# continues to work unchanged; the protocol forms below return the shape the
# abstract demands (Vector of Distribution, OrderedDict for groups, ...).

using OrderedCollections: OrderedDict

# ─── Marginals ──────────────────────────────────────────────────────────────
latent_marginals(r::INLAResult) = r.latent_marginals

# Internal storage is a NamedTuple keyed by hyperparameter name; the protocol
# returns a positional Vector.
hyperparameter_marginals(r::INLAResult) = collect(values(r.hyperparameter_marginals))

# ─── Groups ─────────────────────────────────────────────────────────────────
# Latent components have no names in a manually-built LGM; return empty. DSL /
# formula layers will populate this later.
latent_groups(::INLAResult) = OrderedDict{Symbol, UnitRange{Int}}()

function hyperparameter_groups(r::INLAResult)
    names = collect(keys(r.hyperparameter_marginals))
    groups = OrderedDict{Symbol, UnitRange{Int}}()
    for (i, name) in enumerate(names)
        groups[name] = i:i
    end
    return groups
end

# ─── Mode (working → natural) ───────────────────────────────────────────────
hyperparameter_mode(r::INLAResult) =
    convert(NaturalHyperparameters, r.hyperparameter_mode)

# ─── Model / observations ───────────────────────────────────────────────────
model(r::INLAResult) = r.model

# Pulls the processed observations INLA actually ran on. For prediction models
# (y contained `missing`), `options.y_obs` is the observed-only subset; we fall
# back to re-deriving it from the raw `options.y`.
function observations(r::INLAResult)
    if haskey(r.options, :y_obs)
        return r.options.y_obs
    end
    y_obs, _, _ = _prepare_for_prediction(r.model, r.options.y)
    return y_obs
end

# ─── Diagnostics ────────────────────────────────────────────────────────────
converged(r::INLAResult) =
    haskey(r.convergence, :mode_converged) ? r.convergence.mode_converged : false

time_elapsed(r::INLAResult) =
    haskey(r.computation_time, :total) ? r.computation_time.total : NaN

# ─── log p(y) ───────────────────────────────────────────────────────────────
# INLA's log marginal likelihood estimate: the log normalization constant from
# the grid / CCD exploration (grid-integral of the Laplace-approximated joint).
# If exploration didn't compute it (shouldn't happen), return nothing.
function log_marginal_likelihood(r::INLAResult)
    expl = r.exploration
    return hasproperty(expl, :log_normalization_constant) ?
        expl.log_normalization_constant : nothing
end
