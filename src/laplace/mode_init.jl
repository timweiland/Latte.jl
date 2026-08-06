# Mode-finder initialisation strategies.
#
# `find_hyperparameter_mode` historically always started BFGS from the
# prior mode in working space, which sticks at a local maximum on
# non-convex hp posteriors and reports convergence anyway — a silent
# failure mode. This file introduces a `mode_init` kwarg that lets the
# user supply explicit starts or request multi-start. The user types
# values in natural space (matching how hp values are exposed
# elsewhere in the package); we convert to working space internally
# via the existing bijector chain and validate.
#
# Strategy hierarchy:
#   ModeStartStrategy            (abstract)
#     PriorModeStart             (default — current behaviour)
#     RandomStarts               (random working-space starts)
#
# In addition, `mode_init` may be a `NamedTuple` (single start) or a
# `Vector{<:NamedTuple}` (multi-start). `resolve_mode_starts` dispatches.

import Random: AbstractRNG, MersenneTwister

export ModeStartStrategy, PriorModeStart, RandomStarts

"""
    ModeStartStrategy

Abstract supertype for `mode_init` strategies. Subtypes implement
`resolve_mode_starts(strategy, spec) -> Vector{WorkingHyperparameters}`.
"""
abstract type ModeStartStrategy end

"""
    PriorModeStart()

Default strategy: start BFGS once from the prior mode in working
space. Reproduces the historical behaviour of `find_hyperparameter_mode`.
"""
struct PriorModeStart <: ModeStartStrategy end

"""
    RandomStarts(n; rng = Random.default_rng(), σ = 1.0)

Sample `n` working-space starts from a zero-centred isotropic Normal
with scale `σ`. The prior is already centred at zero in working space
(via the bijector), so this is a coarse "spread around the prior
centre" heuristic. Useful when no domain knowledge is available about
where the posterior mode might be.
"""
struct RandomStarts{R <: AbstractRNG} <: ModeStartStrategy
    n::Int
    rng::R
    σ::Float64
end
function RandomStarts(n::Int; rng::AbstractRNG = MersenneTwister(), σ::Real = 1.0)
    return RandomStarts(n, rng, Float64(σ))
end

# ─── resolve_mode_starts dispatch ─────────────────────────────────────────────

"""
    resolve_mode_starts(mode_init, spec::HyperparameterSpec)
        -> Vector{WorkingHyperparameters}

Convert a user-supplied `mode_init` value into a vector of concrete
working-space starts. Validates hp names and finiteness of the
resulting working coordinates; throws `ArgumentError` with the
offending hp name and value if anything is wrong.
"""
function resolve_mode_starts(::PriorModeStart, spec::HyperparameterSpec)
    return [initial_hyperparameter_guess(spec)]
end

function resolve_mode_starts(strategy::RandomStarts, spec::HyperparameterSpec)
    n_hp = _hp_total_dim(spec)
    base = initial_hyperparameter_guess(spec).θ
    starts = WorkingHyperparameters[]
    for _ in 1:strategy.n
        θ = base .+ strategy.σ .* randn(strategy.rng, n_hp)
        push!(starts, WorkingHyperparameters(θ, spec))
    end
    return starts
end

function resolve_mode_starts(nt::NamedTuple, spec::HyperparameterSpec)
    return [_resolve_single_named_tuple(nt, spec)]
end

function resolve_mode_starts(starts::AbstractVector, spec::HyperparameterSpec)
    isempty(starts) && throw(ArgumentError("mode_init: empty Vector of starts is not allowed"))
    return [_resolve_single_named_tuple(nt, spec) for nt in starts]
end

# Convert one user-supplied natural-space NamedTuple to a
# WorkingHyperparameters. Validates hp names against the spec and
# checks the resulting working coords are finite.
function _resolve_single_named_tuple(nt::NamedTuple, spec::HyperparameterSpec)
    hp_names = keys(spec.free)
    supplied = keys(nt)

    # Missing names.
    missing_names = setdiff(hp_names, supplied)
    isempty(missing_names) || throw(
        ArgumentError(
            "mode_init NamedTuple is missing hyperparameter(s) " *
                "$(collect(missing_names)); supply all free hp names " *
                "$(collect(hp_names)) in natural space"
        )
    )

    # Unknown names (typos / stale keys).
    extra_names = setdiff(supplied, hp_names)
    isempty(extra_names) || throw(
        ArgumentError(
            "mode_init NamedTuple has unknown hyperparameter(s) " *
                "$(collect(extra_names)); free hp names are $(collect(hp_names))"
        )
    )

    # Build NaturalHyperparameters in canonical spec order (flattening
    # vector-valued entries in place), then convert.
    natural_vec = _flatten_hp_namedtuple(nt, spec)
    θ_natural = NaturalHyperparameters(natural_vec, spec)
    θ_working = convert(WorkingHyperparameters, θ_natural)

    # Validate finiteness in working space (catches e.g. log(0)).
    coord_names = _expanded_hp_names(spec)
    for (i, v) in enumerate(θ_working.θ)
        isfinite(v) || throw(
            ArgumentError(
                "mode_init: natural value $(coord_names[i]) = $(natural_vec[i]) maps to non-finite " *
                    "working-space coordinate ($v). This usually means the value is at " *
                    "or past a hard boundary (e.g. 0 for a log-transformed positive prior)."
            )
        )
    end
    return θ_working
end

# Backstop for invalid types.
function resolve_mode_starts(other, ::HyperparameterSpec)
    throw(
        ArgumentError(
            "mode_init has unsupported type $(typeof(other)). Use one of: " *
                "PriorModeStart(), RandomStarts(n), a NamedTuple of natural-space hp values, " *
                "or a Vector{NamedTuple}."
        )
    )
end

# ─── Post-exploration mode-quality diagnostic ─────────────────────────────────
# After `explore_hyperparameter_posterior` runs we have a grid of points
# centred at θ*. If any of them has a log-density meaningfully higher
# than θ*'s, the mode finder probably stuck at a local maximum.

"""
    _diagnose_mode_quality(θ_star, exploration, model, mode, tol)

Compare θ*'s log-density to the best log-density found during exploration.
If the gap exceeds `tol`, take action per `mode` ∈ (`:none`, `:warn`,
`:error`). Default `:warn`.

The mode point is located by matching `θ_star` against the grid, not by
position: the exploration stores points in coordinate order with the mode
in the interior, so `grid_points[1]` is a grid edge, not the mode.
"""
function _diagnose_mode_quality(
        θ_star::WorkingHyperparameters, exploration, model::LatentGaussianModel,
        mode::Symbol, tol::Float64,
    )
    mode === :none && return nothing
    mode in (:warn, :error) || throw(
        ArgumentError("mode_diagnostic must be :none, :warn, or :error; got $(mode)")
    )

    pts = exploration.grid_points
    isempty(pts) && return nothing

    log_densities = [p.log_density for p in pts]
    best_idx = argmax(log_densities)
    best_logp = log_densities[best_idx]

    # Find the grid point at the optimizer's mode θ_star. The mode point is
    # inserted during exploration, so an exact match exists; fall back to the
    # nearest node if float noise prevents one.
    θ_star_vec = θ_star.θ
    mode_idx = something(
        findfirst(p -> _θ_close(p.θ.θ, θ_star_vec), pts),
        argmin([sum(abs2, p.θ.θ .- θ_star_vec) for p in pts]),
    )
    mode_logp = log_densities[mode_idx]

    gap = best_logp - mode_logp
    gap > tol || return nothing

    spec = model.hyperparameter_spec
    θ_best_nt = convert(NamedTuple, convert(NaturalHyperparameters, pts[best_idx].θ))
    θ_mode_nt = convert(NamedTuple, convert(NaturalHyperparameters, pts[mode_idx].θ))
    msg = "Mode-quality diagnostic: a better hyperparameter point was found " *
        "during exploration than at the optimizer's mode " *
        "(gap = $(round(gap; digits = 3)) log units, tol = $(tol)). " *
        "Picked mode: $(θ_mode_nt). " *
        "Better point in grid: $(θ_best_nt). " *
        "Consider rerunning with mode_init = [$(θ_best_nt), ...] or a multi-start strategy."
    if mode === :warn
        @warn msg
    else
        error(msg)
    end
    return nothing
end

_θ_close(a::AbstractVector, b::AbstractVector) =
    length(a) == length(b) && all(isapprox.(a, b; atol = 1.0e-10))
