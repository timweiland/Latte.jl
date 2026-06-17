# Latent-field initialization for the inner Gaussian-approximation Newton solve.
#
# This is a DIFFERENT axis from `mode_init` (mode_init.jl): `mode_init` seeds the OUTER
# hyperparameter (θ) optimizer; `latent_init` seeds the INNER latent (x) Newton's FIRST
# solve. Subsequent solves warm-start from the previous mode (see `find_hyperparameter_mode`),
# so only the first inner start is set here.
#
# The default `ZeroLatentStart` reproduces the historical zeros behavior, which is fine in the
# common case: a Gaussian latent's GA mode is start-invariant, and for an observed non-Gaussian
# latent the data gradient pulls the inner Newton to the right level from zeros.
#
# `AutoLatentStart` is an opt-in *best-effort* prior-mode seed (the prior's own GA against a zero
# likelihood). It helps when the prior mode is a better seed than zeros, but it is NOT guaranteed
# to find the true mode for a stiff field far from zero — the prior solve from zeros can stall on
# the same nonlinearity (e.g. `exp(x)`), so it falls back to zeros rather than ship a bad seed.
# For those hard cases pass an explicit `Vector` (TMB-style initial random-effect values). A robust
# deterministic forward-rollout `AutoLatentStart` is a planned follow-up.

export LatentStartStrategy, AutoLatentStart, ZeroLatentStart, resolve_latent_start

"""
    LatentStartStrategy

Abstract supertype for `latent_init` strategies. A `latent_init` may also be a plain
`AbstractVector` (an explicit latent start, like TMB's initial random-effect values).
"""
abstract type LatentStartStrategy end

"""
    AutoLatentStart()

Opt-in best-effort seed: zeros for a Gaussian latent prior (the GA mode is start-invariant there),
and the prior's own mode for a `NonGaussianLatentPrior`. The prior-mode solve is the prior's GA
against a zero likelihood, and is *guarded* — a non-converged or non-finite result falls back to
zeros (never worse than `ZeroLatentStart`). Not guaranteed to reach the true mode for a stiff field
far from zero; for those, pass an explicit `Vector`.
"""
struct AutoLatentStart <: LatentStartStrategy end

"""
    ZeroLatentStart()

Default. Force the inner Newton to start from zeros (the historical behavior) regardless of prior type.
"""
struct ZeroLatentStart <: LatentStartStrategy end

"""
    resolve_latent_start(latent_init, model, θ_natural_nt) -> Union{Nothing, Vector{Float64}}

Resolve a `latent_init` to a concrete first-solve latent start at hyperparameters `θ_natural_nt`,
or `nothing` (⇒ the GA's zeros default).
"""
resolve_latent_start(::ZeroLatentStart, ::LatentGaussianModel, ::NamedTuple) = nothing

function resolve_latent_start(::AutoLatentStart, model::LatentGaussianModel, θ_natural_nt::NamedTuple)
    prior = model.latent_prior
    prior isa NonGaussianLatentPrior || return nothing       # Gaussian ⇒ zeros (start-invariant)
    # Prior mode = argmax log π(x | θ): the iterated-Laplace GA against the identity (zero)
    # likelihood. Guarded: a non-converged prior-mode solve must not be worse than zeros.
    post = try
        gaussian_approximation(prior, ZeroLikelihood(); θ = θ_natural_nt)
    catch
        return nothing
    end
    x0 = collect(Float64, mean(post))
    return all(isfinite, x0) ? x0 : nothing
end

function resolve_latent_start(x0::AbstractVector, model::LatentGaussianModel, ::NamedTuple)
    n = length(model.latent_prior)
    length(x0) == n || throw(
        ArgumentError(
            "latent_init vector has length $(length(x0)); expected $n (the latent dimension)."
        )
    )
    return collect(Float64, x0)
end
