using OrderedCollections: OrderedDict
using Random

export InferenceResult, PosteriorSamples
export latent_marginals, hyperparameter_marginals
export latent_groups, hyperparameter_groups
export latent_components
export hyperparameter_mode, log_marginal_likelihood
export converged, time_elapsed

"""
    abstract type InferenceResult end

Common supertype for the outputs of inference methods that fit latent Gaussian
models (INLA, TMB, HMC-Laplace, ...). Concrete subtypes implement the Tier 1
protocol below; method-agnostic post-processing (`predict`, `waic`, BMA, ...)
dispatches on this abstract.

See `src/LAYOUT.md` for the full protocol specification.
"""
abstract type InferenceResult end

# ─── Tier 1: required methods (to be implemented per concrete result type) ───
"""
    latent_marginals(r::InferenceResult) -> Vector{<:Distribution}
    latent_marginals(r::InferenceResult, name::Symbol) -> Vector{<:Distribution}

Marginal posterior distributions for the latent field. The vector form returns
all marginals positionally. The name-keyed form returns the slice corresponding
to that latent-field group (e.g. `:β`, `:u`); returns a 1-element vector for
scalar groups.
"""
function latent_marginals end

"""
    hyperparameter_marginals(r::InferenceResult) -> Vector{<:Distribution}
    hyperparameter_marginals(r::InferenceResult, name::Symbol) -> Vector{<:Distribution}

Marginal posterior distributions for the hyperparameters, analogous to
`latent_marginals`. Semantics depend on the method — see concrete
implementations' docstrings (e.g. INLA returns natural-space spline marginals,
TMB returns working-space Gaussian approximations).
"""
function hyperparameter_marginals end

"""
    latent_groups(r::InferenceResult) -> OrderedDict{Symbol, UnitRange{Int}}

Name → index-range mapping for the latent field. Scalar group `:τ` maps to
`i:i`; vector group `:β` of length `p` maps to `a:(a+p-1)`. When no naming
exists (manually-constructed LGM), returns an empty `OrderedDict`.

Populated by DSL / formula layers; empty otherwise.
"""
latent_groups(::InferenceResult) = OrderedDict{Symbol, UnitRange{Int}}()

"""
    hyperparameter_groups(r::InferenceResult) -> OrderedDict{Symbol, UnitRange{Int}}

Name → index-range mapping for the hyperparameters, analogous to
`latent_groups`.
"""
function hyperparameter_groups end

"""
    hyperparameter_mode(r::InferenceResult) -> NaturalHyperparameters

Mode of the hyperparameter posterior in natural space. For INLA this centres
the grid; for TMB this is the MAP (the answer); for HMC-Laplace this is the
warm-start used before sampling.
"""
function hyperparameter_mode end

"""
    converged(r::InferenceResult) -> Bool

Whether the underlying optimisation / exploration converged.
"""
function converged end

"""
    time_elapsed(r::InferenceResult) -> Float64

Total wall-clock time for inference, in seconds.
"""
function time_elapsed end

# Internal accessors — not exported; reference to fitted model & data.
"""
    Latte.model(r::InferenceResult) -> LatentGaussianModel

Return the LGM specification that was fit. Not exported; users typically use
field access. Exists for method-agnostic post-processing.
"""
function model end

"""
    Latte.observations(r::InferenceResult) -> AbstractVector

Return the observations that were used during inference.
"""
function observations end

# ─── Tier 2: optional, with method-dependent semantics ───────────────────────
"""
    log_marginal_likelihood(r::InferenceResult) -> Union{Float64, Nothing}

Approximation to `log p(y)`. Each method produces a different approximation
(INLA: grid integral of Laplace-approx integrand; TMB: Laplace at MAP; HMC:
requires bridge sampling). Returns `nothing` when the method has no natural
way to produce an estimate. Concrete implementations' docstrings spell out
which approximation is computed.
"""
log_marginal_likelihood(::InferenceResult) = nothing

# ─── Default derivations from Tier 1 ─────────────────────────────────────────

function latent_marginals(r::InferenceResult, name::Symbol)
    groups = latent_groups(r)
    haskey(groups, name) || throw(KeyError(name))
    return latent_marginals(r)[groups[name]]
end

function hyperparameter_marginals(r::InferenceResult, name::Symbol)
    groups = hyperparameter_groups(r)
    haskey(groups, name) || throw(KeyError(name))
    return hyperparameter_marginals(r)[groups[name]]
end

# ─── PosteriorSamples ────────────────────────────────────────────────────────
"""
    PosteriorSamples(θ::Matrix, x::Matrix; y=nothing)

Container for `n` joint posterior draws of `(θ, x)` — and optionally `y` for
posterior-predictive samples. Row alignment is enforced at construction:
`θ[i, :]`, `x[i, :]`, and (if present) `y[i, :]` come from the same joint
draw. `θ` values are in natural (user-facing) space.

Iterating / indexing yields a `NamedTuple{(:θ, :x)}` (or `(:θ, :x, :y)` when
`y` was supplied).

Returned by `rand(r::InferenceResult, n)`. For single draws use `rand(r)`
which returns a `NamedTuple` directly.
"""
struct PosteriorSamples{
        Θ <: AbstractMatrix, X <: AbstractMatrix,
        Y <: Union{Nothing, AbstractMatrix},
    }
    θ::Θ
    x::X
    y::Y

    function PosteriorSamples(
            θ::Θ, x::X, y::Y
        ) where {Θ <: AbstractMatrix, X <: AbstractMatrix, Y <: Union{Nothing, AbstractMatrix}}
        size(θ, 1) == size(x, 1) || throw(
            DimensionMismatch(
                "PosteriorSamples: θ has $(size(θ, 1)) rows, x has $(size(x, 1))"
            )
        )
        if y !== nothing && size(y, 1) != size(x, 1)
            throw(
                DimensionMismatch(
                    "PosteriorSamples: y has $(size(y, 1)) rows, x has $(size(x, 1))"
                )
            )
        end
        return new{Θ, X, Y}(θ, x, y)
    end
end

PosteriorSamples(θ::AbstractMatrix, x::AbstractMatrix; y = nothing) =
    PosteriorSamples(θ, x, y)

Base.length(s::PosteriorSamples) = size(s.x, 1)
Base.size(s::PosteriorSamples) = (length(s),)

function Base.getindex(s::PosteriorSamples{<:Any, <:Any, Nothing}, i::Int)
    @boundscheck checkbounds(1:length(s), i)
    return (θ = s.θ[i, :], x = s.x[i, :])
end

function Base.getindex(s::PosteriorSamples{<:Any, <:Any, <:AbstractMatrix}, i::Int)
    @boundscheck checkbounds(1:length(s), i)
    return (θ = s.θ[i, :], x = s.x[i, :], y = s.y[i, :])
end

function Base.iterate(s::PosteriorSamples, state = 1)
    state > length(s) && return nothing
    return (s[state], state + 1)
end
