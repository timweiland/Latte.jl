using DynamicPPL: @varname, VarInfo, InitFromPrior, getsym
import DynamicPPL

"""
    _prior_simulate(build_model, y_prototype, rng; replicate_id=0, obs_name=:y) -> SBCReplicate

Convenience entry point: build the model from `y_prototype` and dispatch
on its type (`DynamicPPL.Model` vs `LatentGaussianModel`) to the
appropriate prior-simulation method below.

Note: an `@latte` LGM factory needs concrete observations to assemble,
so `y_prototype` must be concrete (not `Vector{Missing}`) for the LGM
path. SBC's run loop builds the LGM from a concrete dummy `y` and calls
the typed method directly; this convenience form is primarily for the
DPPL path.
"""
function _prior_simulate(
        build_model, y_prototype, rng::AbstractRNG;
        replicate_id::Int = 0, obs_name::Symbol = :y,
    )
    built = build_model(y_prototype)
    return _prior_simulate(
        built, build_model, y_prototype, rng;
        replicate_id = replicate_id, obs_name = obs_name,
    )
end

"""
    _prior_simulate(built, build_model, y_prototype, rng; replicate_id=0, obs_name=:y) -> SBCReplicate

Forward-sample the joint prior `(θ, x, y)` of a model.

`built = build_model(y_prototype)` is the already-constructed model;
SBC dispatches on its type to pick the right prior-simulation path
(`DynamicPPL.Model` vs `LatentGaussianModel`). `build_model` is kept
around so the DPPL path can re-instantiate the model under the supplied
`rng`.

- `y_prototype` signals which observations to sample. Normally
  `Vector{Missing}(missing, n)` (or whatever container the model
  expects, with every entry `missing`).
- `rng` controls the draw; callers should supply a replicate-indexed
  `StableRNG` for reproducibility.

Returns an `SBCReplicate` holding the simulated `y`, the rest of the
prior-drawn quantities in a NamedTuple (`truth`), and the replicate id
for book-keeping.

For the DPPL path, all syms other than `obs_name` are recorded in
`truth` (including latent field components); the `obs_name` sym is
extracted into `replicate.y` in the container shape of `y_prototype`.
"""
function _prior_simulate(
        ::DynamicPPL.Model, build_model, y_prototype, rng::AbstractRNG;
        replicate_id::Int = 0, obs_name::Symbol = :y,
    )
    model_sim = build_model(y_prototype)
    vi = VarInfo(rng, model_sim, InitFromPrior())

    # Collect all VarNames keyed by their root sym
    by_sym = Dict{Symbol, Vector}()
    for vn in keys(vi)
        push!(get!(by_sym, getsym(vn), []), vn)
    end

    haskey(by_sym, obs_name) || error(
        "_prior_simulate: the model has no sym named `$(obs_name)`. " *
            "Pass `obs_name = :yoursym` to identify the observation variable."
    )

    # Reconstruct y in the prototype's container shape
    y_vns = by_sym[obs_name]
    y_sim = _reconstruct_from_vns(vi, y_vns, y_prototype)

    # Build truth NamedTuple from all non-obs syms, preserving vector shape
    truth_syms = [s for s in keys(by_sym) if s != obs_name]
    truth_vals = Tuple(
        _extract_sym(vi, by_sym[s]) for s in truth_syms
    )
    truth_nt = NamedTuple{Tuple(truth_syms)}(truth_vals)

    return SBCReplicate(replicate_id, truth_nt, y_sim)
end

"""
    _prior_simulate(lgm::LatentGaussianModel, build_model, y_prototype, rng; ...) -> SBCReplicate

Forward-sample the joint prior `(θ, x, y)` of a `LatentGaussianModel`.
`y_prototype` is unused here (the LGM already fixes the observation
shape) but kept for signature parity with the DPPL path.

`truth` is keyed by the LGM's free hyperparameter syms in natural space
(`convert(NamedTuple, θ)`), which is exactly what the hyperparameter
target descriptors index.

The observations are drawn in *observation space*: the latent linear
predictor is sliced through `_x_for_obs_model` before sampling `y`, so
an LGM with an augmented latent yields a `y` aligned with the original
observations (matching what `inla(lgm, y)` expects). The joint latent
draw is kept out of `truth` (which holds only hyperparameters, what the
scalar targets index) but recorded in `SBCReplicate.latent_truth` for
[`DataDependentQuantity`](@ref) to rank against.
"""
function _prior_simulate(
        lgm::LatentGaussianModel, build_model, y_prototype, rng::AbstractRNG;
        replicate_id::Int = 0, obs_name::Symbol = :y,
    )
    spec = lgm.hyperparameter_spec
    θ_working = WorkingHyperparameters([rand(rng, hp.prior) for hp in values(spec.free)], spec)
    θ_natural = convert(NaturalHyperparameters, θ_working)
    θ_nt = convert(NamedTuple, θ_natural)

    x = rand(rng, latent_gmrf(lgm, θ_nt))
    η = _x_for_obs_model(lgm, x)
    y = rand(rng, GaussianMarkovRandomFields.conditional_distribution(lgm.observation_model, η; θ_nt...))

    truth_nt = convert(NamedTuple, θ_natural)
    return SBCReplicate(replicate_id, truth_nt, x, y)
end

"""Reconstruct a `y`-typed container from varinfo entries, preserving
container shape (so a `Vector{Missing}` round-trips to a concrete
`Vector{T}`). Element type is the promotion over observed values."""
function _reconstruct_from_vns(vi, vns, prototype::AbstractVector)
    # Each entry of `y` has its own VarName (e.g. `y[1]`, `y[2]`, ...).
    # Reconstruct by index. Assumes 1-based contiguous indexing.
    n = length(prototype)
    vals = Vector{Any}(undef, n)
    for vn in vns
        i = _index_of(vn)
        i === nothing && error(
            "_prior_simulate: expected indexed obs like `y[i]`, got $(vn)"
        )
        vals[i] = vi[vn]
    end
    T = promote_type(map(typeof, vals)...)
    return Vector{T}(vals)
end

"""Single-sym extraction. Scalars stay scalar; indexed vectors become
concrete vectors in index order."""
function _extract_sym(vi, vns::AbstractVector)
    if length(vns) == 1 && _index_of(vns[1]) === nothing
        return vi[vns[1]]
    end
    max_i = maximum(_index_of.(vns))
    vals = Vector{Any}(undef, max_i)
    for vn in vns
        i = _index_of(vn)
        i === nothing && error(
            "_prior_simulate: mixed scalar+indexed values for sym $(getsym(vn))"
        )
        vals[i] = vi[vn]
    end
    return identity.(vals)
end

"""Return the integer index inside a `VarName` like `y[3]`, or
`nothing` for a scalar `τ`. AbstractPPL's `Index` optic stores the
indices tuple under `.ix`; `Iden()` means scalar."""
function _index_of(vn)
    optic = vn.optic
    hasproperty(optic, :ix) || return nothing
    ix = optic.ix
    return length(ix) == 1 && ix[1] isa Integer ? Int(ix[1]) : nothing
end
