using DynamicPPL: @varname, VarInfo, InitFromPrior, getsym

"""
    _prior_simulate(build_model, y_prototype, rng; replicate_id=0, obs_name=:y) -> SBCReplicate

Forward-sample the joint prior `(θ, x, y)` of a DPPL model.

- `build_model(y)` is a user-provided constructor returning a
  `DynamicPPL.Model` for a given `y`. SBC calls it first with a
  "sample me" prototype (typically `Vector{Missing}`) to draw a
  prior replicate, and later with the simulated `y` to run inference.
- `y_prototype` signals which observations to sample. Normally
  `Vector{Missing}(missing, n)` (or whatever container the model
  expects, with every entry `missing`).
- `rng` controls the draw; callers should supply a replicate-indexed
  `StableRNG` for reproducibility.

Returns an `SBCReplicate` holding the simulated `y`, the rest of the
prior-drawn quantities in a NamedTuple (`truth`), and the replicate id
for book-keeping.

All syms in the model other than `obs_name` are recorded in `truth`,
including any latent field components. The `obs_name` sym is
extracted into `replicate.y` in the container shape of `y_prototype`.
"""
function _prior_simulate(
        build_model, y_prototype, rng::AbstractRNG;
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
