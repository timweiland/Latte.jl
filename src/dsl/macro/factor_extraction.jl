# Loop-preserving extraction of a non-Gaussian latent prior into a
# factor-graph *builder*.
#
# The `@latte` macro normally hands the whole-model log-prior to a single
# opaque `AutoDiffLatentPrior` closure. Differentiating that monolith is what
# makes the first `inla()` call on a state-space model pay a large per-model
# compile cost: the AD pipeline specialises on the closure type, so every new
# user model recompiles from scratch.
#
# Instead we walk the model body, recover each conditional-prior factor
# `x_i ~ Dist(params(parents, θ))` together with its loop nest, and emit a
# builder that assembles a `StructuredLatentPrior` (GMRFs): a tuple of small,
# reusable per-factor log-densities. AD then specialises only on those tiny
# factor functions, which compile once and are cheap.
#
# This is a *performance* refinement of the `:sparse_nongaussian` path — the
# structured prior is verified against the monolithic one at build time and
# falls back on any mismatch, so recognition correctness is unchanged.
#
# Reuses `_detect_tilde_pair` from `ast_walker.jl`; defines a *new* walker
# because `_walk_tilde_blocks` flattens away the loop nest and indices this
# extraction needs.

import GaussianMarkovRandomFields as _GMRFs
import Distributions as _Distributions

# Column-major flat index of the latent entry `x[sym][idx...]`.
# `layout[sym] == (offset, dims)`: `offset` is where `sym`'s block starts in the
# concatenated latent vector, `dims` its array shape (so `(n,)` for a vector,
# `(nrows, ncols)` for a matrix).
function _factor_flat(layout, sym::Symbol, idx::Vararg{Int})
    off, dims = layout[sym]
    return off + LinearIndices(dims)[idx...]
end

# One conditional-prior factor *template* (one `~` site, before loop expansion).
struct _FactorTemplate
    loops::Vector{Any}                              # (loopvar, range_expr), outer→inner
    lsym::Symbol                                    # LHS latent symbol
    lidx::Vector{Any}                               # LHS index exprs
    dist::Any                                       # distribution callee (Symbol/Expr)
    dargs::Vector{Any}                              # distribution argument exprs
    parents::Vector{Tuple{Symbol, Vector{Any}}}     # latent reads in dargs (→ vals[2:])
end

_is_latent_ref(e, latent_syms) =
    e isa Expr && e.head === :ref && e.args[1] in latent_syms

# Collect distinct latent reads (sym + index exprs) appearing in `e`, in order.
function _latent_reads(e, latent_syms, acc = Tuple{Symbol, Vector{Any}}[])
    if _is_latent_ref(e, latent_syms)
        key = (e.args[1], collect(e.args[2:end]))
        any(k -> k[1] === key[1] && k[2] == key[2], acc) || push!(acc, key)
    elseif e isa Expr
        for a in e.args
            _latent_reads(a, latent_syms, acc)
        end
    end
    return acc
end

# Walk a model body, recovering a `_FactorTemplate` for every latent `~` site
# (`latent_syms[i][idx...] ~ Dist(args...)`), preserving the enclosing loops.
# Non-latent `~` sites (hyperparameter priors, observations) are ignored.
function _walk_factor_templates(body, latent_syms)
    out = _FactorTemplate[]
    _walk_factors!(out, body, Any[], latent_syms)
    return out
end

function _walk_factors!(out, e, loops, latent_syms)
    e isa Expr || return

    pair = _detect_tilde_pair(e)
    if pair !== nothing
        lhs, rhs, dotted = pair
        if !dotted && _is_latent_ref(lhs, latent_syms) && rhs isa Expr && rhs.head === :call
            parents = Tuple{Symbol, Vector{Any}}[]
            for a in rhs.args[2:end]
                _latent_reads(a, latent_syms, parents)
            end
            push!(
                out,
                _FactorTemplate(
                    copy(loops), lhs.args[1], collect(lhs.args[2:end]),
                    rhs.args[1], collect(rhs.args[2:end]), parents,
                ),
            )
        end
        return
    end

    if e.head === :for
        iterspec = e.args[1]
        iters = iterspec isa Expr && iterspec.head === :block ? iterspec.args : Any[iterspec]
        pushed = 0
        for it in iters
            it isa LineNumberNode && continue
            if it isa Expr && (it.head === :(=) || it.head === :in || it.head === :∈)
                push!(loops, (it.args[1], it.args[2]))
                pushed += 1
            end
        end
        _walk_factors!(out, e.args[2], loops, latent_syms)
        for _ in 1:pushed
            pop!(loops)
        end
    elseif e.head === :function || e.head === :(->)
        # Don't descend into nested function bodies.
    else
        for a in e.args
            _walk_factors!(out, a, loops, latent_syms)
        end
    end
    return
end

# Substitute latent reads in a distribution-argument expr with `vals[k+1]`, where
# `k` is the parent's position (the factor's own variable is `vals[1]`).
function _factor_subst(e, parents, latent_syms)
    if _is_latent_ref(e, latent_syms)
        key = (e.args[1], collect(e.args[2:end]))
        for (k, (s, ix)) in enumerate(parents)
            (s === key[1] && ix == key[2]) && return :(vals[$(k + 1)])
        end
        error("factor extraction: latent read $(e) was not captured as a parent")
    elseif e isa Expr
        return Expr(e.head, map(a -> _factor_subst(a, parents, latent_syms), e.args)...)
    else
        return e
    end
end

# `(vals, θ) -> logpdf(Dist(<dargs, parents→vals>), vals[1])`, with hyperparameters
# unpacked from `θ` and the θ-only prelude (e.g. `σ = exp(log_σ)`) re-run inside.
function _factor_closure_expr(t::_FactorTemplate, hp_names, prelude, latent_syms)
    unpack = [:($h = θ.$h) for h in hp_names]
    dargs_sub = [_factor_subst(a, t.parents, latent_syms) for a in t.dargs]
    lpcall = Expr(
        :call, GlobalRef(_Distributions, :logpdf),
        Expr(:call, t.dist, dargs_sub...), :(vals[1]),
    )
    bodyblk = Expr(:block, unpack..., prelude..., lpcall)
    return Expr(:->, Expr(:tuple, :vals, :θ), bodyblk)
end

# Re-emit the template's loop nest as explicit `for`s that push each factor's
# flat-index tuple `(self, parents...)` into a `Vector{NTuple{K,Int}}`.
function _factor_index_expr(t::_FactorTemplate)
    flatref(sym, idx) = Expr(
        :call, GlobalRef(@__MODULE__, :_factor_flat), :layout, QuoteNode(sym), idx...,
    )
    flats = Any[flatref(t.lsym, t.lidx)]
    for (s, ix) in t.parents
        push!(flats, flatref(s, ix))
    end
    K = 1 + length(t.parents)
    loopbody = Expr(:call, :push!, :__idx, Expr(:tuple, flats...))
    for (v, rng) in reverse(t.loops)
        loopbody = Expr(:for, Expr(:(=), v, rng), loopbody)
    end
    return Expr(
        :block,
        Expr(:(=), :__idx, :(NTuple{$K, Int}[])),
        loopbody,
        :__idx,
    )
end

"""
    _emit_structured_prior_builder(body, posargs, hp_names, prelude, latent_syms; name)

Build an `Expr` for a *builder closure*

    (layout, n_latent, pattern, posargs...) -> StructuredLatentPrior(...)

extracted from a `@latte` model `body`. `latent_syms` are the array-valued
random symbols whose `~` sites become prior factors; `hp_names` the
hyperparameters (unpacked from the per-factor `θ`); `prelude` the θ-only
assignments (e.g. `σ = exp(log_σ)`) re-run inside each factor; `posargs` the
model's positional argument symbols (threaded so loop ranges resolve).

Returns `nothing` when no latent factor is found (the caller then keeps the
monolithic `AutoDiffLatentPrior`).
"""
function _emit_structured_prior_builder(
        body, posargs, hp_names, prelude, latent_syms; name::Symbol = :structured,
    )
    templates = _walk_factor_templates(body, latent_syms)
    isempty(templates) && return nothing

    group_exprs = Any[]
    for t in templates
        idx_e = _factor_index_expr(t)
        clo_e = _factor_closure_expr(t, hp_names, prelude, latent_syms)
        push!(group_exprs, Expr(:call, GlobalRef(_GMRFs, :LatentFactorGroup), idx_e, clo_e))
    end

    call = Expr(
        :call,
        GlobalRef(_GMRFs, :StructuredLatentPrior),
        Expr(
            :parameters,
            Expr(:kw, :hyperparams, QuoteNode(Tuple(hp_names))),
            Expr(:kw, :name, QuoteNode(name)),
        ),
        :n_latent,
        Expr(:tuple, group_exprs...),
        :pattern,
    )
    params = Any[:layout, :n_latent, :pattern, posargs...]
    return Expr(:->, Expr(:tuple, params...), call)
end
