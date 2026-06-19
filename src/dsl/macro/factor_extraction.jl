# Loop-preserving extraction of a non-Gaussian latent prior / observation model into
# factor-graph *builders*.
#
# The `@latte` macro normally hands the whole-model log-prior (and the observation
# log-likelihood) to single opaque AD closures. Differentiating those monoliths is what makes
# the first `inla()` call on a state-space model pay a large per-model compile cost: the AD
# pipeline specialises on the closure type, so every new user model recompiles from scratch.
#
# Instead we walk the model body, recover each conditional factor `x_i ~ Dist(params(parents, θ))`
# (prior) and `y_k ~ Dist(params(latents, θ))` (observation) together with its loop nest, and emit
# builders that assemble a `StructuredLatentPrior` / `StructuredObservationModel` (GMRFs): tuples of
# small, reusable per-factor log-densities. AD then specialises only on those tiny factor functions,
# which compile once and are cheap.
#
# This is a *performance* refinement — the structured prior/obs are verified against the monolithic
# ones at build time and fall back on any mismatch, so recognition correctness is unchanged.
#
# Reuses `_detect_tilde_pair` / `_free_symbols` from `ast_walker.jl`; defines a *new* walker because
# `_walk_tilde_blocks` flattens away the loop nest, indices, and intervening locals this needs.

import GaussianMarkovRandomFields as _GMRFs
import Distributions as _Distributions

# Column-major flat index of the latent entry `x[sym][idx...]`.
# `layout[sym] == (offset, dims)`: `offset` is where `sym`'s block starts in the concatenated
# latent vector, `dims` its array shape (so `(n,)` for a vector, `(nrows, ncols)` for a matrix).
function _factor_flat(layout, sym::Symbol, idx::Vararg{Int})
    off, dims = layout[sym]
    return off + LinearIndices(dims)[idx...]
end

# One conditional-prior factor *template* (one latent `~` site, before loop expansion).
struct _FactorTemplate
    loops::Vector{Any}                              # (loopvar, range_expr), outer→inner
    locals::Vector{Pair{Symbol, Any}}               # loop-body locals lexically preceding this site
    lsym::Symbol                                    # LHS latent symbol
    lidx::Vector{Any}                               # LHS index exprs
    dist::Any                                       # distribution callee (Symbol/Expr)
    dargs::Vector{Any}                              # distribution argument exprs
    parents::Vector{Tuple{Symbol, Vector{Any}}}     # latent reads in dargs/locals (→ vals[2:])
end

# One observation factor template (one `y[idx] ~ Dist(...)` site coupling latents to one datum).
struct _ObsFactorTemplate
    loops::Vector{Any}
    locals::Vector{Pair{Symbol, Any}}
    osym::Symbol                                    # observed (data) symbol
    oidx::Vector{Any}                               # LHS index exprs into the data array
    dist::Any
    dargs::Vector{Any}
    parents::Vector{Tuple{Symbol, Vector{Any}}}     # latent reads (→ vals[1:])
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

_has_latent(e, latent_syms) = !isempty(_latent_reads(e, latent_syms))

# Parents = latent reads in the dist args, then any further latent reads in the closure-scope locals
# (those whose RHS touches a latent). Order fixes the `vals` slot each read maps to.
function _collect_parents(dargs, locals, latent_syms)
    parents = Tuple{Symbol, Vector{Any}}[]
    for a in dargs
        _latent_reads(a, latent_syms, parents)
    end
    for (_l, r) in locals
        _has_latent(r, latent_syms) && _latent_reads(r, latent_syms, parents)
    end
    return parents
end

# ── walker ──

# Walk a model body, recovering prior factor templates (latent `~` sites). Loop-body locals
# lexically preceding each site are captured for inlining.
function _walk_factor_templates(body, latent_syms)
    out = _FactorTemplate[]
    _walk!(out, body, Any[], Pair{Symbol, Any}[], latent_syms, (), :prior)
    return out
end

# Walk a model body, recovering observation factor templates (`obs_syms[k][idx] ~ Dist(...)`).
function _walk_obs_factor_templates(body, latent_syms, obs_syms)
    out = _ObsFactorTemplate[]
    _walk!(out, body, Any[], Pair{Symbol, Any}[], latent_syms, obs_syms, :obs)
    return out
end

function _walk!(out, e, loops, locals, latent_syms, obs_syms, mode)
    e isa Expr || return

    pair = _detect_tilde_pair(e)
    if pair !== nothing
        _record_tilde!(out, pair, loops, locals, latent_syms, obs_syms, mode)
        return
    end

    if e.head === :for
        iterspec = e.args[1]
        iters = iterspec isa Expr && iterspec.head === :block ? iterspec.args : Any[iterspec]
        pushed = 0
        for it in iters
            if it isa Expr && (it.head === :(=) || it.head === :in || it.head === :∈)
                push!(loops, (it.args[1], it.args[2]))
                pushed += 1
            end
        end
        savelen = length(locals)
        _walk!(out, e.args[2], loops, locals, latent_syms, obs_syms, mode)
        resize!(locals, savelen)
        for _ in 1:pushed
            pop!(loops)
        end
    elseif e.head === :function || e.head === :(->)
        # Don't descend into nested function bodies.
    elseif e.head === :block
        savelen = length(locals)
        for a in e.args
            # Capture scalar assignments inside loops as candidate factor locals; recurse otherwise.
            if !isempty(loops) && a isa Expr && a.head === :(=) && a.args[1] isa Symbol
                push!(locals, a.args[1] => a.args[2])
            else
                _walk!(out, a, loops, locals, latent_syms, obs_syms, mode)
            end
        end
        resize!(locals, savelen)
    else
        for a in e.args
            _walk!(out, a, loops, locals, latent_syms, obs_syms, mode)
        end
    end
    return
end

# Lower a broadcast prior `u .~ Dist.(args…)` to a synthetic-loop element factor. Returns
# `(bvar, brange, elem_lhs, elem_rhs)` — a fresh loop variable over `u`'s linear indices, the loop
# range, and the element forms `u[bvar] ~ Dist(elementwise(args)…)` — or `nothing` to fall back to
# the monolithic prior. Supported: a bare latent LHS with scalar args (IID) and bare latent-symbol
# args (element-wise coupling, rewritten to `v[bvar]`). An explicit latent index/slice in an arg
# (`v[1:n-1]`) is not aligned here and falls back.
function _lower_broadcast_prior(lhs, rhs, latent_syms)
    (lhs isa Symbol && lhs in latent_syms) || return nothing
    (rhs isa Expr && rhs.head === :. && length(rhs.args) == 2) || return nothing
    bargs_node = rhs.args[2]
    (bargs_node isa Expr && bargs_node.head === :tuple) || return nothing
    bargs = bargs_node.args
    any(a -> _has_latent_index(a, latent_syms), bargs) && return nothing

    bvar = gensym(:bcast)
    elem_lhs = Expr(:ref, lhs, bvar)
    elem_rhs = Expr(:call, rhs.args[1], Any[_index_bare_latents(a, bvar, latent_syms) for a in bargs]...)
    # `u`'s entry count, via its layout dims — evaluated in builder scope where `layout` is bound.
    brange = Expr(:call, :(:), 1, Expr(:call, :prod, Expr(:ref, Expr(:ref, :layout, QuoteNode(lhs)), 2)))
    return (bvar, brange, elem_lhs, elem_rhs)
end

# Does `e` reference a latent through an explicit index/slice (`v[…]`)?
_has_latent_index(e, latent_syms) =
    e isa Expr && (_is_latent_ref(e, latent_syms) || any(a -> _has_latent_index(a, latent_syms), e.args))

# Rewrite bare latent symbols `v` (whole-array references being broadcast) to element form `v[bvar]`,
# leaving everything else untouched. Callers guarantee no explicit latent index survives here.
function _index_bare_latents(e, bvar, latent_syms)
    e isa Symbol && e in latent_syms && return Expr(:ref, e, bvar)
    e isa Expr || return e
    return Expr(e.head, Any[_index_bare_latents(a, bvar, latent_syms) for a in e.args]...)
end

function _record_tilde!(out, pair, loops, locals, latent_syms, obs_syms, mode)
    lhs, rhs, dotted = pair
    if dotted
        # A broadcast prior `u .~ Dist.(args…)` is element-wise: lower it to a synthetic loop over
        # `u`'s entries (`u[i] ~ Dist(args…[i])`) and reuse the ordinary factor codegen. Only latent
        # priors structure; a dotted observation keeps the monolithic path (returns `nothing`).
        mode === :prior || return
        lowered = _lower_broadcast_prior(lhs, rhs, latent_syms)
        lowered === nothing && return
        bvar, brange, elem_lhs, elem_rhs = lowered
        push!(loops, (bvar, brange))
        _record_tilde!(out, (elem_lhs, elem_rhs, false), loops, locals, latent_syms, obs_syms, mode)
        pop!(loops)
        return
    end
    (rhs isa Expr && rhs.head === :call) || return
    dargs = collect(rhs.args[2:end])
    parents = _collect_parents(dargs, locals, latent_syms)
    if mode === :prior && _is_latent_ref(lhs, latent_syms)
        push!(
            out,
            _FactorTemplate(
                copy(loops), copy(locals), lhs.args[1], collect(lhs.args[2:end]),
                rhs.args[1], dargs, parents,
            ),
        )
    elseif mode === :obs && lhs isa Expr && lhs.head === :ref &&
            lhs.args[1] in obs_syms && length(lhs.args) >= 2 && !isempty(parents)
        push!(
            out,
            _ObsFactorTemplate(
                copy(loops), copy(locals), lhs.args[1], collect(lhs.args[2:end]),
                rhs.args[1], dargs, parents,
            ),
        )
    end
    return
end

# ── codegen helpers ──

# Substitute latent reads with `vals[k + offset]` (`offset = 1` for priors, where `vals[1]` is the
# factor's own variable; `offset = 0` for observations, where all `vals` are latent parents).
function _subst(e, parents, latent_syms, offset)
    if _is_latent_ref(e, latent_syms)
        key = (e.args[1], collect(e.args[2:end]))
        for (k, (s, ix)) in enumerate(parents)
            (s === key[1] && ix == key[2]) && return :(vals[$(k + offset)])
        end
        error("factor extraction: latent read $(e) was not captured as a parent")
    elseif e isa Expr
        return Expr(e.head, map(a -> _subst(a, parents, latent_syms, offset), e.args)...)
    else
        return e
    end
end

# Loop-body locals that DON'T touch a latent — emitted inside the index loop (loop vars / posargs in
# scope) so factor indices like `logN[jp]` resolve.
_index_locals(locals, latent_syms) =
    Any[Expr(:(=), l, r) for (l, r) in locals if !_has_latent(r, latent_syms)]

# Loop-body locals that DO touch a latent — emitted inside the factor closure with latent reads
# substituted (e.g. `Z = exp(vals[2]) + M`).
_closure_locals(locals, parents, latent_syms, offset) =
    Any[Expr(:(=), l, _subst(r, parents, latent_syms, offset)) for (l, r) in locals if _has_latent(r, latent_syms)]

# The prelude assignments (and hyperparameters) a factor actually needs, by backward-chaining from
# the free symbols of its dist args and closure-scope locals. Keeps each closure minimal — and lets
# the observation closure unpack only the hyperparameters its likelihood depends on.
function _needed_prelude(exprs, prelude, hp_names)
    needed = Set{Symbol}()
    for e in exprs
        union!(needed, _free_symbols(e, Set{Symbol}(), Dict{Symbol, Set{Symbol}}()))
    end
    keep = falses(length(prelude))
    for i in length(prelude):-1:1
        lhs = prelude[i].args[1]
        if lhs isa Symbol && lhs in needed
            keep[i] = true
            union!(needed, _free_symbols(prelude[i].args[2], Set{Symbol}(), Dict{Symbol, Set{Symbol}}()))
        end
    end
    kept = Any[prelude[i] for i in 1:length(prelude) if keep[i]]
    needed_hp = Tuple(h for h in hp_names if h in needed)
    return kept, needed_hp
end

_flatref(sym, idx) =
    Expr(:call, GlobalRef(@__MODULE__, :_factor_flat), :layout, QuoteNode(sym), idx...)

# ── prior factor codegen ──

function _factor_closure_expr(t::_FactorTemplate, hp_names, prelude, latent_syms)
    clocals = _closure_locals(t.locals, t.parents, latent_syms, 1)
    dargs_sub = [_subst(a, t.parents, latent_syms, 1) for a in t.dargs]
    pre_inputs = vcat(Any[r for (_l, r) in t.locals if _has_latent(r, latent_syms)], t.dargs)
    kept_prelude, needed_hp = _needed_prelude(pre_inputs, prelude, hp_names)
    unpack = Any[:($h = θ.$h) for h in needed_hp]
    lpcall = Expr(:call, GlobalRef(_Distributions, :logpdf), Expr(:call, t.dist, dargs_sub...), :(vals[1]))
    body = Expr(:block, unpack..., kept_prelude..., clocals..., lpcall)
    return Expr(:->, Expr(:tuple, :vals, :θ), body)
end

function _factor_index_expr(t::_FactorTemplate, latent_syms)
    flats = Any[_flatref(t.lsym, t.lidx)]
    for (s, ix) in t.parents
        push!(flats, _flatref(s, ix))
    end
    K = 1 + length(t.parents)
    # Pass the real `latent_syms` so latent-touching loop-body locals are kept OUT of the index loop
    # (they're inlined into the closure instead, with reads substituted) — parity with the obs side.
    loopbody = Expr(:block, _index_locals(t.locals, latent_syms)..., Expr(:call, :push!, :__idx, Expr(:tuple, flats...)))
    for (v, rng) in reverse(t.loops)
        loopbody = Expr(:for, Expr(:(=), v, rng), loopbody)
    end
    return Expr(:block, Expr(:(=), :__idx, :(NTuple{$K, Int}[])), loopbody, :__idx)
end

# ── observation factor codegen ──

# Returns `(group_expr, needed_hp)`.
function _obs_group_expr(t::_ObsFactorTemplate, hp_names, prelude, latent_syms)
    clocals = _closure_locals(t.locals, t.parents, latent_syms, 0)
    dargs_sub = [_subst(a, t.parents, latent_syms, 0) for a in t.dargs]
    pre_inputs = vcat(Any[r for (_l, r) in t.locals if _has_latent(r, latent_syms)], t.dargs)
    kept_prelude, needed_hp = _needed_prelude(pre_inputs, prelude, hp_names)
    unpack = Any[:($h = θ.$h) for h in needed_hp]
    lpcall = Expr(:call, GlobalRef(_Distributions, :logpdf), Expr(:call, t.dist, dargs_sub...), :yk)
    closure = Expr(:->, Expr(:tuple, :vals, :yk, :θ), Expr(:block, unpack..., kept_prelude..., clocals..., lpcall))

    K = length(t.parents)
    var_flats = Any[_flatref(s, ix) for (s, ix) in t.parents]
    # The observation's flat (column-major) position in its data array, via `LinearIndices` so a
    # multi-indexed datum `y[i, j]` maps correctly (and a flat `y[k]` stays `k`). `__obs_li` is the
    # data array's linear-index map, hoisted out of the loop; the data symbol is a builder param.
    oidx_expr = Expr(:ref, :__obs_li, t.oidx...)
    loopbody = Expr(
        :block, _index_locals(t.locals, latent_syms)...,
        Expr(:call, :push!, :__vars, Expr(:tuple, var_flats...)),
        Expr(:call, :push!, :__oidx, oidx_expr),
    )
    for (v, rng) in reverse(t.loops)
        loopbody = Expr(:for, Expr(:(=), v, rng), loopbody)
    end
    group = Expr(
        :let,
        Expr(
            :block, :(__vars = NTuple{$K, Int}[]), :(__oidx = Int[]),
            :(__obs_li = LinearIndices(size($(t.osym)))),
        ),
        Expr(:block, loopbody, Expr(:call, GlobalRef(_GMRFs, :ObsFactorGroup), :__vars, :__oidx, closure)),
    )
    return group, needed_hp
end

# ── builders ──

# Wrap a builder body with any model-constant prelude (posarg-derived top-level scalars like
# `n = nA * nY`) so loop ranges / shapes inside resolve.
_wrap_consts(model_consts, call) =
    isempty(model_consts) ? call : Expr(:block, model_consts..., call)

"""
    _emit_structured_prior_builder(body, posargs, hp_names, prelude, latent_syms; name, model_consts)

Build an `Expr` for a builder closure `(layout, n_latent, pattern, posargs...) -> StructuredLatentPrior(...)`
extracted from a `@latte` model `body`. `latent_syms` are the array-valued random symbols whose `~`
sites become prior factors; `hp_names` the hyperparameters; `prelude` the closure-visible assignments
(e.g. `σ = exp(log_σ)`, `M = 0.2`); `posargs` the model's positional argument symbols; `model_consts`
posarg-derived top-level scalars emitted at builder scope. Returns `nothing` when no latent factor is
found.
"""
function _emit_structured_prior_builder(
        body, posargs, hp_names, prelude, latent_syms; name::Symbol = :structured, model_consts = Any[],
    )
    templates = _walk_factor_templates(body, latent_syms)
    isempty(templates) && return nothing

    group_exprs = Any[]
    for t in templates
        push!(
            group_exprs,
            Expr(
                :call, GlobalRef(_GMRFs, :LatentFactorGroup),
                _factor_index_expr(t, latent_syms), _factor_closure_expr(t, hp_names, prelude, latent_syms),
            ),
        )
    end

    call = Expr(
        :call,
        GlobalRef(_GMRFs, :StructuredLatentPrior),
        Expr(
            :parameters,
            Expr(:kw, :hyperparams, QuoteNode(Tuple(hp_names))),
            Expr(:kw, :name, QuoteNode(name)),
        ),
        :n_latent, Expr(:tuple, group_exprs...), :pattern,
    )
    params = Any[:layout, :n_latent, :pattern, posargs...]
    return Expr(:->, Expr(:tuple, params...), _wrap_consts(model_consts, call))
end

"""
    _emit_structured_obs_builder(body, posargs, hp_names, prelude, latent_syms, obs_syms; model_consts)

Build an `Expr` for a builder closure `(layout, n_latent, posargs...) -> StructuredObservationModel(...)`
extracted from a `@latte` model `body`. Each `obs_syms[k][idx] ~ Dist(params(latents, θ))` site becomes
an observation factor. Returns `nothing` when no observation factor is found.
"""
function _emit_structured_obs_builder(
        body, posargs, hp_names, prelude, latent_syms, obs_syms; model_consts = Any[],
    )
    templates = _walk_obs_factor_templates(body, latent_syms, obs_syms)
    isempty(templates) && return nothing

    group_exprs = Any[]
    obs_hp = Symbol[]
    for t in templates
        g, needed_hp = _obs_group_expr(t, hp_names, prelude, latent_syms)
        push!(group_exprs, g)
        for h in needed_hp
            h in obs_hp || push!(obs_hp, h)
        end
    end

    call = Expr(
        :call,
        GlobalRef(_GMRFs, :StructuredObservationModel),
        Expr(:parameters, Expr(:kw, :hyperparams, QuoteNode(Tuple(obs_hp)))),
        :n_latent, Expr(:tuple, group_exprs...),
    )
    params = Any[:layout, :n_latent, posargs...]
    return Expr(:->, Expr(:tuple, params...), _wrap_consts(model_consts, call))
end

# ── macro-time inputs derived from the model body ──

# Split the body's top-level scalar assignments into:
#   - `closure_prelude`: assignments visible inside a factor closure — hyperparameter-derived
#     (`σ = exp(log_σ)`) or plain constants (`M = 0.2`); RHS free of positional args and latents.
#   - `model_consts`: positional-arg-derived constants (`n = nA * nY`) — emitted at builder scope.
# Latent container declarations (`logN = Matrix{Real}(undef, …)`) are skipped (handled by shapes).
function _split_top_level_locals(body, hp_names, posargs, latent_syms)
    closure_prelude = Any[]
    model_consts = Any[]
    (body isa Expr && body.head === :block) || return closure_prelude, model_consts
    blocked = union(Set{Symbol}(posargs), Set{Symbol}(latent_syms))   # not closure-visible
    for stmt in body.args
        (stmt isa Expr && stmt.head === :(=) && stmt.args[1] isa Symbol) || continue
        lhs = stmt.args[1]
        lhs in blocked && continue
        rhs_syms = _free_symbols(stmt.args[2], Set{Symbol}(), Dict{Symbol, Set{Symbol}}())
        if isempty(intersect(rhs_syms, blocked))
            push!(closure_prelude, stmt)
        else
            push!(model_consts, stmt)
            push!(blocked, lhs)
        end
    end
    return closure_prelude, model_consts
end

# Recover each latent array's allocation RHS (`logN = Matrix{Real}(undef, nA, nY)` → the RHS expr).
# The layout builder takes `size(...)` of it at build time, which is robust to the allocator form
# (Matrix/Vector/Array `undef`, `zeros`/`ones`/`fill` with or without an eltype, comprehensions,
# `similar`). `nothing` unless every latent symbol has a top-level allocation.
function _extract_latent_allocs(body, latent_syms)
    (body isa Expr && body.head === :block) || return nothing
    allocs = Dict{Symbol, Any}()
    for stmt in body.args
        (stmt isa Expr && stmt.head === :(=)) || continue
        lhs = stmt.args[1]
        (lhs isa Symbol && lhs in latent_syms) || continue
        haskey(allocs, lhs) || (allocs[lhs] = stmt.args[2])
    end
    all(s -> haskey(allocs, s), latent_syms) || return nothing
    return allocs
end

# `(posargs...) -> Dict(sym => (offset, dims))`: the concatenated-latent layout the flat-index map
# needs. `latent_syms` must be in base-latent concatenation order (offsets accumulate in that order).
# Each latent's `dims` is `size(<its allocation>)`, evaluated once in builder scope.
function _emit_structured_layout_builder(body, posargs, latent_syms, model_consts)
    allocs = _extract_latent_allocs(body, latent_syms)
    allocs === nothing && return nothing
    stmts = Any[model_consts...; :(__off = 0); :(__layout = Dict{Symbol, Tuple{Int, Tuple}}())]
    for s in latent_syms
        dsym = Symbol("__dims_", s)
        push!(stmts, :($dsym = size($(allocs[s]))))
        push!(stmts, :(__layout[$(QuoteNode(s))] = (__off, $dsym)))
        push!(stmts, :(__off += prod($dsym)))
    end
    push!(stmts, :__layout)
    return Expr(:->, Expr(:tuple, posargs...), Expr(:block, stmts...))
end

# Wrap a builder lambda `(args...) -> body` as a named top-level function `function name(args...) ...`.
function _named_function(name::Symbol, lambda::Expr)
    sig, fbody = lambda.args[1], lambda.args[2]
    params = (sig isa Expr && sig.head === :tuple) ? sig.args : Any[sig]
    return Expr(:function, Expr(:call, name, params...), fbody)
end

"""
    _emit_structured_support(fname, body, posargs, hp_names, latent_syms, obs_syms)

Macro-time orchestration: emit named function definitions for a structured-prior `builder`, an
optional structured-observation `obs_builder`, and a `layout_builder` for a `@latte` model `fname`.
Returns `(; builder_def, obs_def, layout_def, builder_name, obs_name, layout_name)` (with `obs_def`
/ `obs_name` possibly `nothing`), or `nothing` when there are no extractable latent factors /
recoverable shapes (caller then keeps the monolithic prior).
"""
function _emit_structured_support(fname, body, posargs, hp_names, latent_syms, obs_syms)
    isempty(latent_syms) && return nothing
    prelude, model_consts = _split_top_level_locals(body, hp_names, posargs, latent_syms)

    builder_lambda = _emit_structured_prior_builder(
        body, posargs, hp_names, prelude, latent_syms; model_consts = model_consts,
    )
    builder_lambda === nothing && return nothing
    layout_lambda = _emit_structured_layout_builder(body, posargs, latent_syms, model_consts)
    layout_lambda === nothing && return nothing

    obs_lambda = _emit_structured_obs_builder(
        body, posargs, hp_names, prelude, latent_syms, obs_syms; model_consts = model_consts,
    )

    builder_name = Symbol("__latte_sprior_builder_", fname)
    layout_name = Symbol("__latte_slayout_", fname)
    obs_name = obs_lambda === nothing ? nothing : Symbol("__latte_sobs_builder_", fname)
    return (
        builder_def = _named_function(builder_name, builder_lambda),
        obs_def = obs_lambda === nothing ? nothing : _named_function(obs_name, obs_lambda),
        layout_def = _named_function(layout_name, layout_lambda),
        builder_name = builder_name,
        obs_name = obs_name,
        layout_name = layout_name,
    )
end
