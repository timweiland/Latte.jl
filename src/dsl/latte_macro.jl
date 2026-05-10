# `@latte` — high-level adapter macro for Latte.jl.
#
# Wraps a DPPL `@model` with static AST analysis that auto-detects
# observation grouping by hyperparameter dependency, plus user-driven
# `@random` / `@fixed` markers for explicit control over which sample sites
# are random effects (Laplace-marginalised) vs hyperparameters / fixed
# effects (grid- or MAP-handled by the inference method).
#
# Defaults (when no marker present):
# - LHS is a positional argument of the `@model` function ⇒ observation.
# - LHS is a fresh symbol AND RHS callee is a known random-effect-shaped
#   constructor (`MvNormal`, `IIDModel`, `RWModel`, `BesagModel`,
#   `MaternModel`, `BYM2Model`, `SeparableModel`, `GMRF`,
#   `ConstrainedGMRF`) ⇒ random effect.
# - Anything else ⇒ fixed effect (hyperparameter).
#
# `@random` / `@fixed` markers override the default for a single `~`.
# Outside `@latte` the markers are no-op identity macros so the body still
# compiles inside a plain `@model`.
#
# Turing handoff: every `@latte`-defined model also gets an underlying
# DPPL `@model`-built constructor, accessible via `Latte.dppl_model(name)`.
# Same body, no markers — Turing's `sample(NUTS(), ...)` works directly.

using DynamicPPL: @model
import DynamicPPL
using GaussianMarkovRandomFields:
    AutoDiffObservationModel, CompositeObservationModel,
    CompositeObservations

export @latte, @random, @fixed
export latte_analysis, dppl_model

# ─── Marker macros (no-op outside @latte) ─────────────────────────────────────
"""
    @random a ~ b
    @fixed  a ~ b

Marker macros for use inside `@latte` model bodies. Override the default
classification of a `~` block:

- `@random` marks the block as a random effect (Laplace-marginalised latent).
- `@fixed` marks the block as a fixed effect (hyperparameter).

Outside `@latte` they're identity passthroughs, so a `@latte` model body can
also be sent through plain `DynamicPPL.@model` for Turing handoff with the
same syntax.
"""
macro random(ex)
    return esc(ex)
end

macro fixed(ex)
    return esc(ex)
end

# ─── Side-channel storage ────────────────────────────────────────────────────
const _LATTE_METADATA = IdDict{Any, NamedTuple}()
const _LATTE_DPPL_CONSTRUCTORS = IdDict{Any, Any}()

"""
    Latte.dppl_model(latte_fun) -> DPPL model constructor

Return the underlying `DynamicPPL.@model`-built constructor for a function
defined with `@latte`. Useful for Turing handoff:

```julia
@latte function my_model(y, X)
    σ ~ Gamma(2, 1)
    β ~ MvNormal(zeros(size(X, 2)), 100*I)
    for i in eachindex(y)
        y[i] ~ Normal(dot(X[i, :], β), σ)
    end
end

# Latte path
lgm = my_model(y, X)
inla(lgm, y)

# Turing path (same definition):
turing_model = Latte.dppl_model(my_model)(y, X)
sample(turing_model, NUTS(), 1000)
```
"""
function dppl_model(f)
    haskey(_LATTE_DPPL_CONSTRUCTORS, f) || throw(
        ArgumentError(
            "$(f) was not defined with @latte; no underlying DPPL model registered"
        )
    )
    return _LATTE_DPPL_CONSTRUCTORS[f]
end

"""
    Latte.latte_analysis(latte_fun_or_lgm) -> NamedTuple

Static metadata captured at `@latte` macro time: hyperparameter names,
random-effect names, per-`~`-block records (lhs symbol, hp dependencies,
classification, dotted), and pre-computed observation groups.
"""
function latte_analysis(f)
    haskey(_LATTE_METADATA, f) || throw(
        ArgumentError(
            "$(f) was not defined with @latte; no analysis metadata registered"
        )
    )
    return _LATTE_METADATA[f]
end

# ─── Free-symbol extraction (scope-aware, alias-resolving) ───────────────────
const _RESERVED_SYMS = Set{Symbol}(
    [
        Symbol("nothing"), Symbol("missing"), Symbol("true"), Symbol("false"),
        :Inf, :NaN, :pi, :π,
    ]
)

# Default whitelist of constructor names recognised as random-effect priors
# (multivariate-Gaussian-shaped). Used only when no marker is present.
const _DEFAULT_RANDOM_FAMILIES = Set{Symbol}(
    [
        :MvNormal, :MvNormalCanon, :MvLogNormal,
        :IIDModel, :RWModel, :BesagModel, :MaternModel, :BYM2Model,
        :SeparableModel, :GMRF, :ConstrainedGMRF, :CombinedModel,
        :FixedEffectsModel, :ARModel,
    ]
)

function _free_symbols(expr, bound::Set{Symbol}, aliases::Dict{Symbol, Set{Symbol}})
    out = Set{Symbol}()
    _collect_free_syms!(out, expr, bound, aliases)
    return out
end

function _collect_free_syms!(out, e, bound, aliases)
    if e isa Symbol
        if e in bound
            haskey(aliases, e) && union!(out, aliases[e])
        elseif !(e in _RESERVED_SYMS)
            push!(out, e)
        end
        return out
    elseif e isa QuoteNode || e isa LineNumberNode
        return out
    elseif !(e isa Expr)
        return out
    end

    head = e.head
    if head === :quote
        return out
    elseif head === :. && length(e.args) >= 1
        # Field access OR broadcast: walk args[1] always; walk args[2] only if
        # it's an Expr (broadcast tuple) — for field access args[2] is a
        # QuoteNode and the Symbol-skip path keeps us safe.
        _collect_free_syms!(out, e.args[1], bound, aliases)
        length(e.args) >= 2 && _collect_free_syms!(out, e.args[2], bound, aliases)
    elseif head === :let
        bindings_node = e.args[1]
        body = e.args[2]
        local_bound = copy(bound)
        local_aliases = copy(aliases)
        binding_exprs = if bindings_node isa Expr && bindings_node.head === :block
            [a for a in bindings_node.args if !(a isa LineNumberNode)]
        else
            [bindings_node]
        end
        for bex in binding_exprs
            if bex isa Expr && bex.head === :(=) && length(bex.args) >= 2
                lhs, rhs = bex.args[1], bex.args[2]
                rhs_free = _free_symbols(rhs, local_bound, local_aliases)
                lhs_sym = _binding_target(lhs)
                if lhs_sym !== nothing
                    push!(local_bound, lhs_sym)
                    local_aliases[lhs_sym] = rhs_free
                else
                    union!(out, rhs_free)
                end
            elseif bex isa Symbol
                push!(local_bound, bex)
            else
                _collect_free_syms!(out, bex, local_bound, local_aliases)
            end
        end
        _collect_free_syms!(out, body, local_bound, local_aliases)
    elseif head === :for
        iter, body = e.args[1], e.args[2]
        local_bound = copy(bound)
        local_aliases = copy(aliases)
        if iter isa Expr && (iter.head === :(=) || iter.head === :in || iter.head === :∈)
            i_lhs, i_rhs = iter.args[1], iter.args[2]
            _collect_free_syms!(out, i_rhs, bound, aliases)
            i_sym = _binding_target(i_lhs)
            i_sym !== nothing && push!(local_bound, i_sym)
        else
            _collect_free_syms!(out, iter, bound, aliases)
        end
        _collect_free_syms!(out, body, local_bound, local_aliases)
    elseif head === :generator || head === :comprehension || head === :flatten
        local_bound = copy(bound)
        local_aliases = copy(aliases)
        body = e.args[1]
        for it in e.args[2:end]
            if it isa Expr && (it.head === :(=) || it.head === :in || it.head === :∈)
                i_lhs, i_rhs = it.args[1], it.args[2]
                _collect_free_syms!(out, i_rhs, local_bound, local_aliases)
                i_sym = _binding_target(i_lhs)
                i_sym !== nothing && push!(local_bound, i_sym)
            end
        end
        _collect_free_syms!(out, body, local_bound, local_aliases)
    elseif head === :function || head === :(->)
        sig, body = e.args[1], e.args[2]
        local_bound = copy(bound)
        for s in _arglist_syms(sig)
            push!(local_bound, s)
        end
        _collect_free_syms!(out, body, local_bound, copy(aliases))
    elseif head === :(=) && length(e.args) >= 2
        _collect_free_syms!(out, e.args[2], bound, aliases)
    elseif head === :block
        local_bound = copy(bound)
        local_aliases = copy(aliases)
        for a in e.args
            if a isa LineNumberNode
                continue
            elseif a isa Expr && a.head === :(=) && length(a.args) >= 2
                lhs, rhs = a.args[1], a.args[2]
                rhs_free = _free_symbols(rhs, local_bound, local_aliases)
                lhs_sym = _binding_target(lhs)
                if lhs_sym !== nothing
                    push!(local_bound, lhs_sym)
                    local_aliases[lhs_sym] = rhs_free
                else
                    union!(out, rhs_free)
                end
            else
                _collect_free_syms!(out, a, local_bound, local_aliases)
            end
        end
    elseif head === :kw
        length(e.args) >= 2 && _collect_free_syms!(out, e.args[2], bound, aliases)
    elseif head === :parameters
        for a in e.args
            _collect_free_syms!(out, a, bound, aliases)
        end
    else
        for a in e.args
            _collect_free_syms!(out, a, bound, aliases)
        end
    end
    return out
end

function _binding_target(lhs)
    lhs isa Symbol && return lhs
    lhs isa Expr || return nothing
    if lhs.head === :(::)
        return _binding_target(lhs.args[1])
    end
    return nothing
end

function _arglist_syms(sig)
    sig isa Symbol && return Symbol[sig]
    sig isa Expr || return Symbol[]
    if sig.head === :tuple
        return [s for a in sig.args for s in _arglist_syms(a)]
    elseif sig.head === :(::)
        return _arglist_syms(sig.args[1])
    elseif sig.head === :call
        return [s for a in sig.args[2:end] for s in _arglist_syms(a)]
    elseif sig.head === :parameters
        return [s for a in sig.args for s in _arglist_syms(a)]
    elseif sig.head === :kw
        return _arglist_syms(sig.args[1])
    end
    return Symbol[]
end

# ─── Top-level symbol of a `~` LHS (matches DPPL's get_top_level_symbol) ─────
_lhs_top_sym(s::Symbol) = s
function _lhs_top_sym(e::Expr)
    if e.head === :ref || e.head === :.
        return _lhs_top_sym(e.args[1])
    end
    return nothing
end
_lhs_top_sym(_) = nothing

# ─── Tilde-block walker ───────────────────────────────────────────────────────
"""
    _TildeBlock — internal record of one `~` site found during macro analysis.
"""
struct _TildeBlock
    lhs_sym::Symbol
    rhs_free::Set{Symbol}
    is_dotted::Bool
    family::Union{Symbol, Nothing}        # the RHS callee Symbol if recognisable
    marker::Symbol                         # :random, :fixed, or :auto
end

"""
Detect `(L ~ R)` or `(L .~ R)` in an Expr; return `(L, R, is_dotted)` or
`nothing`. Handles both `Expr(:call, :~, ...)` and `Expr(:.~, ...)` forms.
"""
function _detect_tilde_pair(e)
    e isa Expr || return nothing
    if e.head === :call && length(e.args) == 3 && e.args[1] === :~
        return (e.args[2], e.args[3], false)
    end
    if e.head === :.~ && length(e.args) == 2
        return (e.args[1], e.args[2], true)
    end
    if e.head === :call && length(e.args) == 3 && e.args[1] === :.~
        return (e.args[2], e.args[3], true)
    end
    return nothing
end

"""
Detect a `@random ~_expr` or `@fixed ~_expr` macrocall. Returns
`(:random | :fixed, inner_expr)` or `nothing`. Supports both bare `@random`
and `Latte.@random` forms.
"""
function _detect_marker_call(e)
    e isa Expr && e.head === :macrocall || return nothing
    name = e.args[1]
    marker = if name === Symbol("@random") ||
            (
            name isa Expr && name.head === :. && name.args[end] isa QuoteNode &&
                name.args[end].value === Symbol("@random")
        )
        :random
    elseif name === Symbol("@fixed") ||
            (
            name isa Expr && name.head === :. && name.args[end] isa QuoteNode &&
                name.args[end].value === Symbol("@fixed")
        )
        :fixed
    else
        return nothing
    end
    # Skip LineNumberNode arg, take the wrapped expression.
    inner_args = filter(a -> !(a isa LineNumberNode), e.args[2:end])
    isempty(inner_args) && return nothing
    return marker, inner_args[end]
end

# Top-level callee symbol of a `~` RHS expression (e.g. `Normal(0, 1)` →
# `:Normal`, `MvNormal(zeros(p), I)` → `:MvNormal`, `IIDModel(n)(τ=τ)` →
# `:IIDModel`). Returns `nothing` if the RHS isn't a recognisable call.
function _rhs_family(rhs)
    rhs isa Expr || return nothing
    if rhs.head === :call && length(rhs.args) >= 1
        return _callee_top_sym(rhs.args[1])
    end
    return nothing
end

function _callee_top_sym(e)
    e isa Symbol && return e
    if e isa Expr
        if e.head === :. && length(e.args) >= 2
            tail = e.args[end]
            tail isa QuoteNode && tail.value isa Symbol && return tail.value
            tail isa Symbol && return tail
        elseif e.head === :curly && !isempty(e.args)
            return _callee_top_sym(e.args[1])
        elseif e.head === :call && !isempty(e.args)
            # e.g. `IIDModel(n)(τ=τ)` — the outer call's callee is itself a call.
            return _callee_top_sym(e.args[1])
        end
    end
    return nothing
end

function _walk_tilde_blocks(body, posargs)
    out = _TildeBlock[]
    _walk_tilde!(out, body, Set{Symbol}(posargs), Dict{Symbol, Set{Symbol}}(), :auto)
    return out
end

function _walk_tilde!(out, e, bound, aliases, current_marker)
    e isa Expr || return

    # Marker-prefixed tilde: walk the inner expression with marker applied.
    marker_pair = _detect_marker_call(e)
    if marker_pair !== nothing
        marker, inner = marker_pair
        _walk_tilde!(out, inner, bound, aliases, marker)
        return
    end

    pair = _detect_tilde_pair(e)
    if pair !== nothing
        lhs, rhs, dotted = pair
        lhs_sym = _lhs_top_sym(lhs)
        if lhs_sym !== nothing
            free = _free_symbols(rhs, bound, aliases)
            family = _rhs_family(rhs)
            push!(out, _TildeBlock(lhs_sym, free, dotted, family, current_marker))
        end
        return
    end

    head = e.head
    if head === :let
        bindings_node = e.args[1]
        body = e.args[2]
        local_bound = copy(bound)
        local_aliases = copy(aliases)
        binding_exprs = if bindings_node isa Expr && bindings_node.head === :block
            [a for a in bindings_node.args if !(a isa LineNumberNode)]
        else
            [bindings_node]
        end
        for bex in binding_exprs
            if bex isa Expr && bex.head === :(=) && length(bex.args) >= 2
                lhs, rhs = bex.args[1], bex.args[2]
                rhs_free = _free_symbols(rhs, local_bound, local_aliases)
                lhs_sym = _binding_target(lhs)
                if lhs_sym !== nothing
                    push!(local_bound, lhs_sym)
                    local_aliases[lhs_sym] = rhs_free
                end
            elseif bex isa Symbol
                push!(local_bound, bex)
            end
        end
        _walk_tilde!(out, body, local_bound, local_aliases, current_marker)
    elseif head === :for
        iter, body = e.args[1], e.args[2]
        local_bound = copy(bound)
        if iter isa Expr && (iter.head === :(=) || iter.head === :in || iter.head === :∈)
            i_sym = _binding_target(iter.args[1])
            i_sym !== nothing && push!(local_bound, i_sym)
        end
        _walk_tilde!(out, body, local_bound, copy(aliases), current_marker)
    elseif head === :function || head === :(->)
        # Don't descend into nested function bodies — `~` inside a local
        # function isn't an observation/sample of the outer model.
    elseif head === :block
        local_bound = copy(bound)
        local_aliases = copy(aliases)
        for a in e.args
            if a isa LineNumberNode
                continue
            elseif a isa Expr && a.head === :(=) && length(a.args) >= 2
                lhs, rhs = a.args[1], a.args[2]
                rhs_free = _free_symbols(rhs, local_bound, local_aliases)
                lhs_sym = _binding_target(lhs)
                if lhs_sym !== nothing
                    push!(local_bound, lhs_sym)
                    local_aliases[lhs_sym] = rhs_free
                end
                _walk_tilde!(out, rhs, local_bound, local_aliases, current_marker)
            else
                _walk_tilde!(out, a, local_bound, local_aliases, current_marker)
            end
        end
    else
        for a in e.args
            _walk_tilde!(out, a, bound, aliases, current_marker)
        end
    end
    return
end

# ─── Body transformation: strip markers, lower `.~` to `product_distribution` ─
function _transform_body(body)
    return _strip_markers(_lower_dottilde(body))
end

function _strip_markers(e)
    e isa Expr || return e
    pair = _detect_marker_call(e)
    if pair !== nothing
        _, inner = pair
        return _strip_markers(inner)
    end
    return Expr(e.head, map(_strip_markers, e.args)...)
end

function _lower_dottilde(e)
    e isa Expr || return e
    if e.head === :.~ && length(e.args) == 2
        lhs = _lower_dottilde(e.args[1])
        rhs = _lower_dottilde(e.args[2])
        return Expr(
            :call, :~, lhs,
            Expr(:call, :(Distributions.product_distribution), rhs)
        )
    end
    if e.head === :call && length(e.args) == 3 && e.args[1] === :.~
        lhs = _lower_dottilde(e.args[2])
        rhs = _lower_dottilde(e.args[3])
        return Expr(
            :call, :~, lhs,
            Expr(:call, :(Distributions.product_distribution), rhs)
        )
    end
    return Expr(e.head, map(_lower_dottilde, e.args)...)
end

# ─── Classification of one block ──────────────────────────────────────────────
"""
    _classify_block(blk, posargs) -> :observation | :random | :fixed

Apply the classification rule: positional-arg LHS = observation; markers
override; otherwise default to `:random` for known random-effect families
and `:fixed` for everything else.
"""
function _classify_block(blk::_TildeBlock, posargs::Tuple)
    blk.lhs_sym in posargs && return :observation
    blk.marker === :random && return :random
    blk.marker === :fixed && return :fixed
    blk.family !== nothing && blk.family in _DEFAULT_RANDOM_FAMILIES && return :random
    return :fixed
end

# ─── The macro itself ─────────────────────────────────────────────────────────
"""
    @latte function name(args...; kwargs...)
        ...
    end

Define a model whose call form returns a `LatentGaussianModel` directly.
Auto-detects hyperparameters, latent random effects, and observation
groups from the body via static AST analysis. Use `@random` and `@fixed`
markers inside the body to override the default classification.

The same body is also forwarded to `DynamicPPL.@model` and made available
via `Latte.dppl_model(name)` for Turing handoff.
"""
macro latte(modeldef)
    if !(modeldef isa Expr) || modeldef.head !== :function
        error("@latte requires a function definition")
    end
    sig = modeldef.args[1]
    body = modeldef.args[2]
    fname, posargs = _split_signature(sig)

    blocks = _walk_tilde_blocks(body, posargs)
    posargs_t = Tuple(posargs)

    # Classify and record per-block info.
    obs_records = _serialise_records(b for b in blocks if _classify_block(b, posargs_t) === :observation)
    random_records = _serialise_records(b for b in blocks if _classify_block(b, posargs_t) === :random)
    fixed_records = _serialise_records(b for b in blocks if _classify_block(b, posargs_t) === :fixed)

    # Build the body to forward to @model: strip markers, lower dot-tilde.
    body_for_dppl = _transform_body(body)
    inner_name = Symbol("__latte_dppl_", fname)
    inner_def = Expr(:function, _replace_fname(sig, inner_name), body_for_dppl)
    expanded_inner = macroexpand(__module__, :(DynamicPPL.@model $inner_def))

    # Quote serialised records as Exprs spliceable into the returned quote.
    obs_q = _quote_records(obs_records)
    rand_q = _quote_records(random_records)
    fixed_q = _quote_records(fixed_records)
    posargs_q = QuoteNode(posargs_t)

    return quote
        $(esc(expanded_inner))
        function $(esc(fname))(args...; kwargs...)
            dppl = $(esc(inner_name))(args...; kwargs...)
            return $(@__MODULE__)._build_lgm_from_latte(
                dppl, $rand_q, $fixed_q, $obs_q, $posargs_q,
            )
        end
        $(@__MODULE__)._LATTE_DPPL_CONSTRUCTORS[$(esc(fname))] = $(esc(inner_name))
        $(@__MODULE__)._LATTE_METADATA[$(esc(fname))] = (
            random_records = $rand_q,
            fixed_records = $fixed_q,
            obs_records = $obs_q,
            posargs = $posargs_q,
            random_syms = Tuple(unique(r[1] for r in $rand_q)),
            fixed_syms = Tuple(unique(r[1] for r in $fixed_q)),
        )
        $(esc(fname))
    end
end

function _split_signature(sig)
    sig.head === :call || error("@latte: malformed signature $(sig)")
    fname = sig.args[1]
    fname isa Symbol || error("@latte: expected plain function name")
    posarg_syms = Symbol[]
    for a in sig.args[2:end]
        a isa Expr && a.head === :parameters && continue
        s = _binding_target(a)
        s === nothing || push!(posarg_syms, s)
    end
    return fname, posarg_syms
end

function _replace_fname(sig::Expr, new_name::Symbol)
    args = copy(sig.args)
    args[1] = new_name
    return Expr(sig.head, args...)
end

# Records: (lhs_sym::Symbol, free::Vector{Symbol}, dotted::Bool, family::Union{Symbol,Nothing}, marker::Symbol).
_serialise_records(it) = collect(
    (b.lhs_sym, sort(collect(b.rhs_free)), b.is_dotted, b.family, b.marker)
        for b in it
)

function _quote_records(records)
    items = Expr(:vect)
    for (lhs, free_vec, dotted, family, marker) in records
        free_e = Expr(:vect, [QuoteNode(s) for s in free_vec]...)
        family_e = family === nothing ? :nothing : QuoteNode(family)
        push!(
            items.args,
            Expr(:tuple, QuoteNode(lhs), free_e, dotted, family_e, QuoteNode(marker)),
        )
    end
    return items
end

# ─── LGM construction from records ────────────────────────────────────────────
function _build_lgm_from_latte(
        dppl_model, random_records, fixed_records, obs_records, posargs::Tuple,
    )
    random_syms = Tuple(unique(r[1] for r in random_records))
    hp_names = Tuple(unique(r[1] for r in fixed_records))

    # Per-obs hp deps. Detect ambiguous repeated-LHS-with-different-deps.
    obs_dep = Dict{Symbol, Set{Symbol}}()
    obs_seen = Dict{Symbol, Set{Symbol}}()
    obs_family = Dict{Symbol, Union{Symbol, Nothing}}()
    for (lhs, free_vec, _dotted, family, _marker) in obs_records
        deps = intersect(Set(Symbol.(free_vec)), Set(hp_names))
        if haskey(obs_seen, lhs)
            if obs_seen[lhs] != deps
                throw(
                    ArgumentError(
                        "@latte: observed symbol :$(lhs) appears in multiple ~ blocks " *
                            "with different hp dependencies $((collect(obs_seen[lhs]), collect(deps))); " *
                            "declare obs_groups manually or split the data",
                    ),
                )
            end
        else
            obs_seen[lhs] = deps
            obs_dep[lhs] = deps
            obs_family[lhs] = family
        end
    end

    # Group obs syms by (family, deps). Same as Codex's approach: a Normal
    # block and a Poisson block with the same hp set still split, because the
    # downstream component needs to instantiate one likelihood family per
    # component.
    groups_by_key = Dict{Tuple{Union{Symbol, Nothing}, Set{Symbol}}, Vector{Symbol}}()
    for (sym, deps) in obs_dep
        key = (obs_family[sym], deps)
        push!(get!(groups_by_key, key, Symbol[]), sym)
    end

    # Single-group case: defer to the legacy single-AD path so inference
    # methods can use the existing IFT dispatch on `AutoDiffLikelihood`-with-
    # Dual-hp. The composite path lacks that dispatch (followup #64) and
    # would trip nested-AD tag stacking through the outer hp-gradient pass.
    # Multi-group case: composite is required for distinct-hp routing, even
    # at the cost of (currently) blocking gradient-based inference on those
    # LGMs.
    obs_groups = if isempty(obs_dep) || length(groups_by_key) <= 1
        nothing
    else
        ordered = sort(
            collect(groups_by_key);
            by = ((k, _v),) -> (string(k[1]), sort(collect(k[2]))),
        )
        [
            Symbol("group_$(i)") => Tuple(sort(syms))
                for (i, (_k, syms)) in enumerate(ordered)
        ]
    end

    # Single-group case: prefer fast-path detection (Mooncake on the AD
    # likelihood doesn't handle Dual hp gradients yet, so the fast-path is
    # what makes inla() actually run). Force AD when:
    # - static analysis detected obs-RHS hp aliasing or shadowing (the
    #   runtime fast-path probe would see the wrong kwarg name); or
    # - any random_sym is a scalar `Normal` (TMB-style scalar random
    #   effect). The fast-path probe seeds it as a 1-vector via
    #   `InitFromParams`, which DPPL doesn't unwrap for a scalar variable.
    #   The AD path's `LogDensityFunction` handles flat-x correctly.
    needs_ad_fallback = obs_groups === nothing && (
        _needs_ad_fallback(obs_records, hp_names) ||
            _has_scalar_random(random_records)
    )
    return latte_from_dppl(
        dppl_model;
        random = random_syms,
        obs_groups = obs_groups,
        force_ad_obs_model = needs_ad_fallback,
    )
end

# Heuristic: if the static obs RHS deps contain a hp under an *aliased* or
# *shadowed* binding, runtime fast-path detection will see the wrong kwarg
# name and fail validation. Detect this by comparing the static deps with
# what fast-path would expect (a literal `:σ` for Normal, etc.). For now we
# approximate: if the obs free symbols intersected with hp_names doesn't
# include the fast-path expected kwargs (`:σ` for typical Normal), assume
# we need the AD fallback. Conservative, but only triggers when the user
# uses unusual hp aliasing/shadowing.
function _has_scalar_random(random_records)
    # Heuristic: known scalar prior names that pass through the fast-path
    # probe's NamedTuple seeding incorrectly. `Normal`, `Cauchy`, etc. are
    # `UnivariateDistribution`s; the storage gets a 1-vector via `dims=1`
    # which DPPL doesn't unwrap. Force the AD path to side-step.
    scalar_priors = Set(
        [
            :Normal, :Cauchy, :Laplace, :TDist, :Uniform, :LogNormal,
            :Logistic, :Gumbel, :Frechet,
        ]
    )
    for (_lhs, _free, _dotted, family, _marker) in random_records
        family === nothing && continue
        family in scalar_priors && return true
    end
    return false
end

function _needs_ad_fallback(obs_records, hp_names::Tuple)
    isempty(obs_records) && return false
    hp_set = Set(hp_names)
    for (_lhs, free_vec, _dotted, family, _marker) in obs_records
        deps = intersect(Set(Symbol.(free_vec)), hp_set)
        # If the obs is Normal-like and `:σ` isn't in deps but other hps are,
        # this is most likely an alias / shadow case.
        if family === :Normal && !isempty(deps) && !(:σ in deps)
            return true
        end
        # If the obs is Normal-like and deps is empty but a hp exists named
        # something else, the runtime probe might still trip.
        if family === :Normal && isempty(deps) && !isempty(hp_names)
            return true
        end
    end
    return false
end
