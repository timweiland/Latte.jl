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
import Distributions
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

# ─── Prelude-lift eligibility analyzer ────────────────────────────────────────
# Identifies whether a `@latte` body splits cleanly at the first top-level
# `@random` site so the hp-dependent prelude (kernel construction, GP build,
# etc.) can be computed once per `obs_lik` materialisation and reused across
# AD sweeps of the observation likelihood. See task #80.
"""
    LiftPlan

Result of `_analyze_lift_eligibility` when the body is liftable. Captures
the AST split and the variables that flow from prelude into the
post-random body.

Fields:
- `prelude_stmts`: top-level statements before the first `@random` site.
- `post_stmts`: top-level statements from the first `@random` site onward.
- `capture`: prelude-assigned symbols read by the post-random body.
- `hp_syms`: hp-prior LHS symbols collected from prelude `~` sites
  (these `~` lines get dropped during codegen — hp values arrive via
  the lifted body's `hp_nt` kwarg).
- `random_syms`: top-level `@random` LHS symbols in the post-random region.
"""
struct LiftPlan
    prelude_stmts::Vector{Any}
    post_stmts::Vector{Any}
    capture::Vector{Symbol}
    hp_syms::Vector{Symbol}
    random_syms::Vector{Symbol}
end

# Flatten only top-level :block expressions; do not descend into :let, :for,
# :if, etc. Strips LineNumberNodes.
function _top_level_stmts(body)
    body isa Expr && body.head === :block || return Any[body]
    out = Any[]
    for a in body.args
        a isa LineNumberNode && continue
        if a isa Expr && a.head === :block
            append!(out, _top_level_stmts(a))
        else
            push!(out, a)
        end
    end
    return out
end

# Strip a `@random` / `@fixed` macrocall wrapper and report the marker.
# Returns `(marker, inner_expr)` where marker ∈ (:random, :fixed, :auto).
function _peel_marker(e)
    pair = _detect_marker_call(e)
    pair === nothing && return (:auto, e)
    return pair
end

# Classify a top-level statement.
#
# Returns `(kind, info)` where `kind ∈ (:fixed, :random, :observation,
# :non_tilde)`. For tilde kinds, `info` is a `_TildeBlock`-like NamedTuple
# `(lhs_sym, marker)`. For `:non_tilde`, `info` is `nothing`.
function _classify_top_stmt(stmt, posargs::Tuple)
    marker, inner = _peel_marker(stmt)
    pair = _detect_tilde_pair(inner)
    if pair === nothing
        return (:non_tilde, nothing)
    end
    lhs, rhs, _dotted = pair
    lhs_sym = _lhs_top_sym(lhs)
    lhs_sym === nothing && return (:non_tilde, nothing)
    family = _rhs_family(rhs)
    blk = _TildeBlock(lhs_sym, Set{Symbol}(), false, family, marker)
    kind = _classify_block(blk, posargs)
    return (kind, (lhs_sym = lhs_sym, marker = marker))
end

# Walk `e` recursively and return `true` if any tilde or marker-wrapped
# tilde site is found anywhere inside.
function _contains_any_tilde(e)
    e isa Expr || return false
    _detect_tilde_pair(e) !== nothing && return true
    _detect_marker_call(e) !== nothing && return true
    return any(_contains_any_tilde, e.args)
end

# Walk `e` and return `true` if any tilde inside would classify as `:random`
# or `:fixed` (i.e. a non-observation tilde). Used to reject "hidden"
# random/hp sites inside post-random non-tilde statements.
function _contains_non_obs_tilde(e, posargs::Tuple)
    e isa Expr || return false
    pair = _detect_marker_call(e)
    if pair !== nothing
        marker, inner = pair
        inner_pair = _detect_tilde_pair(inner)
        if inner_pair !== nothing
            lhs_sym = _lhs_top_sym(inner_pair[1])
            lhs_sym !== nothing || @goto recurse
            family = _rhs_family(inner_pair[2])
            blk = _TildeBlock(lhs_sym, Set{Symbol}(), false, family, marker)
            _classify_block(blk, posargs) !== :observation && return true
        end
    end
    pair2 = _detect_tilde_pair(e)
    if pair2 !== nothing
        lhs_sym = _lhs_top_sym(pair2[1])
        if lhs_sym !== nothing
            family = _rhs_family(pair2[2])
            blk = _TildeBlock(lhs_sym, Set{Symbol}(), false, family, :auto)
            _classify_block(blk, posargs) !== :observation && return true
        end
    end
    @label recurse
    return any(arg -> _contains_non_obs_tilde(arg, posargs), e.args)
end

# Collect symbols assigned by top-level `=` statements in the prelude.
# Supports `x = ...`, `(a, b) = ...`, `x::T = ...`. Ignores
# `A[i] = ...` and `obj.field = ...` (not new bindings).
function _collect_assigned_top(stmts)
    out = Set{Symbol}()
    for s in stmts
        s isa Expr || continue
        s.head === :(=) || continue
        _collect_assignment_targets!(out, s.args[1])
    end
    return out
end

function _collect_assignment_targets!(out, lhs)
    lhs isa Symbol && (push!(out, lhs); return)
    lhs isa Expr || return
    if lhs.head === :(::)
        _collect_assignment_targets!(out, lhs.args[1])
    elseif lhs.head === :tuple
        for a in lhs.args
            _collect_assignment_targets!(out, a)
        end
    end
    return
end

"""
    _analyze_lift_eligibility(body, posargs) -> Union{Nothing, LiftPlan}

Decide whether `body` is shaped well enough for the prelude-lift codegen
path. Returns a `LiftPlan` on success; returns `nothing` (with no error)
to signal "fall back to the standard DPPL obs path".

Eligibility rules (all required):

1. There is at least one top-level `@random`/random-marker `~` site.
2. The first top-level random site is not hidden inside `for`/`let`/`if`/
   nested function — it must appear directly in the top-level statement
   sequence.
3. The prelude (statements before the first top-level random site) only
   contains: top-level hp `~` priors, or ordinary non-tilde statements
   that themselves contain no tilde at any depth. Observations and
   nested random sites are rejected.
4. The post-random region must not contain any hp `~` priors at any
   depth, nor any "hidden" random sites inside non-tilde top-level
   statements (random `~` lurking inside a `for`).
5. The body's outermost expression is `Expr(:block, ...)` — a single
   `let ... end` wrapping the whole body falls back.
"""
function _analyze_lift_eligibility(body, posargs::Tuple)
    # Rule 5: reject single-let wrap (or anything that isn't a block).
    body isa Expr || return nothing
    body.head === :block || return nothing

    top_stmts = _top_level_stmts(body)
    isempty(top_stmts) && return nothing

    classifications = [_classify_top_stmt(s, posargs) for s in top_stmts]

    first_random_idx = findfirst(c -> c[1] === :random, classifications)
    first_random_idx === nothing && return nothing  # rule 1

    # Rules 3 + 2 (the "first random not hidden" part is enforced because
    # we only see TOP-LEVEL random sites — anything inside a for/let/if
    # would appear as :non_tilde here, and the body would fall through to
    # the "no top-level random" rejection above OR be caught by rule 3
    # when the prelude `:non_tilde` statement contains a hidden random).
    for i in 1:(first_random_idx - 1)
        kind, _info = classifications[i]
        if kind === :random || kind === :observation
            return nothing  # rule 3 (random already handled; obs disallowed)
        elseif kind === :non_tilde
            _contains_any_tilde(top_stmts[i]) && return nothing  # rule 3
        end
        # :fixed at top-level: OK.
    end

    # Rule 4: nothing classified `:fixed` post-random; no hidden random in
    # post-random non-tilde stmts.
    for i in first_random_idx:length(classifications)
        kind, _info = classifications[i]
        if kind === :fixed
            return nothing
        elseif kind === :non_tilde
            _contains_non_obs_tilde(top_stmts[i], posargs) && return nothing
        end
    end

    # ─── Collect plan details ─────────────────────────────────────────────
    prelude_stmts = top_stmts[1:(first_random_idx - 1)]
    post_stmts = top_stmts[first_random_idx:end]

    hp_syms = Symbol[]
    for (i, (kind, info)) in enumerate(classifications[1:(first_random_idx - 1)])
        kind === :fixed && info !== nothing && push!(hp_syms, info.lhs_sym)
    end

    random_syms = Symbol[]
    for (kind, info) in classifications[first_random_idx:end]
        kind === :random && info !== nothing && push!(random_syms, info.lhs_sym)
    end

    # Tight capture: assigned-in-prelude ∩ free-in-post.
    # `bound_init` deliberately excludes `assigned_in_prelude` so post-body
    # references to prelude vars survive into `free_in_post`.
    assigned = _collect_assigned_top(prelude_stmts)
    bound_init = Set{Symbol}()
    for s in posargs
        push!(bound_init, s)
    end
    for s in random_syms
        push!(bound_init, s)
    end
    for s in hp_syms
        push!(bound_init, s)
    end
    free_in_post = _free_symbols(
        Expr(:block, post_stmts...), bound_init, Dict{Symbol, Set{Symbol}}(),
    )
    capture = sort(collect(intersect(free_in_post, assigned)))

    return LiftPlan(prelude_stmts, post_stmts, capture, hp_syms, random_syms)
end

# ─── Codegen: prelude + obs body + pointwise from a LiftPlan ──────────────────

# Rewrite a single prelude statement: drop hp `~` priors entirely; pass other
# statements through verbatim. Returns `nothing` to signal "remove this stmt".
function _lift_lower_prelude_stmt(stmt, hp_syms::Set{Symbol})
    pair = _detect_tilde_pair(stmt)
    if pair !== nothing
        lhs_sym = _lhs_top_sym(pair[1])
        lhs_sym in hp_syms && return nothing
    end
    return stmt
end

# Build a `(; a, b, c) = nt` destructure as an Expr. Returns `nothing` when
# the symbol list is empty (callers filter `nothing` out).
function _destructure_into_locals(nt_expr, syms::Vector{Symbol})
    isempty(syms) && return nothing
    return Expr(
        :(=),
        Expr(:tuple, Expr(:parameters, syms...)),
        nt_expr,
    )
end

# Build a NamedTuple constructor expr `(; a, b, c)` referencing the local
# symbols by name. Returns `:(NamedTuple())` for an empty list.
function _capture_namedtuple_expr(syms::Vector{Symbol})
    isempty(syms) && return :(NamedTuple())
    return Expr(:tuple, Expr(:parameters, syms...))
end

# Rewrite an expression in the post-random region:
# - `@random` / `@fixed` markers are unwrapped.
# - Random tilde sites (`@random x ~ g` or auto-classified random) are dropped.
# - Observation tilde sites are lowered into a guarded
#   `__logp += loglikelihood(rhs, lhs)` (or `push!(__contribs, ...)` for
#   the pointwise variant).
# - `:function`, `:->`, and `:quote` subtrees are NOT descended into.
function _lift_lower_post_expr(
        e, hp_syms::Set{Symbol}, random_syms::Set{Symbol}, posargs::Set{Symbol};
        pointwise::Bool,
    )
    e isa Expr || return e

    marker_pair = _detect_marker_call(e)
    if marker_pair !== nothing
        _marker, inner = marker_pair
        return _lift_lower_post_expr(
            inner, hp_syms, random_syms, posargs; pointwise = pointwise,
        )
    end

    tilde_pair = _detect_tilde_pair(e)
    if tilde_pair !== nothing
        return _lower_tilde_site(e, tilde_pair, random_syms, posargs; pointwise = pointwise)
    end

    head = e.head
    (head === :function || head === :(->) || head === :quote) && return e

    new_args = Any[]
    for a in e.args
        rewritten = _lift_lower_post_expr(
            a, hp_syms, random_syms, posargs; pointwise = pointwise,
        )
        rewritten === nothing && continue
        push!(new_args, rewritten)
    end
    return Expr(head, new_args...)
end

# Lower a single tilde site in the post-random region. Returns:
# - `nothing` if the site is a random `~` (already bound from `__flat_x`).
# - an `if … in __group_syms; <accumulate>; end` Expr if the site is an
#   observation `~` (`accumulate` is `__logp +=` or `push!(__contribs, …)`
#   depending on `pointwise`; dotted `~` wraps rhs in `product_distribution`).
# - the original expression unchanged for any other tilde (defensive — the
#   analyzer rejects hp `~` post-random, so this branch is unreachable in
#   practice).
function _lower_tilde_site(
        e, tilde_pair, random_syms::Set{Symbol}, posargs::Set{Symbol};
        pointwise::Bool,
    )
    lhs, rhs, dotted = tilde_pair
    lhs_sym = _lhs_top_sym(lhs)
    lhs_sym === nothing && return e
    lhs_sym in random_syms && return nothing
    lhs_sym in posargs || return e

    rhs_lowered = dotted ?
        Expr(:call, :(Distributions.product_distribution), rhs) : rhs
    ll_call = Expr(:call, :(Distributions.loglikelihood), rhs_lowered, lhs)
    body_expr = pointwise ?
        Expr(:call, :push!, :__contribs, ll_call) :
        Expr(:(+=), :__logp, ll_call)
    return Expr(
        :if,
        Expr(:call, :in, QuoteNode(lhs_sym), :__group_syms),
        Expr(:block, body_expr),
    )
end

# Build the random-binding statements. For each random sym, emit
#   rsym = __is_scalar.<rsym> ? __flat_x[first(__offsets.<rsym>)] :
#                                Vector(view(__flat_x, __offsets.<rsym>))
function _random_binding_stmts(random_syms::Vector{Symbol})
    return [
        :(
                $rsym = $(:__is_scalar).$rsym ?
                $(:__flat_x)[first($(:__offsets).$rsym)] :
                Vector(view($(:__flat_x), $(:__offsets).$rsym))
            ) for rsym in random_syms
    ]
end

# Build the hp NamedTuple construction expr from kwargs at canonical order.
function _hp_nt_construct_expr(hp_syms::Vector{Symbol})
    if isempty(hp_syms)
        return :(__hp_nt = NamedTuple())
    end
    hp_tuple = Tuple(hp_syms)
    return :(
        __hp_nt = NamedTuple{$hp_tuple}(
            Tuple(kwargs[k] for k in $hp_tuple),
        )
    )
end

# Promote-type expression for the loglik accumulator. Guards against empty
# hp lists (which would call `promote_type(eltype(x))` — fine, but the
# explicit branch matches Codex's guidance).
function _logp_eltype_expr(hp_syms::Vector{Symbol})
    isempty(hp_syms) && return :(eltype(__flat_x))
    return :(promote_type(eltype(__flat_x), map(typeof, values(__hp_nt))...))
end

"""
    _generate_lift_callables(plan::LiftPlan, fname::Symbol, posargs::Tuple)
        -> NamedTuple

Generate the prelude, obs body, and obs pointwise body function definitions
for a `@latte` model that passes eligibility analysis. The returned
NamedTuple has fields:

- `prelude_def::Expr` — `function __latte_prelude_<fname>(__args_nt, __hp_nt) … end`
- `obs_body_def::Expr` — `function __latte_obs_body_<fname>(__flat_x; y, kwargs...) … end`
- `pointwise_def::Expr` — sibling that returns a `Vector{T}` of per-site contributions
- `prelude_fname::Symbol`, `obs_body_fname::Symbol`, `pointwise_fname::Symbol`
  — the names emitted into the user module.

Callers `eval` the three function defs into the user module (typically via
the `@latte` macro's returned quote) and then refer to the bodies through
the symbol names.
"""
function _generate_lift_callables(plan::LiftPlan, fname::Symbol, posargs::Tuple)
    prelude_fname = Symbol("__latte_prelude_", fname)
    obs_body_fname = Symbol("__latte_obs_body_", fname)
    pointwise_fname = Symbol("__latte_obs_pointwise_", fname)

    hp_syms_set = Set(plan.hp_syms)
    random_syms_set = Set(plan.random_syms)
    posargs_set = Set(posargs)
    posargs_vec = collect(posargs)

    # ─── Prelude function ─────────────────────────────────────────────────
    prelude_destr_args = _destructure_into_locals(:__args_nt, posargs_vec)
    prelude_destr_hp = _destructure_into_locals(:__hp_nt, plan.hp_syms)
    prelude_lowered = Any[]
    for s in plan.prelude_stmts
        out = _lift_lower_prelude_stmt(s, hp_syms_set)
        out === nothing && continue
        push!(prelude_lowered, out)
    end
    capture_expr = _capture_namedtuple_expr(plan.capture)
    prelude_body_args = Any[]
    prelude_destr_args === nothing || push!(prelude_body_args, prelude_destr_args)
    prelude_destr_hp === nothing || push!(prelude_body_args, prelude_destr_hp)
    append!(prelude_body_args, prelude_lowered)
    push!(prelude_body_args, Expr(:return, capture_expr))
    prelude_def = Expr(
        :function,
        Expr(:call, prelude_fname, :__args_nt, :__hp_nt),
        Expr(:block, prelude_body_args...),
    )

    # ─── Obs body (scalar logp) ───────────────────────────────────────────
    obs_signature = Expr(
        :call, obs_body_fname,
        Expr(:parameters, :y, Expr(:..., :kwargs)),
        :__flat_x,
    )
    payload_unpack = Any[
        :(__args = y.args),
        :(__prelude_state = y.prelude_state),
        :(__group_syms = y.group_syms),
        :(__offsets = y.offsets),
        :(__is_scalar = y.is_scalar),
    ]
    hp_nt_construct = _hp_nt_construct_expr(plan.hp_syms)
    args_destr = _destructure_into_locals(:__args, posargs_vec)
    hp_destr = _destructure_into_locals(:__hp_nt, plan.hp_syms)
    capture_destr = _destructure_into_locals(:__prelude_state, plan.capture)
    random_bindings = _random_binding_stmts(plan.random_syms)
    logp_init = :(__logp = zero($(_logp_eltype_expr(plan.hp_syms))))
    obs_post_lowered = Any[]
    for s in plan.post_stmts
        out = _lift_lower_post_expr(
            s, hp_syms_set, random_syms_set, posargs_set; pointwise = false,
        )
        out === nothing && continue
        push!(obs_post_lowered, out)
    end
    obs_body_args = Any[]
    append!(obs_body_args, payload_unpack)
    push!(obs_body_args, hp_nt_construct)
    args_destr === nothing || push!(obs_body_args, args_destr)
    hp_destr === nothing || push!(obs_body_args, hp_destr)
    capture_destr === nothing || push!(obs_body_args, capture_destr)
    append!(obs_body_args, random_bindings)
    push!(obs_body_args, logp_init)
    append!(obs_body_args, obs_post_lowered)
    push!(obs_body_args, Expr(:return, :__logp))
    obs_body_def = Expr(:function, obs_signature, Expr(:block, obs_body_args...))

    # ─── Pointwise body (Vector{T} of per-site contributions) ─────────────
    pw_signature = Expr(
        :call, pointwise_fname,
        Expr(:parameters, :y, Expr(:..., :kwargs)),
        :__flat_x,
    )
    pw_post_lowered = Any[]
    for s in plan.post_stmts
        out = _lift_lower_post_expr(
            s, hp_syms_set, random_syms_set, posargs_set; pointwise = true,
        )
        out === nothing && continue
        push!(pw_post_lowered, out)
    end
    pw_contribs_init = :(__contribs = ($(_logp_eltype_expr(plan.hp_syms)))[])
    pw_body_args = Any[]
    append!(pw_body_args, payload_unpack)
    push!(pw_body_args, hp_nt_construct)
    args_destr === nothing || push!(pw_body_args, args_destr)
    hp_destr === nothing || push!(pw_body_args, hp_destr)
    capture_destr === nothing || push!(pw_body_args, capture_destr)
    append!(pw_body_args, random_bindings)
    push!(pw_body_args, pw_contribs_init)
    append!(pw_body_args, pw_post_lowered)
    push!(pw_body_args, Expr(:return, :__contribs))
    pointwise_def = Expr(:function, pw_signature, Expr(:block, pw_body_args...))

    return (
        prelude_def = prelude_def,
        obs_body_def = obs_body_def,
        pointwise_def = pointwise_def,
        prelude_fname = prelude_fname,
        obs_body_fname = obs_body_fname,
        pointwise_fname = pointwise_fname,
    )
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

    # Prelude-lift eligibility analysis + codegen.
    # Reject lift outright when the signature has kwargs — the macro currently
    # only routes positional args into the prelude's `__args_nt`, so kwargs
    # would silently disappear from generated code. Catch this here rather
    # than introducing latent runtime bugs.
    sig_has_kwargs = any(
        a -> a isa Expr && a.head === :parameters && !isempty(a.args),
        sig.args[2:end],
    )
    lift_plan = sig_has_kwargs ? nothing : _analyze_lift_eligibility(body, posargs_t)
    lift_emit = nothing
    lift_spec_expr = :(nothing)
    lift_meta_expr = :(nothing)
    if lift_plan !== nothing
        defs = _generate_lift_callables(lift_plan, fname, posargs_t)
        lift_emit = quote
            $(esc(defs.prelude_def))
            $(esc(defs.obs_body_def))
            $(esc(defs.pointwise_def))
        end
        lift_spec_expr = Expr(
            :tuple,
            Expr(
                :parameters,
                Expr(:kw, :prelude_fn, esc(defs.prelude_fname)),
                Expr(:kw, :obs_body_fn, esc(defs.obs_body_fname)),
                Expr(:kw, :pointwise_fn, esc(defs.pointwise_fname)),
                Expr(:kw, :capture, QuoteNode(lift_plan.capture)),
                Expr(:kw, :hp_syms, QuoteNode(lift_plan.hp_syms)),
                Expr(:kw, :random_syms, QuoteNode(lift_plan.random_syms)),
            ),
        )
        # Metadata stub for introspection (function names only — full
        # closures live in the user module).
        lift_meta_expr = Expr(
            :tuple,
            Expr(
                :parameters,
                Expr(:kw, :prelude_fname, QuoteNode(defs.prelude_fname)),
                Expr(:kw, :obs_body_fname, QuoteNode(defs.obs_body_fname)),
                Expr(:kw, :pointwise_fname, QuoteNode(defs.pointwise_fname)),
                Expr(:kw, :capture, QuoteNode(lift_plan.capture)),
                Expr(:kw, :hp_syms, QuoteNode(lift_plan.hp_syms)),
                Expr(:kw, :random_syms, QuoteNode(lift_plan.random_syms)),
            ),
        )
    end

    # Quote serialised records as Exprs spliceable into the returned quote.
    obs_q = _quote_records(obs_records)
    rand_q = _quote_records(random_records)
    fixed_q = _quote_records(fixed_records)
    posargs_q = QuoteNode(posargs_t)

    return quote
        $(esc(expanded_inner))
        $(lift_emit)
        function $(esc(fname))(args...; kwargs...)
            dppl = $(esc(inner_name))(args...; kwargs...)
            return $(@__MODULE__)._build_lgm_from_latte(
                dppl, $rand_q, $fixed_q, $obs_q, $posargs_q;
                lift_spec = $lift_spec_expr,
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
            lift_meta = $lift_meta_expr,
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
        dppl_model, random_records, fixed_records, obs_records, posargs::Tuple;
        lift_spec = nothing,
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
        lift_spec = lift_spec,
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
