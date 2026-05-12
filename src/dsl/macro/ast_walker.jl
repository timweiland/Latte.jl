# AST helpers used by the `@latte` macro and the prelude-lift codegen.
#
# Self-contained Julia AST manipulation:
# - `_free_symbols` / `_collect_free_syms!`: scope-aware free-variable
#   extraction, used to compute hp dependencies of obs RHS expressions and
#   to compute prelude→post-random capture sets.
# - `_TildeBlock`, `_detect_tilde_pair`, `_detect_marker_call`,
#   `_walk_tilde_blocks`: classify and collect `~` / `@random` / `@fixed`
#   sites in a model body.
# - `_transform_body`, `_strip_markers`, `_lower_dottilde`: rewrite the
#   user body into something DPPL's `@model` will accept (strip Latte's
#   markers, lower `.~` to `product_distribution(...)`).
# - `_classify_block`: decide whether a `~` site is an observation, a
#   random effect, or a hyperparameter prior given the classification rules.
#
# These helpers contain no DPPL or GMRFs dependencies — they only see Exprs.

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

# Prelude-lift analyzer + codegen lives in `prelude_lift.jl`, included
# right after this file in `Latte.jl`. The `@latte` macro below calls
# `_analyze_lift_eligibility` / `_generate_lift_callables` from that file
# at macro-expansion time (after both files have been loaded).
