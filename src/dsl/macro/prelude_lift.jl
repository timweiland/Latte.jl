# Prelude-lift for `@latte` AD obs likelihoods.
#
# When an `@latte` body has a clean split at the first top-level `@random`
# site, the hp-dependent prelude (kernel construction, GP build, Vecchia
# factorisation, etc.) can be computed once per `obs_lik` materialisation
# and reused across every AD sweep through the observation likelihood
# instead of being rebuilt inside each loglik call by DPPL. The macro
# emits two extra top-level functions (`__latte_prelude_*` and
# `__latte_obs_body_*`, plus a pointwise sibling) and the lifted-obs
# wrappers in `obs_model.jl` / `obs_groups.jl` consume them via the
# GMRFs `y` payload slot.
#
# This file is purely the AST analyser + codegen. The macro-side glue,
# emission, and per-component routing live in `latte_macro.jl` /
# `adapter.jl` / `obs_groups.jl` / `obs_model.jl`.
#
# Relies on AST helpers from `latte_macro.jl`: `_TildeBlock`,
# `_detect_tilde_pair`, `_detect_marker_call`, `_lhs_top_sym`,
# `_rhs_family`, `_classify_block`, `_free_symbols`. Include order in
# `Latte.jl` MUST place `latte_macro.jl` before this file.

# ─── Prelude-lift eligibility analyzer ────────────────────────────────────────
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
