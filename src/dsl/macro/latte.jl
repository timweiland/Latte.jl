# The `@latte` macro itself + LGM construction from macro-time records.
#
# At macro-expansion time the body is walked, classified, optionally
# analysed for prelude-lift eligibility, then forwarded to DPPL's `@model`
# constructor (preserving the unmodified body for Turing handoff via
# `Latte.dppl_model(name)`). The runtime call form returns a
# `LatentGaussianModel` built by `_build_lgm_from_latte`, which routes the
# obs construction to one of: fast-path, lifted-AD (single or composite),
# or legacy DPPL-AD (single or composite).

using DynamicPPL: @model
import DynamicPPL
import Distributions


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
