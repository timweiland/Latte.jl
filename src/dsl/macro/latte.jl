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
import LinearAlgebra


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

    # Concrete-`LatentModel` recognition (macro-pure). MVP scope: recognize
    # iff there is exactly one random block whose RHS matches a recognized
    # concrete LatentModel `Family(args...)(; k = hp, …)`, the route draws
    # only from hyperparameters, and the constructor depends only on
    # positional args / literals / globals (not on hyperparameters or random
    # effects). Otherwise fall back to the DAG / sparse-AD extraction path.
    recognition_expr = _recognition_expr(blocks, posargs_t)

    # Structured (factor-graph) support: for indexed-latent `~` sites (a non-Gaussian state-space
    # prior), emit a prior builder, an optional observation builder, and a layout-builder so the
    # nonlinear path can use a `StructuredLatentPrior` / `StructuredObservationModel` instead of
    # opaque AD closures. Purely a performance refinement — `build_latent_model` / the obs guard
    # verify them against the monolithic versions and fall back on any mismatch.
    latent_syms = Tuple(
        unique(
            b.lhs_sym for b in blocks
                if _classify_block(b, posargs_t) === :random && b.indexed && !b.is_dotted
        ),
    )
    obs_syms = Tuple(unique(b.lhs_sym for b in blocks if _classify_block(b, posargs_t) === :observation))
    hp_syms = Tuple(unique(r[1] for r in fixed_records))
    struct_support = _emit_structured_support(fname, body, posargs_t, hp_syms, latent_syms, obs_syms)

    # Build the body to forward to @model: strip markers, lower dot-tilde. When
    # a latent was recognized, swap its `~` RHS for a cheap probing stand-in so
    # the model-body probes (hp / dims / obs) don't materialise the real prior
    # (which a workspace-only backend refuses); the real prior comes from the
    # recognition spec.
    body_for_dppl = _transform_body(_maybe_probe_rewrite(body, blocks, posargs_t, recognition_expr))
    inner_name = Symbol("__latte_dppl_", fname)
    inner_def = Expr(:function, _replace_fname(sig, inner_name), body_for_dppl)
    # Splice the `DynamicPPL` module object (resolved in *this* package's
    # scope, where it's always imported) rather than the bareword path, so
    # `@latte` works in user modules that haven't `using DynamicPPL`.
    expanded_inner = macroexpand(__module__, :($(DynamicPPL).@model $inner_def))

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

    # Emit the structured builder defs (if any) and the spliceable `structured` spec.
    if struct_support === nothing
        struct_emit = nothing
        structured_expr = :(nothing)
    else
        obs_emit = struct_support.obs_def === nothing ? nothing : esc(struct_support.obs_def)
        struct_emit = quote
            $(esc(struct_support.builder_def))
            $(obs_emit)
            $(esc(struct_support.layout_def))
        end
        obs_builder_expr = struct_support.obs_name === nothing ? :nothing : esc(struct_support.obs_name)
        structured_expr = Expr(
            :tuple,
            Expr(
                :parameters,
                Expr(:kw, :builder, esc(struct_support.builder_name)),
                Expr(:kw, :obs_builder, obs_builder_expr),
                Expr(:kw, :layout_builder, esc(struct_support.layout_name)),
                Expr(:kw, :obs_syms, QuoteNode(obs_syms)),
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
        $(struct_emit)
        function $(esc(fname))(args...; likelihood_hessian_pattern = :auto, augment = false, kwargs...)
            dppl = $(esc(inner_name))(args...; kwargs...)
            return $(@__MODULE__)._build_lgm_from_latte(
                dppl, $rand_q, $fixed_q, $obs_q, $posargs_q, args;
                lift_spec = $lift_spec_expr,
                recognition = $recognition_expr,
                structured = $structured_expr,
                likelihood_hessian_pattern = likelihood_hessian_pattern,
                augment = augment,
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

# Macro-time: build the spliceable `recognition` expression for the runtime
# `_build_lgm_from_latte` call, or `:(nothing)` when the body isn't
# recognizable (→ DAG / sparse-AD fallback). See the recognition rules in
# `@latte`'s body.
function _recognition_expr(blocks, posargs_t::Tuple)
    random_blocks = [b for b in blocks if _classify_block(b, posargs_t) === :random]
    isempty(random_blocks) && return :(nothing)

    hp_syms = Set(b.lhs_sym for b in blocks if _classify_block(b, posargs_t) === :fixed)
    random_syms = Set(b.lhs_sym for b in random_blocks)

    # All-or-nothing: every random block must match the curried shape, draw
    # only from hyperparameters, and have a constructor free of hp / random
    # dependencies. Any miss falls the whole body back to the DAG path.
    entries = Expr[]
    for blk in random_blocks
        blk.rhs === nothing && return :(nothing)
        rec = _recognize_latent_rhs(blk.rhs)
        if rec !== nothing
            # Curried `Family(args)(; k = hp)`: the route must draw only from
            # hyperparameters, and the constructor must be free of hp / latent
            # dependencies.
            ctor_expr, route = rec
            all(hp in hp_syms for (_k, hp) in route) || return :(nothing)
            ctor_free = _free_symbols(ctor_expr, Set{Symbol}(posargs_t), Dict{Symbol, Set{Symbol}}())
            isempty(intersect(ctor_free, union(hp_syms, random_syms))) || return :(nothing)
        else
            # Hyperparameter-free (fixed) prior, e.g. `β ~ MvNormal(zeros(p), c·I)`
            # or `β ~ FixedEffectsModel(p; λ)`: the whole RHS is the constructor
            # and the route is empty. Require it to be free of hp / latent deps
            # so it is genuinely fixed; the runtime coerces the value to a
            # `LatentModel` (falling back to the DAG path if it can't).
            rhs_free = _free_symbols(blk.rhs, Set{Symbol}(posargs_t), Dict{Symbol, Set{Symbol}}())
            isempty(intersect(rhs_free, union(hp_syms, random_syms))) || return :(nothing)
            ctor_expr = blk.rhs
            route = Pair{Symbol, Symbol}[]
        end

        closure = Expr(:->, Expr(:tuple, posargs_t...), ctor_expr)
        route_nt = Expr(:tuple, Expr(:parameters, [Expr(:kw, k, QuoteNode(hp)) for (k, hp) in route]...))
        push!(
            entries,
            Expr(
                :tuple,
                Expr(
                    :parameters,
                    Expr(:kw, :sym, QuoteNode(blk.lhs_sym)),
                    Expr(:kw, :ctor, esc(closure)),
                    Expr(:kw, :route, route_nt),
                ),
            ),
        )
    end
    return Expr(:vect, entries...)
end

# Rewrite a recognized curried latent RHS `Family(args)(; k = hp, …)` to the
# probing stand-in `_recognized_latent_probe(Family(args), (; k = hp, …))`, so
# the DPPL-model probes never materialise a workspace-only prior. Fixed / plain
# priors (no curried shape) materialise cheaply and are returned unchanged.
function _probe_latent_rhs(rhs)
    rec = _recognize_latent_rhs(rhs)
    rec === nothing && return rhs
    ctor_expr, route = rec
    kw_nt = Expr(:tuple, Expr(:parameters, [Expr(:kw, k, hp) for (k, hp) in route]...))
    return Expr(:call, GlobalRef(@__MODULE__, :_recognized_latent_probe), ctor_expr, kw_nt)
end

# Replace the RHS of any `lhs ~ rhs` whose top symbol is a key of `repl`.
function _rewrite_random_tildes(ex, repl::AbstractDict)
    ex isa Expr || return ex
    if ex.head === :call && length(ex.args) == 3 && ex.args[1] === :~
        sym = _lhs_top_sym(ex.args[2])
        if sym !== nothing && haskey(repl, sym)
            return Expr(:call, :~, ex.args[2], repl[sym])
        end
    end
    return Expr(ex.head, Any[_rewrite_random_tildes(a, repl) for a in ex.args]...)
end

# When recognition fired, swap each recognized curried random block's RHS for
# the probing stand-in. No-op otherwise (the DAG path needs the real prior).
function _maybe_probe_rewrite(body, blocks, posargs_t::Tuple, recognition_expr)
    recognition_expr === nothing && return body
    repl = Dict{Symbol, Any}()
    for b in blocks
        _classify_block(b, posargs_t) === :random || continue
        new_rhs = _probe_latent_rhs(b.rhs)
        new_rhs === b.rhs || (repl[b.lhs_sym] = new_rhs)
    end
    isempty(repl) && return body
    return _rewrite_random_tildes(body, repl)
end

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
        posarg_vals::Tuple = ();
        lift_spec = nothing,
        recognition = nothing,
        structured = nothing,
        likelihood_hessian_pattern = :auto,
        augment::Bool = false,
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

    # Recognition path: the macro recognized a concrete `LatentModel`. Build
    # the prebuilt latent and reuse the shared obs / hp / augmentation
    # assembly. The bare `latte_from_dppl` fallback keeps today's DAG /
    # sparse-AD behavior.
    if recognition !== nothing
        latent = _build_recognized_latent(recognition, posarg_vals)
        if latent !== nothing
            return _assemble_lgm(
                dppl_model, latent;
                random = random_syms,
                augment = augment,
                obs_groups = obs_groups,
                force_ad_obs_model = needs_ad_fallback,
                likelihood_hessian_pattern = likelihood_hessian_pattern,
                lift_spec = lift_spec,
            )
        end
        # `_build_recognized_latent` returned `nothing`: a recognized RHS turned
        # out not to be a `LatentModel` at runtime. Fall through to the DAG /
        # sparse-AD path below.
    end

    return latte_from_dppl(
        dppl_model;
        random = random_syms,
        augment = augment,
        obs_groups = obs_groups,
        force_ad_obs_model = needs_ad_fallback,
        likelihood_hessian_pattern = likelihood_hessian_pattern,
        lift_spec = lift_spec,
        structured_spec = _build_structured_spec(structured, posarg_vals),
    )
end

# Bundle the macro-emitted `(; builder, obs_builder, layout_builder, obs_syms)` with the model's
# positional-arg values into the spec the adapter consumes. The layout, prior, and observation are
# built lazily under guards, so a codegen miss falls back to the monolithic versions rather than
# erroring here.
_build_structured_spec(::Nothing, ::Tuple) = nothing
_build_structured_spec(structured::NamedTuple, posarg_vals::Tuple) = (
    builder = structured.builder,
    obs_builder = structured.obs_builder,
    layout_builder = structured.layout_builder,
    obs_syms = structured.obs_syms,
    posarg_vals = posarg_vals,
)

# Instantiate the macro-recognized latent prior from the recognition spec and
# the runtime positional-arg values. A single recognized component becomes a
# `RoutedLatentModel`; multiple components are composed (in body order) via an
# upstream `CombinedModel`, wrapped in one `RoutedLatentModel` whose route maps
# the CombinedModel's auto-prefixed kwarg names to the outer hp symbols.
#
# The macro recognizes by *shape* only, so a recognized RHS may instantiate to
# something that isn't a `LatentModel`. Returns `nothing` in that case so the
# caller falls back to the DAG / sparse-AD path.
function _build_recognized_latent(recognition, posarg_vals::Tuple)
    inners = LatentModel[]
    routes = NamedTuple[]
    for e in recognition
        inner = _coerce_latent(e.ctor(posarg_vals...))
        inner isa LatentModel || return nothing
        push!(inners, inner)
        push!(routes, e.route)
    end
    length(inners) == 1 && return RoutedLatentModel(inners[1], routes[1])
    combined = CombinedModel(inners)
    return RoutedLatentModel(combined, _combined_route(inners, routes))
end

# Coerce a recognized RHS value to a `LatentModel`. `LatentModel`s pass through
# (e.g. the inner `RWModel` / `IIDModel` of a recognized curried prior); a fixed
# multivariate-normal prior with zero mean and isotropic covariance `c·I` is
# materialized as a `FixedEffectsModel` (precision `(1/c)·I`). Anything else
# returns `nothing`, sending the caller to the DAG fallback.
_coerce_latent(x::LatentModel) = x
_coerce_latent(d::Distributions.AbstractMvNormal) = _fixed_gaussian_latent(d)
_coerce_latent(_) = nothing

function _fixed_gaussian_latent(d::Distributions.AbstractMvNormal)
    all(iszero, Distributions.mean(d)) || return nothing
    Σ = Distributions.cov(d)
    n = size(Σ, 1)
    c = Σ[1, 1]
    # Only the isotropic `c·I` case maps to a `FixedEffectsModel` (precision
    # `(1/c)·I`). Verify structure without materializing a dense identity:
    # off-diagonals zero (`isdiag`) and a constant diagonal.
    (c > 0 && LinearAlgebra.isdiag(Σ) && all(i -> isapprox(Σ[i, i], c), 1:n)) || return nothing
    return GaussianMarkovRandomFields.FixedEffectsModel(n; λ = inv(c))
end

# Build the route for a `CombinedModel`-backed `RoutedLatentModel`. Replicates
# CombinedModel's hyperparameter-prefixing (`model_name` + `""`/`_2`/`_3`
# suffix for duplicate names) so each component's inner kwarg `k` maps to the
# combined kwarg `Symbol("$(k)_$(final_name)")`, pointing at the outer hp
# symbol the component draws from.
function _combined_route(inners, routes)
    out = Pair{Symbol, Symbol}[]
    name_counts = Dict{Symbol, Int}()
    for (inner, route) in zip(inners, routes)
        base = model_name(inner)
        name_counts[base] = get(name_counts, base, 0) + 1
        suffix = name_counts[base] == 1 ? "" : "_$(name_counts[base])"
        final_name = Symbol("$(base)$(suffix)")
        for (k, hp) in pairs(route)
            push!(out, Symbol("$(k)_$(final_name)") => hp)
        end
    end
    return (; out...)
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
