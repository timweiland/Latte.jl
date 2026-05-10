# Likelihood fast-paths: detect common `y[i] ~ Distribution(link(linear(x)))`
# patterns in a DPPL model and substitute GMRFs.jl's hand-coded
# `ExponentialFamily` observation model for the default
# `AutoDiffObservationModel`. Dramatically faster; avoids a nested-AD bug
# in the default path.
#
# Fast-path is a pure optimization: anything we can't recognise falls
# through to the existing AD wrapping unchanged.

using SparseArrays
using ADTypes: AutoSparse, AutoForwardDiff
using DifferentiationInterface
using SparseConnectivityTracer: TracerLocalSparsityDetector
using SparseMatrixColorings: GreedyColoringAlgorithm
using Distributions: Poisson, Bernoulli, Binomial, Normal, NegativeBinomial, Gamma, mean
using GaussianMarkovRandomFields:
    ExponentialFamily, LinearlyTransformedObservationModel,
    LogLink, LogitLink, IdentityLink,
    PoissonObservations, BinomialObservations, NegativeBinomialObservations

# ─── Custom DPPL accumulator: records observation sites ───────────────────
# `PriorDistributionAccumulator` skips observed (y) sites. We mirror its
# shape but hook `accumulate_observe!!` instead, collecting per-site
# (sym, dist, y) tuples in DPPL emission order. The sym is needed for
# per-group filtering (see `obs_groups.jl`); the y for emission-order
# observation payloads in fast-path composite components.
mutable struct _ObsDistributionAccumulator <: DynamicPPL.AbstractAccumulator
    sites::Vector{NamedTuple}
end
_ObsDistributionAccumulator() = _ObsDistributionAccumulator(NamedTuple[])

DynamicPPL.accumulator_name(::Type{_ObsDistributionAccumulator}) = :ObsDistributionAccumulator
Base.copy(a::_ObsDistributionAccumulator) = _ObsDistributionAccumulator(copy(a.sites))
DynamicPPL.reset(a::_ObsDistributionAccumulator) = (empty!(a.sites); a)
DynamicPPL.split(::_ObsDistributionAccumulator) = _ObsDistributionAccumulator()
function DynamicPPL.combine(a::_ObsDistributionAccumulator, b::_ObsDistributionAccumulator)
    return _ObsDistributionAccumulator(vcat(a.sites, b.sites))
end
function DynamicPPL.accumulate_observe!!(a::_ObsDistributionAccumulator, right, left, vn, template)
    push!(a.sites, (; sym = getsym(vn), dist = right, y = left))
    return a
end
function DynamicPPL.accumulate_assume!!(a::_ObsDistributionAccumulator, val, tv, lj, vn, d, t)
    return a
end

"""
    _probe_obs_distribution_sites(dppl_model, hp_nt, latent_nt)

Run `dppl_model` once with hyperparameters fixed to `hp_nt` and latent
variables initialised from `latent_nt`, recording per-site
`(sym, dist, y)` tuples in DPPL emission order.
"""
function _probe_obs_distribution_sites(
        dppl_model, hp_nt::NamedTuple, latent_nt::NamedTuple,
    )
    cond = DynamicPPL.fix(dppl_model, hp_nt)
    vi = DynamicPPL.OnlyAccsVarInfo((_ObsDistributionAccumulator(),))
    vi = last(DynamicPPL.init!!(cond, vi, DynamicPPL.InitFromParams(latent_nt, nothing), DynamicPPL.UnlinkAll()))
    return DynamicPPL.getacc(vi, Val(:ObsDistributionAccumulator)).sites
end

# Distribution-only view, kept for callers that don't need sym/y.
function _probe_obs_distributions(dppl_model, hp_nt::NamedTuple, latent_nt::NamedTuple)
    return [s.dist for s in _probe_obs_distribution_sites(dppl_model, hp_nt, latent_nt)]
end

# ─── Family dispatch: distribution type → (family, link, natural-param) ──
# Extension point: one line per new family. `nothing` signals "not
# supported, punt to AD fallback".
_ef_family_info(::Type{<:Poisson}) = (Poisson, LogLink(), d -> log(mean(d)))
_ef_family_info(::Type{<:Bernoulli}) = (Bernoulli, LogitLink(), d -> (p = mean(d); log(p / (1 - p))))
_ef_family_info(::Type{<:Binomial}) = (Binomial, LogitLink(), d -> log(d.p / (1 - d.p)))
_ef_family_info(::Type{<:Normal}) = (Normal, IdentityLink(), d -> mean(d))
# NegativeBinomial(r, p): mean μ = r(1-p)/p, so η = log μ = log(r(1-p)/p)
_ef_family_info(::Type{<:NegativeBinomial}) = (NegativeBinomial, LogLink(), d -> log(d.r * (1 - d.p) / d.p))
# Gamma(α, θ): mean μ = αθ, so η = log μ = log(αθ)
_ef_family_info(::Type{<:Gamma}) = (Gamma, LogLink(), d -> log(d.α * d.θ))
_ef_family_info(_) = nothing

# Per-family map: inner-kwarg name → getter from a Distribution instance.
# Used by `_infer_fast_component_route` to discover which outer hp drives
# each nuisance kwarg the GMRF `ExponentialFamily{F}` consumes (`σ` for
# Normal, `r` for NegativeBinomial, `phi` for Gamma — see GMRFs's
# `_hyperparameter_names`). Families with no nuisance kwargs (Poisson,
# Bernoulli, Binomial) get an empty NamedTuple.
_fast_family_hyperparam_getters(::Type{<:Normal}) = (σ = d -> d.σ,)
_fast_family_hyperparam_getters(::Type{<:NegativeBinomial}) = (r = d -> d.r,)
_fast_family_hyperparam_getters(::Type{<:Gamma}) = (phi = d -> d.α,)
_fast_family_hyperparam_getters(_) = NamedTuple()

# ─── Fast-path result wrapper ─────────────────────────────────────────────
"""
    _FastObsResult{M, R, P, Y}

Internal return value of the generalised fast-path detector. Carries the
constructed `LinearlyTransformedObservationModel` (with any constant
nuisance kwargs already baked in via `_FixedKwargsObservationModel`)
plus the metadata needed to plug it into a `CompositeObservationModel`:

- `model`: the assembled `LinearlyTransformedObservationModel` (or
  similar — anything that implements `loglik` / `loggrad` / `loghessian`).
  Constants are folded into the base before the LTM wraps it, so the
  composite doesn't need to know about them.
- `route`: a rename-only `NamedTuple{inner_kwargs}(outer_hp_symbols)` —
  what `CompositeObservationModel` uses to forward outer kwargs into the
  component. Only kwargs driven by an outer hp appear here; constant
  kwargs are pre-bound on `model` and absent from the route.
- `pattern`: boolean sparsity pattern of `A'A` for this component, used
  by the adapter to pre-populate the latent prior `Q`.
- `y`: observed values in DPPL emission order, aligned with `A` rows.
"""
struct _FastObsResult{M, R, P, Y}
    model::M
    route::R
    pattern::P
    y::Y
end

# ─── Main detection + assembly ────────────────────────────────────────────
"""
    try_exponential_family_fast_path(dppl_model, random_syms, dims, hp_names)
        -> ObservationModel or nothing

Detect whether the DPPL likelihood is a homogeneous single-family
distribution with a canonical link and a linear predictor in `x`. If so,
return a `LinearlyTransformedObservationModel` wrapping an
`ExponentialFamily` (optionally further wrapped in
`OffsetObservationModel` when the linear predictor has a non-zero
constant term). Return `nothing` for anything non-conformant; caller
falls through to the AD-based wrapping.

Current support:
- Family: `Poisson` + `LogLink`, `Bernoulli` + `LogitLink`, `Normal` +
  `IdentityLink`.
- Predictor: affine in the concatenated latent vector `x = [β; u; ...]`.
  Non-zero constant term (e.g. Poisson log-exposure, Bernoulli logit
  shift, Normal mean offset) is captured by wrapping the base obs model
  in `OffsetObservationModel` — works uniformly for augmented and
  non-augmented LGMs.
- Likelihood: homogeneous (all y sites use the same distribution family).
"""
function try_exponential_family_fast_path(
        dppl_model, random_syms::Tuple, dims::Dict{Symbol, Int}, hp_names::Tuple
    )
    r = _try_exponential_family_fast_path(
        dppl_model, random_syms, dims, hp_names;
        obs_syms = nothing, infer_route = false,
    )
    return r === nothing ? nothing : r.model
end

"""
    try_group_exponential_family_fast_path(dppl_model, group_syms, random_syms, dims, hp_names)
        -> _FastObsResult or nothing

Per-group variant for use inside `obs_groups`. Filters the probe to sites
whose `getsym(vn) ∈ group_syms`, runs the same homogeneous-family +
linearity checks on the subset, and infers a rename-only kwarg route
mapping the family's nuisance kwargs to outer hp names. Returns a
`_FastObsResult` or `nothing` if any check fails.
"""
function try_group_exponential_family_fast_path(
        dppl_model, group_syms::Tuple, random_syms::Tuple,
        dims::Dict{Symbol, Int}, hp_names::Tuple,
    )
    return _try_exponential_family_fast_path(
        dppl_model, random_syms, dims, hp_names;
        obs_syms = group_syms, infer_route = true,
    )
end

# Generalised body. `obs_syms === nothing` means "all sites" (whole-model
# fast path); otherwise filter by sym before checking homogeneity / linearity.
# `infer_route = true` attempts to derive a NamedTuple route mapping the
# family's nuisance kwargs to outer hp names; failure to find a clean
# rename-only route returns `nothing` (caller falls back to AD).
function _try_exponential_family_fast_path(
        dppl_model, random_syms::Tuple, dims::Dict{Symbol, Int}, hp_names::Tuple;
        obs_syms::Union{Nothing, Tuple} = nothing,
        infer_route::Bool = false,
    )
    probe_hp = NamedTuple{hp_names}(Tuple(1.0 for _ in hp_names))
    # Detect scalar (univariate) latents so probe seeding uses scalars,
    # not 1-vectors — DPPL's body for `α ~ Normal(0,1)` needs scalar α.
    is_scalar = Dict(s => _is_scalar_latent(dppl_model, s, probe_hp) for s in random_syms)
    probe_x_nt = NamedTuple{random_syms}(
        Tuple(_zero_seed(is_scalar[s], dims[s]) for s in random_syms)
    )

    # 1) probe per-site (sym, dist, y) and filter to the requested group
    sites = _probe_obs_distribution_sites(dppl_model, probe_hp, probe_x_nt)
    if obs_syms !== nothing
        sites = filter(s -> s.sym in obs_syms, sites)
    end
    isempty(sites) && return nothing

    y_dists = [s.dist for s in sites]

    # 2) homogeneous single-family check + supported family lookup
    T = typeof(first(y_dists))
    all(d -> typeof(d) === T, y_dists) || return nothing
    fam_info = _ef_family_info(T)
    fam_info === nothing && return nothing
    family, link, natural_param = fam_info

    # 3) linearity probe via sparse-AD Jacobian of the natural predictor.
    # Re-run the probe inside `η_of_x_at(hp)`, applying the same group filter
    # so the Jacobian's row count matches the assembled `A`. The closure is
    # parameterised on hp so we can recompute (A, b) at perturbed hp values
    # to detect hp-dependent design matrices and offsets (which would
    # silently freeze hp-coupling at probe_hp and starve the outer hp
    # gradient).
    n_latent = sum(dims[s] for s in random_syms)
    offsets = _component_offsets(random_syms, dims)

    η_of_x_at(hp_nt) = function (x_vec)
        x_nt = NamedTuple{random_syms}(
            Tuple(
                is_scalar[s] ? x_vec[first(offsets[s])] : Vector(x_vec[offsets[s]])
                    for s in random_syms
            )
        )
        these = _probe_obs_distribution_sites(dppl_model, hp_nt, x_nt)
        if obs_syms !== nothing
            these = filter(s -> s.sym in obs_syms, these)
        end
        return [natural_param(s.dist) for s in these]
    end

    backend = AutoSparse(
        AutoForwardDiff();
        sparsity_detector = TracerLocalSparsityDetector(),
        coloring_algorithm = GreedyColoringAlgorithm(),
    )
    # The tracer may fail to flow through black-box likelihoods (e.g.
    # OrdinaryDiffEq solvers). Treat any failure here as "can't prove
    # linearity" → punt to AD fallback.
    # Use a small probe step rather than `ones(n_latent)`. Linear models with
    # per-observation multipliers (e.g. β · t for large t) produce η values
    # that saturate the link's natural-param round-trip when probed at ±1
    # (e.g. logit(σ(367)) → Inf). A small step keeps every η in the
    # well-conditioned range while still detecting non-linearity (which
    # produces O(1) Jacobian differences regardless of probe magnitude).
    probe_step = fill(1.0e-3, n_latent)

    # `compute_affine(hp_nt)` returns `(A, b)` for that hp, or `nothing` if
    # the function isn't linear in x at that hp. Used both for baseline
    # construction and for the hp-perturbation invariance check.
    function compute_affine(hp_nt)
        η_x = η_of_x_at(hp_nt)
        A_local, A_check_local = try
            prep = prepare_jacobian(η_x, backend, zeros(n_latent))
            (
                jacobian(η_x, prep, backend, zeros(n_latent)),
                jacobian(η_x, prep, backend, probe_step),
            )
        catch e
            @debug "fast-path linearity probe failed" exception = e
            return nothing
        end
        # Tolerance must absorb finite-precision noise in the natural-param
        # round-trip; genuine non-linearities produce O(1) differences.
        if !isapprox(A_local, A_check_local; atol = 1.0e-4)
            @debug "fast-path: rejected as non-linear" max_diff = maximum(abs, A_local - A_check_local)
            return nothing
        end
        b_local = η_x(zeros(n_latent))
        return (A_local, b_local)
    end

    baseline_affine = compute_affine(probe_hp)
    baseline_affine === nothing && return nothing
    A, b = baseline_affine

    # 3b) hp-invariance check: re-evaluate (A, b) under one-at-a-time
    # perturbations of each outer hp. If any matrix entry or offset entry
    # varies, the design encodes hp dependence that would be silently
    # frozen by the LTM — reject and let the group fall back to AD.
    # (Codex: parametric A(θ) factorisation is a separate follow-up.)
    if !isempty(hp_names)
        for k_out in hp_names
            hp_pert = NamedTuple{hp_names}(
                Tuple(name === k_out ? 1.5 : 1.0 for name in hp_names)
            )
            pert_affine = compute_affine(hp_pert)
            pert_affine === nothing && return nothing
            A_pert, b_pert = pert_affine
            if !isapprox(A, A_pert; atol = 1.0e-4, rtol = 1.0e-6)
                @debug "fast-path: rejected — A depends on outer hp $(k_out)"
                return nothing
            end
            if !isapprox(b, b_pert; atol = 1.0e-4, rtol = 1.0e-6)
                @debug "fast-path: rejected — offset b depends on outer hp $(k_out)"
                return nothing
            end
        end
    end

    # 4) Rename-only route inference. For each nuisance kwarg the family
    # exposes (e.g. `:σ` for Normal), classify it as either:
    #   - driven by exactly one outer hp 1:1 → record `inner => outer` in route
    #   - constant in the body (no outer hp drives it) → record the
    #     baseline value in `fixed_kwargs`, baked into the component
    # Reject if multiple outer hps influence the kwarg (composite
    # expression / transform), if it differs across sites within the
    # group (heteroskedastic), or if it depends on the latent vector
    # (e.g. `σ = exp(α)` — invisible at the zero probe but real).
    route = nothing
    fixed_kwargs = NamedTuple()
    if infer_route
        route_result = _infer_fast_component_route(
            dppl_model, family, hp_names, probe_x_nt, sites,
        )
        route_result === nothing && return nothing
        route, fixed_kwargs = route_result
    end

    # 5) assemble. Non-zero `b = η(0)` → wrap in OffsetObservationModel
    # (obs-layer offset, works for any LGM shape). Fold any pre-bound
    # constant nuisance kwargs into the base before the LTM wraps it,
    # so the composite's outer kwarg surface stays clean and downstream
    # `obs isa LinearlyTransformedObservationModel` checks still work.
    A_sp = SparseMatrixCSC(A)
    dropzeros!(A_sp)
    base = ExponentialFamily(family, link)
    # Binomial needs per-site trial counts to materialise `BinomialObservations`
    # at likelihood call time. Extract them from the probed distributions and
    # wrap so `_normalize_observations` can assemble (y, trials) pairs.
    if family === Binomial
        trials_vec = Int[d.n for d in y_dists]
        base = BinomialTrialsObservationModel(base, trials_vec)
    end
    obs = all(iszero, b) ? base : OffsetObservationModel(base, Vector{Float64}(b))
    if !isempty(fixed_kwargs)
        obs = _FixedKwargsObservationModel(obs, fixed_kwargs)
    end
    model = LinearlyTransformedObservationModel(obs, A_sp)

    pattern = _bool_pattern_AtA_from_jac(A_sp)
    # Wrap y in the family-appropriate observation type. Whole-model fast
    # path (`obs_syms === nothing`) keeps the raw y vector — `inla(lgm, y)`
    # already wraps it via the LGM's own materialisation path. The composite
    # path baked-in payload skips that wrapping, so for grouped fast
    # components we emit pre-wrapped observations the component's
    # `_materialize` dispatch expects.
    y_emission_order = [s.y for s in sites]
    y_for_component = obs_syms === nothing ?
        y_emission_order :
        _wrap_y_for_fast_component(family, y_emission_order, y_dists)
    return _FastObsResult(model, route, pattern, y_for_component)
end

# Family-specific observation wrapping for grouped fast-path components.
# Poisson / NegativeBinomial / Binomial materialise via dispatch on a
# concrete Observations type; the others (Normal, Bernoulli, Gamma, TDist)
# accept a raw vector at `_materialize` time.
function _wrap_y_for_fast_component(family, y_emission_order, y_dists)
    if family === Poisson
        return PoissonObservations(Int.(y_emission_order))
    elseif family === NegativeBinomial
        return NegativeBinomialObservations(Int.(y_emission_order))
    elseif family === Binomial
        trials_vec = Int[d.n for d in y_dists]
        return BinomialObservations(Int.(y_emission_order), trials_vec)
    end
    return y_emission_order
end

# Boolean pattern of A'A for a sparse design matrix. Mirrors the helper
# in `adapter.jl` but kept local to avoid a circular dep at file-load time.
function _bool_pattern_AtA_from_jac(A::AbstractMatrix)
    pat = SparseMatrixCSC{Bool, Int}(A .!= 0)
    return pat' * pat
end

# Classify the family's nuisance kwargs as either rename-driven (exactly
# one outer hp drives them 1:1) or constant (no outer hp drives them, but
# also invariant under a nonzero latent probe — i.e. genuinely a literal
# in the model body, not a function of latents).
#
# Returns either `(route::NamedTuple{(driving_inners...,)}, fixed::NamedTuple{(constant_inners...,)})`
# or `nothing` when classification fails.
function _infer_fast_component_route(
        dppl_model, family, hp_names::Tuple,
        probe_x_nt::NamedTuple, baseline_sites::AbstractVector,
    )
    getters = _fast_family_hyperparam_getters(family)
    # No nuisance kwargs (Poisson, Bernoulli, Binomial) → empty route, no constants.
    isempty(getters) && return (NamedTuple(), NamedTuple())

    inner_names = keys(getters)
    baseline_syms = unique(s.sym for s in baseline_sites)
    base_hp_val = 1.0
    base_hp_nt = NamedTuple{hp_names}(Tuple(base_hp_val for _ in hp_names))

    # A nonzero latent probe — used to verify nuisance kwargs don't
    # depend on the latent vector (e.g. `σ = exp(α)` would slip through
    # the hp-only checks since at probe_x_nt = zero, `exp(0) = 1`).
    latent_pert_nt = _perturb_latent_probe(probe_x_nt)

    route_pairs = Pair{Symbol, Symbol}[]
    fixed_pairs = Pair{Symbol, Float64}[]

    for inner_name in inner_names
        getter = getproperty(getters, inner_name)
        baseline_vals = [Float64(getter(s.dist)) for s in baseline_sites]
        if !all(isapprox(baseline_vals[1]; atol = 1.0e-9), baseline_vals)
            return nothing  # heteroskedastic at baseline
        end
        baseline_v = baseline_vals[1]

        # 1) hp-perturbation pass: which outer hps influence this kwarg?
        matches = Symbol[]
        for k_out in hp_names
            perturbed = base_hp_val + 0.5     # 1.5
            hp_pert = NamedTuple{hp_names}(
                Tuple(name === k_out ? perturbed : base_hp_val for name in hp_names)
            )
            sites_pert = _probe_obs_distribution_sites(dppl_model, hp_pert, probe_x_nt)
            sites_pert = filter(s -> s.sym in baseline_syms, sites_pert)
            isempty(sites_pert) && return nothing
            v_pert = [Float64(getter(s.dist)) for s in sites_pert]
            if !all(isapprox(v_pert[1]; atol = 1.0e-9), v_pert)
                return nothing  # heteroskedastic under perturbation
            end
            v = v_pert[1]
            if isapprox(v, perturbed; atol = 1.0e-9)
                push!(matches, k_out)
            elseif !isapprox(v, baseline_v; atol = 1.0e-9)
                # Outer hp influenced the kwarg through some transform we
                # can't represent as a rename — bail to AD.
                return nothing
            end
        end

        # 2) latent-invariance check: kwarg must NOT depend on latents,
        # whether classified as driven or fixed. The fast path encodes
        # nuisance values as a per-component NamedTuple at materialisation
        # time; a latent-dependent value can't be represented.
        sites_x = _probe_obs_distribution_sites(dppl_model, base_hp_nt, latent_pert_nt)
        sites_x = filter(s -> s.sym in baseline_syms, sites_x)
        isempty(sites_x) && return nothing
        v_x = [Float64(getter(s.dist)) for s in sites_x]
        if !all(isapprox(v_x[1]; atol = 1.0e-9), v_x)
            return nothing  # heteroskedastic under latent perturbation
        end
        if !isapprox(v_x[1], baseline_v; atol = 1.0e-9)
            return nothing  # latent-dependent — fast path can't represent this
        end

        if length(matches) == 1
            push!(route_pairs, inner_name => only(matches))
        elseif length(matches) == 0
            push!(fixed_pairs, inner_name => baseline_v)
        else
            return nothing  # shared across multiple outer hps
        end
    end

    route_nt = NamedTuple{Tuple(p.first for p in route_pairs)}(
        Tuple(p.second for p in route_pairs)
    )
    fixed_nt = NamedTuple{Tuple(p.first for p in fixed_pairs)}(
        Tuple(p.second for p in fixed_pairs)
    )
    return (route_nt, fixed_nt)
end

# Build a small nonzero latent probe to test invariance of the nuisance
# kwargs. Using a *small* magnitude (1e-2) keeps the link's natural-param
# round-trip well-conditioned — the same reason as the linearity probe.
function _perturb_latent_probe(probe_x_nt::NamedTuple)
    return NamedTuple{keys(probe_x_nt)}(
        Tuple(_latent_pert_for(v) for v in values(probe_x_nt))
    )
end
_latent_pert_for(v::Real) = oftype(v, 1.0e-2)
_latent_pert_for(v::AbstractVector) = fill!(similar(v), oftype(eltype(v)(0), 1.0e-2))

# ─── Small helper shared with obs_model.jl ────────────────────────────────
function _component_offsets(random_syms::Tuple, dims::Dict{Symbol, Int})
    offsets = Dict{Symbol, UnitRange{Int}}()
    off = 0
    for s in random_syms
        offsets[s] = (off + 1):(off + dims[s])
        off += dims[s]
    end
    return offsets
end
