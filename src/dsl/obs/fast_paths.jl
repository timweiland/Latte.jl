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
    NonlinearLeastSquaresModel,
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
`ExponentialFamily` (carrying an `offset` when the linear predictor has a
non-zero constant term). Return `nothing` for anything non-conformant;
caller falls through to the AD-based wrapping.

Current support:
- Family: `Poisson` + `LogLink`, `Bernoulli` + `LogitLink`, `Normal` +
  `IdentityLink`.
- Predictor: affine in the concatenated latent vector `x = [β; u; ...]`.
  Non-zero constant term (e.g. Poisson log-exposure, Bernoulli logit
  shift, Normal mean offset) is captured by the LTM's `offset` (η = A·x +
  b). For the composite path it rides the per-component forward-mode IFT;
  for the single-obs path the augmenting LGM constructor absorbs it into
  the augmented prior mean.
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
        dims::Dict{Symbol, Int}, hp_names::Tuple;
        nls_enabled::Bool = true,
    )
    return _try_exponential_family_fast_path(
        dppl_model, random_syms, dims, hp_names;
        obs_syms = group_syms, infer_route = true,
        nls_enabled = nls_enabled,
    )
end

# ─── θ-dependent design-matrix builders for ParameterizedMatrix ───────────
# A θ-dependent design A(θ) must be rebuilt per θ, including under a Dual θ
# (outer ForwardDiff). Two strategies, fast path first:

# Affine: A(θ) = A₀ + Σₖ θₖ·Aₖ. Extract the intercept A₀ and slopes Aₖ ONCE via
# the sparse Jacobian at *primal* probe points (it only breaks under Dual θ),
# verify affine-ness at an all-perturbed point, and return a builder that forms
# A(θ) as a sparse linear combination — O(nnz) per θ, Dual-safe, no per-θ
# predictor evaluations. Returns `nothing` if A is not affine in θ.
function _affine_design_builder(compute_affine, hp_names::Tuple, A_names::Tuple, probe_hp::NamedTuple)
    δ = 0.5
    base = compute_affine(probe_hp)
    base === nothing && return nothing
    A0m = SparseMatrixCSC(base[1])
    colptr, rowval = A0m.colptr, A0m.rowval
    base_nz = A0m.nzval
    slopes = Vector{Vector{Float64}}(undef, length(A_names))
    nz0 = copy(base_nz)
    for (i, k) in enumerate(A_names)
        hpk = NamedTuple{hp_names}(map(s -> s === k ? probe_hp[s] + δ : probe_hp[s], hp_names))
        r = compute_affine(hpk)
        r === nothing && return nothing
        m = SparseMatrixCSC(r[1])
        (m.colptr == colptr && m.rowval == rowval) || return nothing
        sl = (m.nzval .- base_nz) ./ δ
        slopes[i] = sl
        nz0 = nz0 .- probe_hp[k] .* sl
    end
    hp_all = NamedTuple{hp_names}(map(s -> s in A_names ? probe_hp[s] + δ : probe_hp[s], hp_names))
    rall = compute_affine(hp_all)
    rall === nothing && return nothing
    mall = SparseMatrixCSC(rall[1])
    (mall.colptr == colptr && mall.rowval == rowval) || return nothing
    pred = copy(nz0)
    for (i, k) in enumerate(A_names)
        pred = pred .+ (probe_hp[k] + δ) .* slopes[i]
    end
    isapprox(pred, mall.nzval; atol = 1.0e-6, rtol = 1.0e-4) || return nothing
    mrows, ncols = size(A0m)
    return let nz0 = nz0, slopes = slopes, A_names = A_names,
            colptr = colptr, rowval = rowval, mrows = mrows, ncols = ncols
        (; kw...) -> begin
            θ = NamedTuple(kw)
            nz = nz0 .+ sum(θ[A_names[i]] .* slopes[i] for i in eachindex(A_names))
            SparseMatrixCSC(mrows, ncols, copy(colptr), copy(rowval), nz)
        end
    end
end

# Fallback: rebuild A(θ) column-by-column via A[:, j] = η(eⱼ; θ) − η(0; θ).
# Plain predictor evaluations (no inner AD), so an outer Dual θ flows through —
# but O(nonzero-columns) evaluations per θ. Used only when A is non-affine.
function _column_design_builder(η_of_x_at, hp_names::Tuple, A_names::Tuple, probe_hp::NamedTuple, n_latent::Int, pat::SparseMatrixCSC)
    return let η_fn = η_of_x_at, all_names = hp_names, A_names = A_names,
            probe = probe_hp, nlat = n_latent, pat = pat
        (; kw...) -> begin
            θ = NamedTuple(kw)
            hp_nt = NamedTuple{all_names}(map(s -> (s in A_names ? θ[s] : probe[s]), all_names))
            η = η_fn(hp_nt)
            b0 = η(zeros(nlat))
            nzval = Vector{eltype(b0)}(undef, length(pat.nzval))
            e = zeros(nlat)
            @inbounds for j in 1:nlat
                r = nzrange(pat, j)
                isempty(r) && continue
                e[j] = 1.0
                colj = η(e) .- b0
                e[j] = 0.0
                for kk in r
                    nzval[kk] = colj[pat.rowval[kk]]
                end
            end
            SparseMatrixCSC(pat.m, pat.n, copy(pat.colptr), copy(pat.rowval), nzval)
        end
    end
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
        nls_only::Bool = false,
        nls_enabled::Bool = true,
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
    # The predictor isn't affine in x, OR (for Normal noise) the tiny-step affine
    # probe above false-accepted a mildly-curved mean as affine. Either way, a
    # Gaussian obs `y[i] ~ Normal(f(x), σ)` with a curved forward map `f` is the
    # Nonlinear Least Squares (Gauss–Newton) case. With NLS enabled (the default)
    # dispatch it to GMRFs's `NonlinearLeastSquaresModel`; with NLS opted out
    # (`nls = false`) punt to the exact AD path — never the affine linearization,
    # which is exactly the approximation a curved mean must avoid.
    treat_nonlinear = baseline_affine === nothing ||
        (family === Normal && _predictor_is_curved(η_of_x_at(probe_hp), n_latent, backend))
    if treat_nonlinear
        nls_enabled || return nothing
        return _try_nls_fast_obs(
            family, η_of_x_at, probe_hp, hp_names, n_latent,
            sites, y_dists, obs_syms, dppl_model, probe_x_nt, backend, probe_step,
        )
    end
    # `nls_only` runs when AD was forced for the EF route (the macro's obs-shadow
    # heuristic) but the NLS route is still permitted. An affine predictor here
    # is EF-eligible — respect the AD-force and punt, leaving EF behavior intact.
    nls_only && return nothing
    A, b = baseline_affine

    # 3b) hp-invariance check: re-evaluate (A, b) under one-at-a-time
    # perturbations of each outer hp. Both the design matrix `A` and the
    # offset `b` may depend on outer hp — each is *kept* (parameterized at
    # assembly) rather than frozen. `A_hp_names` / `b_hp_names` collect the
    # outer hp the design / offset depend on.
    A_hp_names = Symbol[]
    b_hp_names = Symbol[]
    if !isempty(hp_names)
        for k_out in hp_names
            hp_pert = NamedTuple{hp_names}(
                Tuple(name === k_out ? 1.5 : 1.0 for name in hp_names)
            )
            pert_affine = compute_affine(hp_pert)
            pert_affine === nothing && return nothing
            A_pert, b_pert = pert_affine
            if !isapprox(A, A_pert; atol = 1.0e-4, rtol = 1.0e-6)
                push!(A_hp_names, k_out)
            end
            if !isapprox(b, b_pert; atol = 1.0e-4, rtol = 1.0e-6)
                push!(b_hp_names, k_out)
            end
        end
    end

    # A θ-dependent design matrix becomes a `ParameterizedMatrix`, which only
    # the composite path can carry: the single-obs path augments the LTM, and
    # `AugmentedLatentModel` needs a concrete design matrix (a θ-dependent A
    # there would make the augmented precision pattern θ-dependent). So on the
    # whole-model (augmented) path a θ-dependent A still falls back to AD.
    if !isempty(A_hp_names) && !infer_route
        @debug "fast-path: rejected — A depends on outer hp on the augmented path"
        return nothing
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

    # 5) assemble. The linear predictor is η = A·x + b; the constant term
    # `b = η(0)` (possibly θ-dependent) becomes the LTM's offset. The composite
    # path rides the per-component forward-mode IFT through that offset; the
    # single-obs path's augmenting LGM constructor absorbs the offset into the
    # augmented prior mean. Fold any pre-bound constant nuisance kwargs into the
    # base first, so the composite's outer kwarg surface stays clean.
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
    inner = isempty(fixed_kwargs) ? base : _FixedKwargsObservationModel(base, fixed_kwargs)
    # θ-dependent design matrix → ParameterizedMatrix. Prefer the affine
    # decomposition A(θ) = A₀ + Σₖ θₖ·Aₖ (built once, O(nnz) per θ, Dual-safe);
    # fall back to per-θ column extraction only for genuinely nonlinear A(θ).
    # A θ-invariant A stays a plain sparse matrix (zero overhead). The composite
    # IFT threads the resulting Dual-valued A through.
    design = if !isempty(A_hp_names)
        A_names = Tuple(A_hp_names)
        A_builder = _affine_design_builder(compute_affine, hp_names, A_names, probe_hp)
        if A_builder === nothing
            A_builder = _column_design_builder(η_of_x_at, hp_names, A_names, probe_hp, n_latent, A_sp)
        end
        GaussianMarkovRandomFields.ParameterizedMatrix(A_builder; hyperparameters = A_names, n_latent = n_latent)
    else
        A_sp
    end
    offset = if !isempty(b_hp_names)
        # θ-dependent offset: recompute b = η(x=0; θ) per θ by re-probing the
        # predictor at zero. Only the offset-affecting hp (`b_names`) move it;
        # the rest take their probe value, so a grouped component only routes
        # `b_names`.
        b_names = Tuple(b_hp_names)
        offset_builder = let η_fn = η_of_x_at, all_names = hp_names, b_names = b_names,
                probe = probe_hp, nlat = n_latent
            (; kw...) -> begin
                θ = NamedTuple(kw)
                hp_nt = NamedTuple{all_names}(map(s -> (s in b_names ? θ[s] : probe[s]), all_names))
                η_fn(hp_nt)(zeros(nlat))
            end
        end
        GaussianMarkovRandomFields.ParameterizedOffset(offset_builder; hyperparameters = b_names)
    elseif all(iszero, b)
        nothing
    else
        Vector{Float64}(b)
    end
    model = offset === nothing ?
        LinearlyTransformedObservationModel(inner, design) :
        LinearlyTransformedObservationModel(inner, design; offset = offset)

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
    # Thread the design- and offset-hyperparameters through the per-component
    # composite route (1:1), so a grouped component receives them. The
    # single-obs path ignores `route` (the obs is materialised with all hp
    # directly).
    extra_hp = unique(vcat(A_hp_names, b_hp_names))
    if !isempty(extra_hp)
        extra = NamedTuple{Tuple(extra_hp)}(Tuple(extra_hp))
        route = route === nothing ? extra : merge(route, extra)
    end
    return _FastObsResult(model, route, pattern, y_for_component)
end

# ─── Nonlinear-least-squares fast path ────────────────────────────────────
"""
    _predictor_is_curved(f, n_latent, backend) -> Bool

Curvature-direct affine check for the Normal/identity case. The 2-point Jacobian
probe in `_try_exponential_family_fast_path` uses a tiny step (kept small so link
round-trips don't saturate), so it can miss mild curvature and false-accept a
curved mean as affine. Normal/identity has no link round-trip, so we compare the
forward-map Jacobian at the zero seed against a unit-scale step: a genuinely
affine mean has a constant Jacobian; a curved one does not. Applied only for
Normal noise — other families keep the tiny-step probe unchanged. On any probe
failure we report curved (route to NLS/AD) rather than risk mis-linearizing.
"""
function _predictor_is_curved(f, n_latent::Int, backend)
    return try
        prep = prepare_jacobian(f, backend, zeros(n_latent))
        J0 = jacobian(f, prep, backend, zeros(n_latent))
        J1 = jacobian(f, prep, backend, ones(n_latent))
        !isapprox(J0, J1; atol = 1.0e-4, rtol = 1.0e-6)
    catch e
        @debug "NLS curvature probe failed; routing to NLS/AD to be safe" exception = e
        true
    end
end

"""
    _try_nls_fast_obs(family, η_of_x_at, probe_hp, hp_names, n_latent, sites,
                      y_dists, obs_syms, dppl_model, probe_x_nt, backend, probe_step)
        -> _FastObsResult or nothing

Reached when the natural-predictor linearity probe fails — the mean is
nonlinear in the latent vector `x`. For a Gaussian observation
`y[i] ~ Normal(f(x), σ)` this is the Nonlinear Least Squares case, dispatched
to GMRFs's `NonlinearLeastSquaresModel` (Gauss–Newton). `f = η_of_x_at(probe_hp)`
is the forward operator (identity link ⇒ the natural param *is* the mean).

The forward map `f` may depend on outer hyperparameters: those are carried into
the residual via the NLS model's `hyperparams`, so the outer θ-gradient
differentiates `f(x; θ)` exactly (only the latent Hessian is Gauss–Newton). The
noise scale σ may be:
- constant — shared or per-site (heteroskedastic), frozen into the model;
- driven 1:1 by a hyperparameter named `:σ` — left flowing so the noise scale is
  inferred (the single-obs path routes hyperparameters by name, matching the
  exponential-family route).
Any of the following punts back to the AD fallback (returns `nothing`):
- non-Normal noise,
- σ that depends on the latent vector, on multiple hyperparameters, on a
  differently-named hyperparameter, or on a transform of one.
"""
function _try_nls_fast_obs(
        family, η_of_x_at, probe_hp::NamedTuple, hp_names::Tuple, n_latent::Int,
        sites::AbstractVector, y_dists::AbstractVector, obs_syms,
        dppl_model, probe_x_nt::NamedTuple, backend, probe_step,
    )
    family === Normal || return nothing                       # NLS = Gaussian noise only

    σ_vals = Float64[d.σ for d in y_dists]
    homoskedastic = all(s -> isapprox(s, first(σ_vals); rtol = 1.0e-8), σ_vals)

    f = η_of_x_at(probe_hp)
    # Probe the mean's hp-dependence at both the zero seed and a nonzero latent:
    # a multiplicative dependence such as `exp(α·x)` is invisible at x = 0
    # (`exp(0) = 1` for every α) but shows at a nonzero x.
    x_nz = fill(0.5, n_latent)

    # Classify each outer hp: does it drive the mean (→ parameterized residual),
    # and/or σ (→ routed or punted below)? Probing evaluates the forward map and
    # re-runs the model body, so guard it — any failure punts to the AD fallback.
    mean_hp_names = Symbol[]
    σ_driving_hp = Symbol[]
    sigma_latent_invariant = try
        η0 = f(zeros(n_latent))
        η_nz = f(x_nz)
        for k_out in hp_names
            hp_pert = NamedTuple{hp_names}(
                Tuple(name === k_out ? 1.5 : 1.0 for name in hp_names)
            )
            f_pert = η_of_x_at(hp_pert)
            mean_moves = !isapprox(η0, f_pert(zeros(n_latent)); atol = 1.0e-4, rtol = 1.0e-6) ||
                !isapprox(η_nz, f_pert(x_nz); atol = 1.0e-4, rtol = 1.0e-6)
            mean_moves && push!(mean_hp_names, k_out)
            pert = _probe_obs_distribution_sites(dppl_model, hp_pert, probe_x_nt)
            obs_syms !== nothing && (pert = filter(s -> s.sym in obs_syms, pert))
            σ_pert = Float64[s.dist.σ for s in pert]
            all(i -> isapprox(σ_pert[i], σ_vals[i]; rtol = 1.0e-8), eachindex(σ_vals)) ||
                push!(σ_driving_hp, k_out)
        end
        # σ must be latent-invariant. A latent-dependent noise such as
        # `Normal(f(x), exp(x))` looks constant at the zero seed, but freezing or
        # routing σ would fit the wrong model. Re-probe at a perturbed latent.
        sites_xp = _probe_obs_distribution_sites(dppl_model, probe_hp, _perturb_latent_probe(probe_x_nt))
        obs_syms !== nothing && (sites_xp = filter(s -> s.sym in obs_syms, sites_xp))
        σ_xp = Float64[s.dist.σ for s in sites_xp]
        all(i -> isapprox(σ_xp[i], σ_vals[i]; rtol = 1.0e-8), eachindex(σ_vals))
    catch e
        @debug "NLS fast-path: hp-classification probe failed" exception = e
        return nothing
    end
    sigma_latent_invariant || return nothing

    # Decide how σ enters the model:
    #  - no hp drives it → freeze the constant (scalar if shared, else per-site);
    #  - exactly the `:σ` hyperparameter drives it 1:1 (identity) and it's shared
    #    across sites → leave σ flowing so the outer θ-gradient infers it;
    #  - anything else (multiple hps, a transform, a differently-named hp,
    #    heteroskedastic-and-hp-driven) → punt to AD.
    σ_fixed = if isempty(σ_driving_hp)
        homoskedastic ? first(σ_vals) : σ_vals
    elseif homoskedastic && σ_driving_hp == [:σ] &&
            _sigma_tracks_hp_identity(dppl_model, probe_hp, probe_x_nt, obs_syms)
        nothing                                               # σ flows as a hyperparameter
    else
        return nothing
    end

    # Sparse Jacobian of the forward map → the Gauss–Newton Hessian pattern J'J.
    # The ∂f/∂x sparsity pattern is hp-invariant (hp only scale the values), so
    # probing at `probe_hp` is enough even for a hp-dependent mean.
    J = try
        prep = prepare_jacobian(f, backend, zeros(n_latent))
        SparseMatrixCSC(jacobian(f, prep, backend, probe_step))
    catch e
        @debug "NLS fast-path: Jacobian sparsity probe failed" exception = e
        return nothing
    end
    pattern = _bool_pattern_AtA_from_jac(J)

    # hp-free mean → bake `probe_hp` into the residual; hp-dependent mean → a
    # parameterized residual `f(x; θ...)` carrying `mean_hp_names`, which NLS
    # splats in at materialization (Dual-θ safe, so the IFT θ-gradient is exact).
    residual, nls_hyperparams = if isempty(mean_hp_names)
        (f, ())
    else
        (
            _make_nls_residual(η_of_x_at, hp_names, mean_hp_names, probe_hp),
            Tuple(mean_hp_names),
        )
    end
    nls = NonlinearLeastSquaresModel(residual, n_latent; hyperparams = nls_hyperparams)
    inner = σ_fixed === nothing ? nls : _FixedKwargsObservationModel(nls, (; σ = σ_fixed))
    y_emission_order = [s.y for s in sites]
    y_for_component = obs_syms === nothing ?
        y_emission_order :
        _wrap_y_for_fast_component(family, y_emission_order, y_dists)
    return _FastObsResult(inner, nothing, pattern, y_for_component)
end

# Confirm the noise scale σ equals the `:σ` hyperparameter itself (identity map),
# not a transform of it. Probe at two distinct `:σ` values: if the observed σ
# matches each, the body passes σ straight through, so leaving it flowing is
# exact. A transform such as `1/sqrt(σ)` fails this and punts.
function _sigma_tracks_hp_identity(dppl_model, probe_hp::NamedTuple, probe_x_nt, obs_syms)
    for σ_val in (1.0, 1.5)
        hp = merge(probe_hp, (; σ = σ_val))
        s = _probe_obs_distribution_sites(dppl_model, hp, probe_x_nt)
        obs_syms !== nothing && (s = filter(t -> t.sym in obs_syms, s))
        all(t -> isapprox(t.dist.σ, σ_val; rtol = 1.0e-8), s) || return false
    end
    return true
end

# Build a parameterized NLS residual `f(x; θ...)` for a mean that depends on the
# `mean_hp_names` hyperparameters. NLS splats those θ in at materialization; the
# remaining hyperparameters don't affect the mean, so they take their probe value
# (the residual re-evaluates the natural predictor through `η_of_x_at`). Keeping
# the dependence symbolic — rather than freezing it at the probe — is what lets
# the outer θ-gradient differentiate the forward map exactly.
function _make_nls_residual(η_of_x_at, hp_names::Tuple, mean_hp_names, probe_hp::NamedTuple)
    mean_set = Tuple(mean_hp_names)
    return function (x; kw...)
        θ = NamedTuple(kw)
        full = NamedTuple{hp_names}(
            map(s -> (s in mean_set ? θ[s] : probe_hp[s]), hp_names)
        )
        return η_of_x_at(full)(x)
    end
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
