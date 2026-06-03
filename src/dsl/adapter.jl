# One-shot combinator: DynamicPPL model → `LatentGaussianModel`.
#
# The entry point users call:
#
#     model = latte_from_dppl(dppl_model; random = (:β, :u))
#
# Then feed `model` into any inference method: `inla(model, y)`, `tmb(model, y)`,
# etc. The adapter classifies random components, extracts the latent DAG,
# builds the hyperparameter spec via `Bijectors.bijector`, and wraps the
# likelihood as an `AutoDiffObservationModel`.

using Distributions: UnivariateDistribution
using OrderedCollections: OrderedDict

export latte_from_dppl

"""
    latte_from_dppl(dppl_model; random::Union{Symbol, Tuple}) -> LatentGaussianModel

Turn a DynamicPPL `@model` into a `LatentGaussianModel`.

`random` lists the symbols of the latent Gaussian components (the "random
effects" in TMB / MixedModels vocabulary — the conditionally-Gaussian field
that inference methods Laplace-integrate or sample over).

All remaining scalar univariate priors in the model are treated as
hyperparameters. Non-scalar non-`random` priors are not supported.

The returned model can be fed into any Latte inference method:

```julia
using Distributions: UnivariateDistribution

@model function m(y, X, group)
    τ ~ Gamma(2, 1)
    β ~ MvNormal(zeros(size(X, 2)), 100.0 * I)
    u ~ MvNormal(zeros(maximum(group)), (1 / τ) * I)
    for i in eachindex(y)
        y[i] ~ Poisson(exp(X[i, :]' * β + u[group[i]]); check_args = false)
    end
end

lgm = latte_from_dppl(m(y_obs, X, group); random = (:β, :u))
result = inla(lgm, y_obs)     # or tmb(lgm, y_obs), ...
```
"""
function latte_from_dppl(
        dppl_model;
        random::Union{Symbol, Tuple},
        force_ad_obs_model::Bool = false,
        augment::Bool = true,
        likelihood_hessian_pattern::Union{Symbol, SparseMatrixCSC} = :auto,
        obs_groups = nothing,
        lift_spec = nothing,
    )
    return _assemble_lgm(
        dppl_model, nothing;
        random = random,
        force_ad_obs_model = force_ad_obs_model,
        augment = augment,
        likelihood_hessian_pattern = likelihood_hessian_pattern,
        obs_groups = obs_groups,
        lift_spec = lift_spec,
    )
end

"""
    _assemble_lgm(dppl_model, latent_override; random, kwargs...)

Internal shared LGM assembly. When `latent_override === nothing` the latent
prior is extracted from the DPPL graph via `build_latent_model` (DAG /
sparse-AD). When a prebuilt latent is supplied — the `@latte` macro's
concrete-`LatentModel` recognition path passes a `RoutedLatentModel` /
`CombinedModel` — it is used as-is. Hyperparameter spec, obs-model fast-path
/ grouping / AD fallback, pattern plumbing, augmentation, and latent layout
are shared across both paths.
"""
function _assemble_lgm(
        dppl_model, latent_override;
        random::Union{Symbol, Tuple},
        force_ad_obs_model::Bool = false,
        augment::Bool = true,
        likelihood_hessian_pattern::Union{Symbol, SparseMatrixCSC} = :auto,
        obs_groups = nothing,
        lift_spec = nothing,
    )
    random_syms = random isa Symbol ? (random,) : random
    priors = extract_priors(dppl_model)
    random_set = Set(random_syms)

    # Hyperparameters = scalar univariate priors not in `random`.
    hp_names = Tuple(
        unique(
            getsym(vn) for (vn, d) in pairs(priors)
                if !(getsym(vn) in random_set) && d isa UnivariateDistribution
        )
    )

    spec = extract_hp_spec(dppl_model, hp_names)

    # Probe dims (used by both fast-path detection and obs-model extraction).
    probe_hp = NamedTuple{hp_names}(Tuple(1.0 for _ in hp_names))
    dims = Dict(s => variable_length(dppl_model, s, probe_hp) for s in random_syms)

    # Composite-obs path: when `obs_groups` is supplied each group goes
    # through per-group fast-path detection first; only groups that fail
    # detection (heterogeneous family, non-linear predictor, transformed
    # nuisance hp, …) fall back to the AD per-component closure.
    obs_groups_spec = _normalize_obs_groups(obs_groups)
    use_obs_groups = obs_groups_spec !== nothing
    if use_obs_groups
        _validate_obs_groups(obs_groups_spec, dppl_model, hp_names, random_syms, dims)
    end

    # Detect whole-model fast path first (skipped when grouping is requested).
    fast_obs = (force_ad_obs_model || use_obs_groups) ? nothing :
        try_exponential_family_fast_path(dppl_model, random_syms, dims, hp_names)
    use_fast_path = fast_obs !== nothing

    # Per-group fast-path planning. Only attempted when grouping is in
    # play and the user hasn't forced AD.
    group_fast = if use_obs_groups && !force_ad_obs_model
        Dict(
            name => try_group_exponential_family_fast_path(
                    dppl_model, syms, random_syms, dims, hp_names,
                ) for (name, syms) in obs_groups_spec
        )
    else
        Dict{Symbol, Any}()
    end
    obs_groups_need_ad = use_obs_groups && any(
        get(group_fast, name, nothing) === nothing for (name, _) in obs_groups_spec
    )

    # Pattern plumbing:
    #  - Augmented fast-path: LGM auto-augmentation already covers the
    #    `A'A` pattern in Q_joint's bottom-right block, so no extra
    #    union needed on the base Q.
    #  - Non-augmented fast-path: likelihood is `y ~ family(link(A·x))`
    #    with Hessian pattern `A' · diag · A` = `A'A`'s pattern. Q_base
    #    must pre-include this pattern for the workspace's symbolic
    #    factorization. Compute `A'A`'s boolean pattern directly from A.
    #  - AD fallback (augmented or not): auto-detect via DPPL.
    n_tot = sum(dims[s] for s in random_syms)
    dense_pattern() = SparseMatrixCSC{Bool, Int}(trues(n_tot, n_tot))

    extra_pattern = if use_fast_path && !augment
        A = _extract_design_matrix(fast_obs)
        _bool_AtA_pattern(A)
    elseif use_fast_path
        nothing
    elseif use_obs_groups
        # Union of per-group fast `A'A` patterns; if any group fell back
        # to AD, also union in the AD-side likelihood pattern (auto-detected,
        # dense, or user-supplied).
        fast_pat = _union_group_fast_patterns(group_fast, n_tot)
        ad_pat = if !obs_groups_need_ad
            nothing
        elseif likelihood_hessian_pattern isa SparseMatrixCSC
            likelihood_hessian_pattern
        elseif likelihood_hessian_pattern === :dense
            dense_pattern()
        elseif likelihood_hessian_pattern === :auto
            try
                detect_likelihood_pattern(dppl_model, hp_names, n_tot)
            catch e
                @warn "Tracer-based sparsity detection failed; falling back to a dense likelihood Hessian pattern. Pass `likelihood_hessian_pattern = ...` to silence or override." exception = e
                dense_pattern()
            end
        else
            throw(ArgumentError("likelihood_hessian_pattern must be :auto, :dense, or a SparseMatrixCSC"))
        end
        _union_patterns(fast_pat, ad_pat)
    elseif likelihood_hessian_pattern isa SparseMatrixCSC
        likelihood_hessian_pattern
    elseif likelihood_hessian_pattern === :dense
        dense_pattern()
    elseif likelihood_hessian_pattern === :auto
        # Try structural tracing; if the likelihood's internals don't survive
        # SparseConnectivityTracer (common for black-box code paths like
        # OrdinaryDiffEq solvers), fall back to a dense pattern. Dense costs
        # a bit more per evaluation but is correct for any likelihood.
        try
            detect_likelihood_pattern(dppl_model, hp_names, n_tot)
        catch e
            @warn "Tracer-based sparsity detection failed; falling back to a dense likelihood Hessian pattern. Pass `likelihood_hessian_pattern = ...` to silence or override." exception = e
            dense_pattern()
        end
    else
        throw(ArgumentError("likelihood_hessian_pattern must be :auto, :dense, or a SparseMatrixCSC"))
    end

    # Latent prior: either the macro-recognized prebuilt latent, or the
    # default DAG / sparse-AD extraction from the DPPL graph. The pattern-
    # augmentation inside `build_latent_model` is redundant here — we've
    # already resolved `extra_pattern` above — so tell it not to re-detect.
    latent = if latent_override === nothing
        l, path = build_latent_model(
            dppl_model, random_syms, hp_names;
            skip_pattern_augment = true,
            extra_pattern = extra_pattern,
        )
        @debug "latent extraction path" path fast_path = use_fast_path augmented = augment
        l
    elseif extra_pattern === nothing
        latent_override
    else
        # The DAG path bakes `extra_pattern` into its precision; a recognized
        # override doesn't, so its precision pattern can be too sparse for a
        # likelihood that couples latents (e.g. `dot(A, β)`). Wrap it to union
        # the pattern in (structural zeros only).
        _PatternAugmentedLatentModel(latent_override, extra_pattern)
    end

    obs = if use_fast_path
        fast_obs
    elseif use_obs_groups
        # Per-group routing: fast-path components stay fast; AD-fallback
        # components use the lifted body when `lift_spec !== nothing`,
        # else the legacy DPPL AD closure. The composite wrapper handles
        # the prelude precomputation at materialisation time.
        _build_obs_groups_observation_model(
            dppl_model, obs_groups_spec, hp_names, length(latent),
            random_syms, dims, extra_pattern;
            fast_results = group_fast, lift_spec = lift_spec,
        )
    elseif lift_spec !== nothing
        _build_single_lifted_obs_model(
            dppl_model, length(latent), random_syms, dims;
            hp_names = hp_names, hessian_pattern = extra_pattern,
            lift_spec = lift_spec,
        )
    else
        extract_obs_model(
            dppl_model, length(latent), random_syms, dims;
            hp_names = hp_names,
            hessian_pattern = extra_pattern,  # nothing, dense, or user-supplied
        )
    end
    # Build a `sym → augmented-latent range` layout so downstream callers
    # (`linear_combinations(result; β = …)`, `result.latent_marginals[:β]`,
    # etc.) can look things up by DPPL symbol. When the LGM is augmented
    # the η positions prepend the base latent, so each range is offset by
    # `n_obs`; for non-augmented LGMs the ranges start at 1.
    augmented = (obs isa LinearlyTransformedObservationModel) && augment
    n_obs = augmented ? size(_extract_design_matrix(obs), 1) : 0
    layout = _build_latent_layout(random_syms, dims, n_obs)

    # Only the LTM-specialised LGM constructor takes `augment_latent=` —
    # for the AD fallback (AutoDiffObservationModel), there's no
    # auto-augmentation machinery to opt into, just the base constructor.
    if obs isa LinearlyTransformedObservationModel
        return LatentGaussianModel(
            spec, latent, obs;
            augment_latent = augment, latent_layout = layout,
        )
    else
        return LatentGaussianModel(
            spec, latent, obs, nothing;
            latent_layout = layout,
        )
    end
end

function _build_latent_layout(
        random_syms::Tuple, dims::Dict{Symbol, Int}, n_obs::Int,
    )
    layout = OrderedDict{Symbol, UnitRange{Int}}()
    off = n_obs
    for s in random_syms
        layout[s] = (off + 1):(off + dims[s])
        off += dims[s]
    end
    return layout
end

# Extract the design matrix A from whatever the fast path produced:
#   LTM(ExponentialFamily, A; offset = b)     → A
_extract_design_matrix(ltm::LinearlyTransformedObservationModel) = ltm.design_matrix

# Boolean pattern of A'A — the sparsity of the likelihood Hessian w.r.t.
# x when the linear predictor is A·x. Used to pre-populate Q's pattern
# for non-augmented fast-path LGMs.
function _bool_AtA_pattern(A::AbstractMatrix)
    pat = SparseMatrixCSC{Bool, Int}(A .!= 0)
    return pat' * pat   # boolean matrix product → union of column patterns
end

# Union the per-group fast `A'A` patterns from a `group_fast` dict,
# skipping groups that fell through to AD. Returns `nothing` when no
# group succeeded — caller treats that as "fast contributes nothing".
function _union_group_fast_patterns(group_fast::AbstractDict, n_tot::Int)
    fasts = [r for r in values(group_fast) if r !== nothing]
    isempty(fasts) && return nothing
    pat = SparseMatrixCSC{Bool, Int}(spzeros(Bool, n_tot, n_tot))
    for r in fasts
        pat = pat .| r.pattern
    end
    return pat
end

# `_union_patterns` is defined in `dsl/latent_prior.jl` and already handles
# the `nothing` × pattern combinations we need here.
