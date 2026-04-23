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

    # Detect fast path first.
    fast_obs = force_ad_obs_model ? nothing :
        try_exponential_family_fast_path(dppl_model, random_syms, dims, hp_names)
    use_fast_path = fast_obs !== nothing

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

    # The pattern-augmentation inside `build_latent_model` is now redundant:
    # we've already resolved `extra_pattern` above. Tell it not to re-detect.
    latent, path = build_latent_model(
        dppl_model, random_syms, hp_names;
        skip_pattern_augment = true,
        extra_pattern = extra_pattern,
    )
    @debug "latent extraction path" path fast_path = use_fast_path augmented = augment

    obs = use_fast_path ? fast_obs :
        extract_obs_model(
            dppl_model, length(latent), random_syms, dims;
            hp_names = hp_names,
            hessian_pattern = extra_pattern,  # nothing, dense, or user-supplied
        )
    # Only the LTM-specialised LGM constructor takes `augment_latent=` —
    # for the AD fallback (AutoDiffObservationModel), there's no
    # auto-augmentation machinery to opt into, just the base constructor.
    if obs isa LinearlyTransformedObservationModel
        return LatentGaussianModel(spec, latent, obs; augment_latent = augment)
    else
        return LatentGaussianModel(spec, latent, obs)
    end
end

# Extract the design matrix A from whatever the fast path produced:
#   LTM(ExponentialFamily, A)                 → A
#   LTM(OffsetObservationModel(ExpFam), A)    → A
_extract_design_matrix(ltm::LinearlyTransformedObservationModel) = ltm.design_matrix

# Boolean pattern of A'A — the sparsity of the likelihood Hessian w.r.t.
# x when the linear predictor is A·x. Used to pre-populate Q's pattern
# for non-augmented fast-path LGMs.
function _bool_AtA_pattern(A::AbstractMatrix)
    pat = SparseMatrixCSC{Bool, Int}(A .!= 0)
    return pat' * pat   # boolean matrix product → union of column patterns
end
