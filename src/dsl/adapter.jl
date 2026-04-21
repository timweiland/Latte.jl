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
function latte_from_dppl(dppl_model; random::Union{Symbol, Tuple})
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
    latent, path = build_latent_model(dppl_model, random_syms, hp_names)
    @debug "latent extraction path" path

    # Probe dims for the obs model so it can slice x into (β, u, ...) pieces.
    probe_hp = NamedTuple{hp_names}(Tuple(1.0 for _ in hp_names))
    dims = Dict(s => variable_length(dppl_model, s, probe_hp) for s in random_syms)

    obs = extract_obs_model(
        dppl_model, length(latent), random_syms, dims;
        hp_names = hp_names,
    )
    return LatentGaussianModel(spec, latent, obs)
end
