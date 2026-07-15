# Build a `HyperparameterSpec` from a DPPL model by reading the prior on each
# hyperparameter and asking `Bijectors` for the working-space transform.

using Bijectors: bijector
using Distributions: UnivariateDistribution

"""
    extract_hp_spec(dppl_model, hp_names::Tuple)

Construct a `HyperparameterSpec` whose free hyperparameters are the DPPL
priors on `hp_names`, each wrapped with the `Bijectors`-inferred transform to
working space. All hyperparameters are assumed to be in natural space in the
DPPL model.
"""
function extract_hp_spec(dppl_model, hp_names::Tuple)
    priors = extract_priors(dppl_model)
    by_sym = Dict(getsym(vn) => d for (vn, d) in pairs(priors))
    hp_nt = NamedTuple{hp_names}(
        Tuple(
            Hyperparameter(
                    by_sym[k];
                    transform = bijector(by_sym[k]),
                    prior_space = :natural,
                )
                for k in hp_names
        )
    )
    return HyperparameterSpec(free = hp_nt, fixed = (;))
end

"""
    _hp_probe_nt(dppl_model, hp_names::Tuple) -> NamedTuple

Natural-space probe values for the hyperparameters — `1.0` per scalar, a
vector of ones per vector-valued (multivariate-prior) hyperparameter — used
by the adapter-time probes that condition the DPPL model on concrete hp
values.
"""
function _hp_probe_nt(dppl_model, hp_names::Tuple)
    isempty(hp_names) && return NamedTuple()
    priors = extract_priors(dppl_model)
    by_sym = Dict(getsym(vn) => d for (vn, d) in pairs(priors))
    return NamedTuple{hp_names}(Tuple(_hp_probe_value(by_sym[k]) for k in hp_names))
end

_hp_probe_value(::UnivariateDistribution) = 1.0
_hp_probe_value(d) = ones(length(d))
