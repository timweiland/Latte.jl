# Build a `HyperparameterSpec` from a DPPL model by reading the prior on each
# hyperparameter and asking `Bijectors` for the working-space transform.

using Bijectors: bijector

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
