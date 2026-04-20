using Distributions
using Bijectors
using Printf

export HyperparameterSpec

using Bijectors: elementwise
export elementwise

"""
    HyperparameterSpec{Free, Fixed}

Complete specification of hyperparameters with both free and fixed parameters.

# Type Parameters
- `Free`: Concrete NamedTuple type for free parameters
- `Fixed`: Concrete NamedTuple type for fixed parameter values

# Fields
- `free::Free`: Free parameters to be estimated (NamedTuple of Hyperparameter objects)
- `fixed::Fixed`: Fixed parameter values (NamedTuple of scalar values)

# Example
```julia
using Bijectors

spec = HyperparameterSpec(
    free = (
        σ = Hyperparameter(Exponential(1.0), transform=elementwise(log), prior_space=:natural),
        ρ = Hyperparameter(Beta(2, 2), transform=Bijectors.Logit(0.0, 1.0), prior_space=:natural)
    ),
    fixed = (μ = 0.0,)
)
```
"""
struct HyperparameterSpec{Free, Fixed}
    free::Free
    fixed::Fixed

    function HyperparameterSpec(; free::NamedTuple, fixed::NamedTuple = NamedTuple())
        # Validate: must have at least one free parameter
        if isempty(keys(free))
            error("INLA requires at least one free hyperparameter. All-fixed hyperparameter specs are not supported.")
        end

        # Validate: no overlap between free and fixed
        free_names = keys(free)
        fixed_names = keys(fixed)
        overlap = intersect(Set(free_names), Set(fixed_names))
        if !isempty(overlap)
            error("Parameters cannot be both free and fixed: $(collect(overlap))")
        end

        return new{typeof(free), typeof(fixed)}(free, fixed)
    end
end

"""
    Base.show(io::IO, spec::HyperparameterSpec)

Pretty printing for HyperparameterSpec objects.
"""
function Base.show(io::IO, spec::HyperparameterSpec)
    n_free = length(keys(spec.free))
    n_fixed = length(keys(spec.fixed))
    n_total = n_free + n_fixed

    println(io, "HyperparameterSpec with $n_total parameters:")

    # Show free parameters
    println(io, "  Free parameters ($n_free):")
    for (name, hp) in pairs(spec.free)
        transform_name = hp.transform === identity ? "identity" : string(typeof(hp.transform).name.name)
        space_indicator = prior_space(hp) == :natural ? "ⁿ" : "ʷ"
        println(io, "    $name ~ $(hp.prior) via $(transform_name)$(space_indicator)")
    end

    # Show fixed parameters
    if !isempty(spec.fixed)
        println(io, "  Fixed parameters ($n_fixed):")
        for (name, value) in pairs(spec.fixed)
            println(io, "    $name = $value")
        end
    end

    return nothing
end
