export warmup

"""
    warmup(model_or_factory, y; random, adapter_kwargs=(;), inla_kwargs...) -> nothing

Run a single end-to-end `inla(latte_from_dppl(model), y)` cycle to force
type inference and native-code specialisation on the user's exact model
type. Use this when the in-package `@compile_workload` (which only covers
the post-DPPL pipeline for canonical fast-path likelihoods) does not
cover your specific model — typically because your DPPL `@model` has
non-standard hyperparameter shapes or observation likelihoods.

# Usage

For interactive sessions: just call `warmup(model, y)` once on a tiny
representative dataset; subsequent `inla()` calls will use the cached
specialisation.

```julia
using Latte, DynamicPPL, Distributions, GaussianMarkovRandomFields

@model function my_glmm(y, X, group)
    τ ~ Gamma(2, 1)
    β ~ MvNormal(zeros(size(X, 2)), 100.0 * I)
    u ~ IIDModel(maximum(group))(τ = τ)
    for i in eachindex(y)
        y[i] ~ Poisson(exp(X[i, :]' * β + u[group[i]]))
    end
end

# One-time warmup with a tiny slice — same shape, faster.
Latte.warmup(my_glmm(y[1:8], X[1:8, :], group[1:8]), y[1:8]; random = (:β, :u))

# Subsequent calls hit cached code:
result = inla(latte_from_dppl(my_glmm(y, X, group); random = (:β, :u)), y)
```

For application packages that ship around Latte: put a `warmup` call
inside your own `@compile_workload` so the specialisation lands in your
package's precompile cache. Example, in `MyApp/src/precompile.jl`:

```julia
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    y_tiny, X_tiny, group_tiny = _build_tiny_workload_dataset()
    @compile_workload begin
        Latte.warmup(my_glmm(y_tiny, X_tiny, group_tiny), y_tiny; random = (:β, :u))
    end
end
```

To exercise non-default adapter knobs (`augment`, `force_ad_obs_model`,
`likelihood_hessian_pattern`), pass them via `adapter_kwargs`:

```julia
Latte.warmup(model, y; random = (:β, :u),
             adapter_kwargs = (; augment = false, force_ad_obs_model = true))
```

# Arguments
- `model_or_factory`: A `DynamicPPL` model instance (output of `mymodel(args...)`)
  or any object that `latte_from_dppl` accepts.
- `y::AbstractVector`: Observed data — should match the shape passed when
  building `model_or_factory`.

# Keyword arguments
- `random`: forwarded to `latte_from_dppl`. Tuple (or single Symbol) of
  latent symbols in the DPPL model.
- `adapter_kwargs::NamedTuple = (;)`: extra kwargs forwarded to
  `latte_from_dppl` alongside `random` (e.g. `augment`,
  `force_ad_obs_model`, `likelihood_hessian_pattern`).
- All remaining kwargs are forwarded to `inla` (e.g. `latent_marginalization_method`,
  `exploration_strategy`, `accumulators`). `progress = false` is set
  unconditionally since this is a precompile pass.

# Returns
`nothing` — the result is discarded; the only effect is JIT specialisation.

# See also
- The package-level `@compile_workload` in `src/precompile_workloads.jl`,
  which covers Poisson/Bernoulli/Binomial/Normal IID models without
  a DPPL adapter pass — those are precompiled for everyone via Latte's
  own precompile cache.
"""
function warmup(
        model_or_factory, y::AbstractVector;
        random,
        adapter_kwargs::NamedTuple = NamedTuple(),
        inla_kwargs...
    )
    lgm = latte_from_dppl(model_or_factory; random = random, adapter_kwargs...)
    inla(
        lgm, y;
        progress = false,
        inla_kwargs...,
    )
    return nothing
end
