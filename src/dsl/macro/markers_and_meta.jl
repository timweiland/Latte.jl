# `@latte` — high-level adapter macro for Latte.jl.
#
# Wraps a DPPL `@model` with static AST analysis that auto-detects
# observation grouping by hyperparameter dependency, plus user-driven
# `@random` / `@fixed` markers for explicit control over which sample sites
# are random effects (Laplace-marginalised) vs hyperparameters / fixed
# effects (grid- or MAP-handled by the inference method).
#
# Defaults (when no marker present):
# - LHS is a positional argument of the `@model` function ⇒ observation.
# - LHS is a fresh symbol AND RHS callee is a known random-effect-shaped
#   constructor (`MvNormal`, `IIDModel`, `RWModel`, `BesagModel`,
#   `MaternModel`, `BYM2Model`, `SeparableModel`, `GMRF`,
#   `ConstrainedGMRF`) ⇒ random effect.
# - Anything else ⇒ fixed effect (hyperparameter).
#
# `@random` / `@fixed` markers override the default for a single `~`.
# Outside `@latte` the markers are no-op identity macros so the body still
# compiles inside a plain `@model`.
#
# Turing handoff: every `@latte`-defined model also gets an underlying
# DPPL `@model`-built constructor, accessible via `Latte.dppl_model(name)`.
# Same body, no markers — Turing's `sample(NUTS(), ...)` works directly.

export @latte, @random, @fixed

# ─── Marker macros (no-op outside @latte) ─────────────────────────────────────
"""
    @random a ~ b
    @fixed  a ~ b

Marker macros for use inside `@latte` model bodies. Override the default
classification of a `~` block:

- `@random` marks the block as a random effect (Laplace-marginalised latent).
- `@fixed` marks the block as a fixed effect (hyperparameter).

Outside `@latte` they're identity passthroughs, so a `@latte` model body can
also be sent through plain `DynamicPPL.@model` for Turing handoff with the
same syntax.
"""
macro random(ex)
    return esc(ex)
end

"""
    @fixed

Mark a `~` site in an `@latte` model body as a fixed effect / hyperparameter, overriding the
default site classification. Outside `@latte` it is an identity passthrough; see [`@random`](@ref).
"""
macro fixed(ex)
    return esc(ex)
end

# ─── Side-channel storage ────────────────────────────────────────────────────
const _LATTE_METADATA = IdDict{Any, NamedTuple}()
const _LATTE_DPPL_CONSTRUCTORS = IdDict{Any, Any}()

"""
    Latte.dppl_model(latte_fun) -> DPPL model constructor

Return the underlying `DynamicPPL.@model`-built constructor for a function
defined with `@latte`. Useful for Turing handoff:

```julia
@latte function my_model(y, X)
    σ ~ Gamma(2, 1)
    β ~ MvNormal(zeros(size(X, 2)), 100*I)
    for i in eachindex(y)
        y[i] ~ Normal(dot(X[i, :], β), σ)
    end
end

# Latte path
lgm = my_model(y, X)
inla(lgm, y)

# Turing path (same definition):
turing_model = Latte.dppl_model(my_model)(y, X)
sample(turing_model, NUTS(), 1000)
```
"""
function dppl_model(f)
    haskey(_LATTE_DPPL_CONSTRUCTORS, f) || throw(
        ArgumentError(
            "$(f) was not defined with @latte; no underlying DPPL model registered"
        )
    )
    return _LATTE_DPPL_CONSTRUCTORS[f]
end

"""
    Latte.latte_analysis(latte_fun_or_lgm) -> NamedTuple

Static metadata captured at `@latte` macro time: hyperparameter names,
random-effect names, per-`~`-block records (lhs symbol, hp dependencies,
classification, dotted), and pre-computed observation groups.
"""
function latte_analysis(f)
    haskey(_LATTE_METADATA, f) || throw(
        ArgumentError(
            "$(f) was not defined with @latte; no analysis metadata registered"
        )
    )
    return _LATTE_METADATA[f]
end
