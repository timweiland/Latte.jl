using Distributions: Normal
using OrderedCollections: OrderedDict

export TMBResult

"""
    TMBResult <: InferenceResult

Result of TMB-style inference via `tmb(model, y)`. Contains the hyperparameter
MAP with a Gaussian covariance from the Hessian of the negative log posterior
at the mode, plus the inner Laplace random-effect posterior at that MAP.

# Fields
- `hyperparameter_marginals::Vector{Normal}` — working-space Gaussian marginals
  at the MAP (derived from `θ_map` and `sqrt.(diag(θ_cov))`).
- `latent_marginals::Vector{Normal}` — natural-space Gaussian marginals from
  the inner Laplace approximation at the MAP.
- `θ_map::Vector{Float64}` — MAP in working space.
- `θ_cov::Matrix{Float64}` — working-space covariance (≈ `inv(-H)` of the log
  posterior at the MAP).
- `x_mean::Vector{Float64}` / `x_std::Vector{Float64}` — inner-Laplace posterior
  mean and marginal standard deviations.
- `log_marginal_likelihood::Float64` — Laplace approximation to `log p(y)` at
  the MAP.
- `model::LatentGaussianModel` — the fitted model.
- `observations::AbstractVector` — the observations used.
- `converged::Bool` / `time_elapsed::Float64` — diagnostics.

TMB-flavour aliases are exposed: `fixed_effects`, `random_effects`, `fixef`,
`ranef` (see `src/LAYOUT.md`, Terminology section).
"""
struct TMBResult{Hp, Lm, M, Y} <: InferenceResult
    hyperparameter_marginals::Hp        # Vector{Normal}
    latent_marginals::Lm                # Vector{Normal}
    θ_map::Vector{Float64}
    θ_cov::Matrix{Float64}
    x_mean::Vector{Float64}
    x_std::Vector{Float64}
    log_marginal_likelihood::Float64
    model::M
    observations::Y
    converged::Bool
    time_elapsed::Float64
end

# ─── Protocol implementations ──────────────────────────────────────────────
latent_marginals(r::TMBResult) = r.latent_marginals
hyperparameter_marginals(r::TMBResult) = r.hyperparameter_marginals

latent_groups(::TMBResult) = OrderedDict{Symbol, UnitRange{Int}}()

function hyperparameter_groups(r::TMBResult)
    names = collect(keys(r.model.hyperparameter_spec.free))
    groups = OrderedDict{Symbol, UnitRange{Int}}()
    for (i, name) in enumerate(names)
        groups[name] = i:i
    end
    return groups
end

function hyperparameter_mode(r::TMBResult)
    wh = WorkingHyperparameters(r.θ_map, r.model.hyperparameter_spec)
    return convert(NaturalHyperparameters, wh)
end

model(r::TMBResult) = r.model
observations(r::TMBResult) = r.observations
converged(r::TMBResult) = r.converged
time_elapsed(r::TMBResult) = r.time_elapsed
log_marginal_likelihood(r::TMBResult) = r.log_marginal_likelihood

# ─── TMB-style aliases (TMB / MixedModels vocabulary) ──────────────────────
export fixed_effects, random_effects, fixef, ranef

"""
    fixed_effects(r::TMBResult)

Alias for `hyperparameter_marginals(r)`. Convenience for TMB / MixedModels users;
see the Terminology section of `src/LAYOUT.md` for why the core protocol uses
"hyperparameters" / "latent" instead.
"""
fixed_effects(r::TMBResult) = hyperparameter_marginals(r)
fixef(r::TMBResult) = hyperparameter_marginals(r)

"""
    random_effects(r::TMBResult)

Alias for `latent_marginals(r)`.
"""
random_effects(r::TMBResult) = latent_marginals(r)
ranef(r::TMBResult) = latent_marginals(r)

# ─── Pretty printing ───────────────────────────────────────────────────────
function Base.show(io::IO, r::TMBResult)
    spec = r.model.hyperparameter_spec
    names = collect(keys(spec.free))
    println(io, "TMBResult:")
    println(io, "  Model: ", typeof(r.model))
    println(io, "  Hyperparameters (working-space MAP ± SE):")
    θ_se = sqrt.(max.(diag(r.θ_cov), 0.0))
    for i in eachindex(names)
        @printf(io, "    %-8s %+8.4f ± %.4f\n", String(names[i]), r.θ_map[i], θ_se[i])
    end
    println(io, "  Latent dimension: ", length(r.x_mean))
    println(
        io, "  log p(y) ≈ ", @sprintf("%.4f", r.log_marginal_likelihood),
        " (Laplace at MAP)"
    )
    return println(io, "  Time: ", @sprintf("%.2f", r.time_elapsed), " s")
end
