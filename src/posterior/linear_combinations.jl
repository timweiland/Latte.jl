using LinearAlgebra: dot
import Distributions

export linear_combinations, lincomb_variance

"""
    lincomb_variance(q, a) -> Real

Posterior variance of the linear functional `aᵀx`, i.e. `aᵀ Σ a = aᵀ Q⁻¹ a`,
via a single factor solve. The third covariance primitive of Latte's
posterior-query interface (see [`selected_covariance`](@ref),
[`conditional_column`](@ref)).

!!! warning "Constraint correction gap (pre-existing)"
    The GMRF default solves against the *base* factor and does **not** apply the
    constraint Woodbury correction that `selected_covariance` / `conditional_column`
    do. This reproduces the historical `linear_combinations` behavior exactly and
    is a known correctness gap for constrained models — it is preserved
    deliberately here and must not be "fixed" inline without sign-off, since
    doing so would change results for constrained models.
"""
function lincomb_variance(q::AbstractGMRF, a::AbstractVector)
    base = q isa ConstrainedGMRF ? q.base_gmrf : q
    GaussianMarkovRandomFields.ensure_loaded!(base)
    return dot(a, GaussianMarkovRandomFields.workspace_solve(base.workspace, collect(a)))
end

# Dense fallback for non-GMRF posteriors (see `selected_covariance`). `AbstractGMRF`
# is more specific, so GMRF posteriors keep the factor-solve method above.
lincomb_variance(q::Distributions.AbstractMvNormal, a::AbstractVector) =
    dot(a, Distributions.cov(q) * a)

"""
    _reconstruct_ga(model, y_obs, θ, ws) -> (ga, prior_gmrf, obs_lik, θ_natural_nt)

Reconstruct the Gaussian approximation at hyperparameter configuration `θ`.

Converts θ to natural space, builds the prior GMRF and observation likelihood,
and returns the Gaussian approximation along with the prior GMRF, the materialized
observation likelihood (both needed for a VBC mean correction), and the
natural-space NamedTuple. Used by `linear_combinations` and `rand(::INLAResult)`.
"""
function _reconstruct_ga(model, y_obs, θ, ws)
    θ_natural = convert(NaturalHyperparameters, θ)
    θ_natural_nt = convert(NamedTuple, θ_natural)
    obs_lik = model.observation_model(y_obs; θ_natural_nt...)
    if model.latent_prior isa NonGaussianLatentPrior
        # No fixed GMRF to materialise; θ is a concrete integration point (primal), so the
        # workspace is safe. `prior_gmrf` is `nothing` — only the VBC mean correction reads
        # it, and that is gated off for non-Gaussian priors (the GaussianMarginal default
        # leaves the latent mean uncorrected, so the downstream `x_shift` is zero).
        ga = gaussian_approximation(model.latent_prior, obs_lik; θ = θ_natural_nt, ws = ws)
        return ga, nothing, obs_lik, θ_natural_nt
    end
    prior_gmrf = latent_gmrf(model, ws, θ_natural_nt)
    return gaussian_approximation(prior_gmrf, obs_lik), prior_gmrf, obs_lik, θ_natural_nt
end

"""
    _integration_weights(exploration::HyperparameterExploration) -> Vector{Float64}

Compute normalized integration weights from exploration results with log-sum-exp stabilization.
"""
function _integration_weights(exploration)
    integration_points = exploration.grid_points[exploration.integration_indices]
    log_weights = [p.log_density for p in integration_points]
    weights = exp.(log_weights .- maximum(log_weights))
    weights ./= sum(weights)
    return weights
end

"""
    linear_combinations(result::INLAResult, A::AbstractMatrix) -> Vector{WeightedMixture}
    linear_combinations(result::INLAResult, a::AbstractVector) -> WeightedMixture

Compute posterior marginals for linear combinations z = A * x of the latent field.

For each row aₖ of A, the marginal of zₖ = aₖᵀx is computed as a weighted mixture
of Gaussian conditionals over the hyperparameter integration points. This treats
each linear combination as a derived quantity: rather than augmenting the latent
field with the combinations and refitting, the marginal is assembled directly from
the per-θ Gaussian approximations already computed during inference.

At each integration point θⱼ with weight wⱼ, the conditional distribution of
zₖ = aₖᵀx is Normal(aₖᵀμⱼ, √(aₖᵀQⱼ⁻¹aₖ)), where μⱼ and Qⱼ are the mean
and precision matrix of the Gaussian approximation at θⱼ.

# Arguments
- `result::INLAResult`: Results from `inla()` inference
- `A::AbstractMatrix`: m×n matrix where each row defines a linear combination
- `a::AbstractVector`: Length-n vector defining a single linear combination

# Returns
- `Vector{WeightedMixture}`: One marginal per row of A (matrix form)
- `WeightedMixture`: Single marginal (vector form)

# Example
```julia
result = inla(model, y; progress=false)
n = length(result.latent_marginals)

# Contrast between first two latent variables
a = zeros(n); a[1] = 1; a[2] = -1
contrast_marginal = linear_combinations(result, a)
mean(contrast_marginal)  # posterior mean of x₁ - x₂

# Sum of all latent variables
sum_marginal = linear_combinations(result, ones(1, n))

# Multiple linear combinations at once
A = [1.0 -1.0 zeros(1, n-2)...;
     zeros(1, n-1)... 1.0]
marginals = linear_combinations(result, A)
```

# Notes
- For single-variable queries, prefer `result.latent_marginals[i]` which includes
  Laplace corrections. This function uses Gaussian conditionals, justified by CLT
  for multi-variable combinations.
- Sparse matrices are supported and recommended for large, sparse A.
"""
function linear_combinations(result::INLAResult, A::AbstractMatrix)
    exploration = result.exploration
    model = result.model
    y_obs = _get_y_obs(result)

    m = size(A, 1) # number of linear combinations
    n = size(A, 2) # latent field dimension

    n_latent = length(result.latent_marginals)
    if n != n_latent
        throw(
            DimensionMismatch(
                "A has $n columns but the latent field has $n_latent variables"
            )
        )
    end

    # Get integration points and weights
    weights = _integration_weights(exploration)
    integration_points = exploration.grid_points[exploration.integration_indices]
    n_points = length(integration_points)

    # One-time symbolic factorization, reused for every integration point.
    θ_ref_nt = convert(NamedTuple, convert(NaturalHyperparameters, integration_points[1].θ))
    ws = make_workspace(model.latent_prior; θ_ref_nt...)

    # For each linear combination, collect Normal components across integration points
    components = [Vector{Normal{Float64}}(undef, n_points) for _ in 1:m]

    method = result.options.latent_marginalization_method
    for (j, point) in enumerate(integration_points)
        ga, prior_gmrf, obs_lik, _ = _reconstruct_ga(model, y_obs, point.θ, ws)
        # μ* under VBC (so lincomb means match the corrected latent marginals),
        # else the GA mode. Variance is always the GA's (VBC leaves it untouched).
        μ = _corrected_latent_mean(method, ga, obs_lik, prior_gmrf, model)

        # Posterior variance of each linear functional aₖᵀx via a factor solve.
        for k in 1:m
            a_k = view(A, k, :)
            z_mean = dot(a_k, μ)
            z_var = lincomb_variance(ga, a_k)
            components[k][j] = Normal(z_mean, sqrt(max(z_var, 0.0)))
        end
    end

    return [WeightedMixture(components[k], weights) for k in 1:m]
end

function linear_combinations(result::INLAResult, a::AbstractVector)
    A = reshape(a, 1, :)
    return linear_combinations(result, A)[1]
end

"""
    linear_combinations(result::INLAResult; sym1 = coef1, sym2 = coef2, …)

Named form of `linear_combinations` for DPPL-built LGMs. Each `sym` must
appear in `latent_groups(result)`, and `coef` gives the coefficients for
that block — as a matrix of shape `(m, dim(sym))` or `(m,)` for a 1-dim
block, or as a scalar (broadcast to a ones-column).

Missing symbols (including the augmented η positions) get zero
coefficients. Equivalent to building the padded design matrix by hand
and passing it to the matrix form.

# Example
```julia
lgm = latte_from_dppl(model; random = (:β, :field))
result = inla(lgm, y)

# Predict β + A_pred · field at new locations:
preds = linear_combinations(result; β = 1.0, field = A_pred)
```
"""
function linear_combinations(result::INLAResult; kwargs...)
    layout = latent_groups(result)
    isempty(layout) && throw(
        ArgumentError(
            "Named linear_combinations requires a DPPL-built LGM (populated latent layout). " *
                "For hand-built LGMs, build the design matrix yourself and use " *
                "linear_combinations(result, A)."
        )
    )
    for k in keys(kwargs)
        haskey(layout, k) || throw(
            ArgumentError(
                "Unknown latent symbol `:$k`. Known: $(collect(keys(layout))). " *
                    "η positions in the augmented latent are reached by leaving out " *
                    "their column (they're filled with zeros by default)."
            )
        )
    end

    n_lat = length(latent_marginals(result))
    m = _infer_rows(values(kwargs))
    A = spzeros(m, n_lat)
    for (name, coef) in pairs(kwargs)
        rng = layout[name]
        A[:, rng] = _normalize_coef(coef, m, length(rng))
    end
    return linear_combinations(result, A)
end

_infer_rows(iter) = begin
    for c in iter
        c isa Number && continue
        return size(c, 1)
    end
    throw(
        ArgumentError(
            "Cannot infer output row count — pass at least one matrix- or " *
                "vector-valued coefficient (scalars alone are ambiguous)."
        )
    )
end

function _normalize_coef(coef::Number, m::Int, dim::Int)
    dim == 1 ||
        throw(ArgumentError("Scalar coefficient only works for 1-dim blocks; got dim=$dim"))
    return fill(float(coef), m, 1)
end
function _normalize_coef(coef::AbstractVector, m::Int, dim::Int)
    length(coef) == m || throw(
        DimensionMismatch("Vector coefficient has length $(length(coef)) but expected m=$m")
    )
    dim == 1 || throw(
        ArgumentError("Vector coefficient only works for 1-dim blocks; got dim=$dim")
    )
    return reshape(coef, m, 1)
end
function _normalize_coef(coef::AbstractMatrix, m::Int, dim::Int)
    size(coef) == (m, dim) || throw(
        DimensionMismatch(
            "Matrix coefficient has size $(size(coef)) but expected ($m, $dim)"
        )
    )
    return coef
end
