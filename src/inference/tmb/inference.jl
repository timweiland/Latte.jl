using LinearAlgebra: Symmetric, diag
using Distributions: Normal, MvNormal
using Random
import FiniteDiff

export tmb

"""
    tmb(model::LatentGaussianModel, y; diff_strategy=FiniteDiffStrategy()) -> TMBResult

TMB-style inference: finds the hyperparameter MAP, computes its Gaussian
covariance from the Hessian of the negative log posterior, and attaches the
inner Laplace approximation for the latent field at that MAP.

Output matches TMB's `sdreport` shape: MAP θ̂ with standard errors, inner
Laplace random-effect posterior means and marginal standard deviations, and a
Laplace estimate of `log p(y)`.

The `diff_strategy` argument is forwarded to `find_hyperparameter_mode`. The
outer Hessian (for θ's covariance) is always computed by finite differences of
the objective.
"""
function tmb(
        model::LatentGaussianModel, y;
        diff_strategy = FiniteDiffStrategy()
    )
    t_start = time()

    # Wrap / normalize y the same way inla() does (Vector{Int} → PoissonObservations
    # etc., plus missing-value handling). For TMB we don't yet support prediction
    # via missing values — passing `missing`s would error downstream.
    y_obs, model, _ = _prepare_for_prediction(model, y)
    spec = model.hyperparameter_spec

    # ─── Step 1: MAP of the hyperparameters ─────────────────────────────
    θ̂_wh, _, mode_info = find_hyperparameter_mode(
        model, y_obs; diff_strategy = diff_strategy
    )
    θ̂ = θ̂_wh.θ
    is_converged = hasproperty(mode_info, :converged) ? mode_info.converged : true

    # ─── Step 2: θ posterior covariance from Hessian at the MAP ─────────
    names = collect(keys(spec.free))
    sentinel_hp = NamedTuple{Tuple(names)}(Tuple(1.0 for _ in names))
    ws = make_workspace(model.latent_prior; sentinel_hp...)
    objective(θ_vec) = -hyperparameter_logpdf(
        model, WorkingHyperparameters(θ_vec, spec), y_obs; ws = ws
    )
    H_θ = FiniteDiff.finite_difference_hessian(objective, θ̂)
    Σ_θ = Matrix(inv(Symmetric(H_θ)))
    θ_se = sqrt.(max.(diag(Σ_θ), 0.0))

    # ─── Step 3: log p(y) via Laplace at the MAP ────────────────────────
    logL = -objective(θ̂)

    # ─── Step 4: inner Laplace — latent posterior at the MAP ────────────
    θ̂_natural_nt = convert(
        NamedTuple, convert(NaturalHyperparameters, θ̂_wh)
    )
    prior_gmrf = model.latent_prior(; θ̂_natural_nt...)
    obs_lik = model.observation_model(y_obs; θ̂_natural_nt...)
    x_post = gaussian_approximation(prior_gmrf, obs_lik)

    x_mean = Vector(mean(x_post))
    Σ_x = selinv_mat(x_post)
    x_std = Vector{Float64}(sqrt.(max.(diag(Σ_x), 0.0)))

    # ─── Marginal Distribution objects (the protocol shape) ─────────────
    hp_marg = [Normal(θ̂[i], θ_se[i]) for i in eachindex(θ̂)]
    latent_marg = [Normal(x_mean[i], x_std[i]) for i in eachindex(x_mean)]

    return TMBResult(
        hp_marg, latent_marg,
        θ̂, Σ_θ, x_mean, x_std,
        logL, model, y_obs,
        is_converged, time() - t_start,
    )
end

# ─── rand(r::TMBResult, n) via the joint Gaussian at the MAP ───────────
function Random.rand(rng::AbstractRNG, r::TMBResult, n::Int; include_y::Bool = false)
    # θ samples from N(θ_map, θ_cov)
    θ_dist = MvNormal(r.θ_map, Symmetric(r.θ_cov))
    θ_mat = Matrix(rand(rng, θ_dist, n)')   # (n × n_hp)

    # x samples from the inner Laplace at the MAP. Per-θ reconstruction would
    # be more accurate but expensive; for MVP treat θ and x as independent at
    # the MAP — a standard TMB-style approximation for quick uncertainty
    # propagation. Full joint sampling is a follow-up.
    θ̂_wh = WorkingHyperparameters(r.θ_map, r.model.hyperparameter_spec)
    θ̂_natural_nt = convert(NamedTuple, convert(NaturalHyperparameters, θ̂_wh))
    prior_gmrf = r.model.latent_prior(; θ̂_natural_nt...)
    obs_lik = r.model.observation_model(r.observations; θ̂_natural_nt...)
    x_post = gaussian_approximation(prior_gmrf, obs_lik)

    n_x = length(r.x_mean)
    x_mat = Matrix{Float64}(undef, n, n_x)
    y_mat = nothing
    for i in 1:n
        x_sample = rand(rng, x_post)
        x_mat[i, :] = x_sample
        if include_y
            y_dist = GaussianMarkovRandomFields.conditional_distribution(
                r.model.observation_model, x_sample; θ̂_natural_nt...
            )
            y_sample = rand(rng, y_dist)
            if y_mat === nothing
                y_mat = Matrix{eltype(y_sample)}(undef, n, length(y_sample))
            end
            y_mat[i, :] = y_sample
        end
    end

    return PosteriorSamples(θ_mat, x_mat; y = y_mat)
end

function Random.rand(rng::AbstractRNG, r::TMBResult; include_y::Bool = false)
    return rand(rng, r, 1; include_y = include_y)[1]
end

Random.rand(r::TMBResult, n::Int; kwargs...) =
    rand(Random.default_rng(), r, n; kwargs...)
Random.rand(r::TMBResult; kwargs...) =
    rand(Random.default_rng(), r; kwargs...)
