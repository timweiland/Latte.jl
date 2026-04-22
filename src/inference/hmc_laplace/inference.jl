using AdvancedHMC
using LogDensityProblems
using LinearAlgebra: Symmetric
using Random
import FiniteDiff

export hmc_laplace

# ─── LogDensityProblems target: θ ↦ hyperparameter_logpdf ──────────────────
struct HMCTarget{M, Y, WS, S}
    model::M
    y::Y
    ws::WS
    spec::S
    dim::Int
end

LogDensityProblems.capabilities(::Type{<:HMCTarget}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(t::HMCTarget) = t.dim

# NUTS sometimes proposes θ values where the inner Laplace's posterior
# precision is non-PD (CHOLMOD fails) or the latent prior is otherwise
# degenerate. Return -Inf on any failure so AdvancedHMC treats the step
# as a divergence rather than erroring out of the chain.
function LogDensityProblems.logdensity(t::HMCTarget, θ_vec)
    try
        return hyperparameter_logpdf(
            t.model, WorkingHyperparameters(θ_vec, t.spec), t.y; ws = t.ws,
        )
    catch
        return oftype(θ_vec[1], -Inf)
    end
end

function LogDensityProblems.logdensity_and_gradient(t::HMCTarget, θ_vec)
    f(θ) = try
        hyperparameter_logpdf(
            t.model, WorkingHyperparameters(θ, t.spec), t.y; ws = t.ws,
        )
    catch
        oftype(θ[1], -Inf)
    end
    v = f(θ_vec)
    if !isfinite(v)
        return v, zero(θ_vec)
    end
    g = FiniteDiff.finite_difference_gradient(f, θ_vec)
    return v, g
end

# ─── hmc_laplace entry point ──────────────────────────────────────────────
"""
    hmc_laplace(model::LatentGaussianModel, y;
                n_samples=500, n_warmup=200,
                rng=Random.default_rng(),
                progress=false) -> HMCLaplaceResult

tmbstan-style HMC on the Laplace marginal. Samples θ via NUTS with the
Laplace approximation `q(x | θ)` substituted for the true `p(x | y, θ)`;
the latent field is reconstructed per-sample from the inner Laplace at
each drawn θ.

Pipeline:
1. Run `tmb(model, y)` → MAP `θ̂`, Laplace covariance `Σ_θ̂`. Used as
   warm-start initial point and as the HMC mass matrix (dense metric
   `M⁻¹ = Σ_θ̂`). Preconditioning matches the target's local curvature,
   so NUTS tree depths stay low.
2. NUTS samples θ on `L(θ) = log p(y, θ)` via `hyperparameter_logpdf`.
   Gradients by finite differences (fine for typical small |θ|).
3. For each sample `θ_k`, recompute the inner Gaussian approximation
   `q(x | θ_k)` and extract conditional means + marginal SDs.
4. Assemble `HMCLaplaceResult` with Tier 1 protocol marginals built from
   chain samples.

# Arguments
- `model`: a `LatentGaussianModel` (e.g., from `latte_from_dppl` or a
  hand-built spec).
- `y`: observations. Same handling as `inla()` / `tmb()` —
  `_prepare_for_prediction` normalises integer vectors into
  `PoissonObservations` etc.
- `n_samples` / `n_warmup`: NUTS post-warmup and warmup steps.
- `rng`: seedable RNG for reproducibility.
- `progress`: pass through to AdvancedHMC.
- `diff_strategy`: forwarded to the TMB warm-start (mode + Σ_θ). Default
  `ADStrategy()` is noise-robust on augmented LGMs; pass
  `FiniteDiffStrategy()` for DPPL-adapter-built LGMs until the closure
  Dual-degradation bug is fixed.

# Diagnostics

Method-specific accessors on the returned `HMCLaplaceResult`:
`samples`, `divergences`, `mean_tree_depth`, `acceptance_rate`,
`mean_step_size`. Use these to judge convergence before trusting the
marginals.
"""
function hmc_laplace(
        model::LatentGaussianModel, y;
        n_samples::Int = 500,
        n_warmup::Int = 200,
        rng::AbstractRNG = Random.default_rng(),
        progress::Bool = false,
        diff_strategy = ADStrategy(),
    )
    t_start = time()

    # Normalize y (Vector{Int} → PoissonObservations, etc.)
    y_obs, model, _ = _prepare_for_prediction(model, y)
    spec = model.hyperparameter_spec
    names = collect(keys(spec.free))

    # ── Step 1: TMB warm-start ───────────────────────────────────────────
    tmb_r = tmb(model, y_obs; diff_strategy = diff_strategy)
    θ̂ = tmb_r.θ_map
    Σ_θ = tmb_r.θ_cov

    # ── Step 2: build LogDensityProblems target ──────────────────────────
    sentinel_hp = NamedTuple{Tuple(names)}(Tuple(1.0 for _ in names))
    ws = make_workspace(model.latent_prior; sentinel_hp...)
    target = HMCTarget(model, y_obs, ws, spec, length(θ̂))

    # ── Step 3: NUTS with Laplace-at-MAP preconditioner ──────────────────
    # DenseEuclideanMetric stores M⁻¹. Setting M⁻¹ = Σ_θ gives momentum
    # covariance M = Σ_θ⁻¹, which matches the target's Gaussian-at-MAP
    # curvature — low tree depth, few divergences.
    metric = DenseEuclideanMetric(Symmetric(Matrix(Σ_θ)))
    hamiltonian = Hamiltonian(metric, target)

    initial_ϵ = find_good_stepsize(rng, hamiltonian, θ̂)
    integrator = Leapfrog(initial_ϵ)
    kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
    adaptor = StepSizeAdaptor(0.8, integrator)   # metric fixed at Laplace's

    samples, stats = AdvancedHMC.sample(
        rng, hamiltonian, kernel, θ̂, n_samples, adaptor, n_warmup;
        progress = progress,
    )

    # ── Step 4: reconstruct inner Laplace at each sample ─────────────────
    n_latent = length(model.latent_prior)
    K = length(samples)
    θ_samples = Matrix{Float64}(undef, K, length(θ̂))
    x_cond_means = Matrix{Float64}(undef, K, n_latent)
    x_cond_stds = Matrix{Float64}(undef, K, n_latent)

    for (k, θ_vec) in enumerate(samples)
        θ_samples[k, :] = θ_vec
        θ_wh = WorkingHyperparameters(θ_vec, spec)
        θ_nt = convert(NamedTuple, convert(NaturalHyperparameters, θ_wh))
        prior = model.latent_prior(ws; θ_nt...)
        obs_lik = model.observation_model(y_obs; θ_nt...)
        x_post = gaussian_approximation(prior, obs_lik)
        x_cond_means[k, :] = Vector(mean(x_post))
        Σ_x = selinv_mat(x_post)
        x_cond_stds[k, :] = sqrt.(max.(diag(Σ_x), 0.0))
    end

    hp_marg, latent_marg = _build_hmc_marginals(θ_samples, x_cond_means, x_cond_stds, spec)

    return HMCLaplaceResult(
        hp_marg, latent_marg,
        θ_samples, x_cond_means, x_cond_stds,
        stats, θ̂, n_warmup,
        model, y_obs,
        time() - t_start,
    )
end

# ─── rand(r::HMCLaplaceResult, n) — bootstrap from the chain ──────────────
function Random.rand(rng::AbstractRNG, r::HMCLaplaceResult, n::Int; include_y::Bool = false)
    K = size(r.θ_samples, 1)
    idxs = rand(rng, 1:K, n)   # bootstrap

    θ_mat = r.θ_samples[idxs, :]
    n_x = size(r.x_cond_means, 2)
    x_mat = Matrix{Float64}(undef, n, n_x)
    y_mat = nothing

    # Reconstruct the Gaussian approximation on demand for each unique θ_k —
    # this gives a proper joint x-draw (capturing correlations), matching the
    # convention used in INLA's rand. Batch by unique index to amortize the
    # Laplace reconstruction cost.
    spec = r.model.hyperparameter_spec
    names = collect(keys(spec.free))
    sentinel_hp = NamedTuple{Tuple(names)}(Tuple(1.0 for _ in names))
    ws = make_workspace(r.model.latent_prior; sentinel_hp...)

    for k in unique(idxs)
        θ_wh = WorkingHyperparameters(r.θ_samples[k, :], spec)
        θ_nt = convert(NamedTuple, convert(NaturalHyperparameters, θ_wh))
        prior = r.model.latent_prior(ws; θ_nt...)
        obs_lik = r.model.observation_model(r.observations; θ_nt...)
        x_post = gaussian_approximation(prior, obs_lik)

        for i in findall(==(k), idxs)
            x_sample = rand(rng, x_post)
            x_mat[i, :] = x_sample
            if include_y
                y_dist = GaussianMarkovRandomFields.conditional_distribution(
                    r.model.observation_model, x_sample; θ_nt...
                )
                y_sample = rand(rng, y_dist)
                if y_mat === nothing
                    y_mat = Matrix{eltype(y_sample)}(undef, n, length(y_sample))
                end
                y_mat[i, :] = y_sample
            end
        end
    end

    return PosteriorSamples(θ_mat, x_mat; y = y_mat)
end

function Random.rand(rng::AbstractRNG, r::HMCLaplaceResult; include_y::Bool = false)
    return rand(rng, r, 1; include_y = include_y)[1]
end

Random.rand(r::HMCLaplaceResult, n::Int; kwargs...) =
    rand(Random.default_rng(), r, n; kwargs...)
Random.rand(r::HMCLaplaceResult; kwargs...) =
    rand(Random.default_rng(), r; kwargs...)
