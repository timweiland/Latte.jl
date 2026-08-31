using AdvancedHMC
using DifferentiationInterface
using LogDensityProblems
using LinearAlgebra: Symmetric
using Random
import FiniteDiff

export hmc_laplace

# ─── LogDensityProblems target: θ ↦ hyperparameter_logpdf ──────────────────
struct HMCTarget{M, Y, WS, S, D}
    model::M
    y::Y
    ws::WS
    spec::S
    dim::Int
    diff_strategy::D
end

LogDensityProblems.capabilities(::Type{<:HMCTarget}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(t::HMCTarget) = t.dim

# Build the NUTS target. The workspace is seeded from the (primal) MAP rather
# than a blanket 1.0, which is out of domain for bounded hyperparameters such
# as an AR(1) ρ; it is then reused for every objective evaluation, including
# the Dual-typed ones the AD gradient makes — the same arrangement `tmb` uses
# for its outer Hessian.
function _hmc_target(model, y, θ̂, spec, diff_strategy)
    θ̂_natural_nt = convert(
        NamedTuple, convert(NaturalHyperparameters, WorkingHyperparameters(θ̂, spec))
    )
    ws = make_workspace(model.latent_prior; θ̂_natural_nt...)
    return HMCTarget(model, y, ws, spec, length(θ̂), diff_strategy)
end

# NUTS sometimes proposes θ values where the inner Laplace's posterior
# precision is non-PD (CHOLMOD fails) or the latent prior is otherwise
# degenerate. Map those to -Inf so AdvancedHMC treats the step as a
# divergence rather than erroring out of the chain. Exceptions that signal a
# bug rather than an out-of-domain θ — chiefly the `Float64(::Dual)`
# `MethodError` from an AD-incompatible model — are re-thrown: swallowing one
# under the AD gradient would zero the gradient at every θ and yield a
# silently meaningless chain.
function _hmc_objective(t::HMCTarget)
    return function (θ)
        return try
            hyperparameter_logpdf(
                t.model, WorkingHyperparameters(θ, t.spec), t.y; ws = t.ws,
            )
        catch e
            _is_numerical_failure(e) || rethrow(e)
            oftype(θ[1], -Inf)
        end
    end
end

LogDensityProblems.logdensity(t::HMCTarget, θ_vec) = _hmc_objective(t)(θ_vec)

function LogDensityProblems.logdensity_and_gradient(t::HMCTarget, θ_vec)
    return _logdensity_and_gradient(t.diff_strategy, t, θ_vec)
end

# A fixed number of Dual passes per gradient (ForwardDiff evaluates a chunk of
# partials at a time), rather than one objective evaluation per hyperparameter.
function _logdensity_and_gradient(strategy::ADStrategy, t::HMCTarget, θ_vec)
    f = _hmc_objective(t)
    v, g = DifferentiationInterface.value_and_gradient(f, strategy.backend, θ_vec)
    return isfinite(v) ? (v, g) : (v, zero(θ_vec))
end

# 2d objective evaluations per gradient — kept for models whose objective
# cannot carry Duals (see the `diff_strategy` note on `hmc_laplace`).
function _logdensity_and_gradient(::FiniteDiffStrategy, t::HMCTarget, θ_vec)
    f = _hmc_objective(t)
    v = f(θ_vec)
    isfinite(v) || return v, zero(θ_vec)
    return v, FiniteDiff.finite_difference_gradient(f, θ_vec)
end

# ─── hmc_laplace entry point ──────────────────────────────────────────────
"""
    hmc_laplace(model::LatentGaussianModel, y;
                n_samples=500, n_warmup=200,
                rng=Random.default_rng(),
                progress=false,
                diff_strategy=ADStrategy()) -> HMCLaplaceResult

tmbstan-style HMC on the Laplace marginal. Samples θ via NUTS with the
Laplace approximation `q(x | θ)` substituted for the true `p(x | y, θ)`;
the latent field is reconstructed per-sample from the inner Laplace at
each drawn θ.

Pipeline:
1. Run `tmb(model, y)` → MAP `θ̂`, Laplace covariance `Σ_θ̂`. Used as
   warm-start initial point and as the HMC mass matrix (dense metric
   `M⁻¹ = Σ_θ̂`). Preconditioning matches the target's local curvature,
   so NUTS tree depths stay low.
2. NUTS samples θ on `L(θ) = log p(y, θ)` via `hyperparameter_logpdf`,
   with gradients from `diff_strategy` — by default a handful of AD passes
   through the inner Laplace, not one solve per hyperparameter.
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
- `diff_strategy`: used both for the TMB warm-start (mode + Σ_θ) and for the
  NUTS gradient of `L(θ)`. Default `ADStrategy()` costs `ceil(d / chunk)` AD
  passes per gradient (one, for `d` up to ForwardDiff's default chunk size);
  `FiniteDiffStrategy()` costs `2d` evaluations of the inner Laplace, which is
  what makes the FD path impractical past a handful of hyperparameters. AD is
  noise-robust on augmented LGMs and works for `@latte` models with recognized
  GMRF latents (IID / RW / AR1 / Besag) and for the broad class of
  custom-`logpdf` likelihoods. Reach for
  `FiniteDiffStrategy()` only in the narrow case where a hyperparameter-derived
  value is hoisted into the observation payload by the `@latte` prelude-lift
  (e.g. `φ = exp(log_φ)` in Tweedie), which can't stay `Dual`-typed; tracked in
  `tasks/dppl-adapter-outer-ad-closure.org`.

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

    # ── Step 1: TMB warm-start ───────────────────────────────────────────
    tmb_r = tmb(model, y_obs; diff_strategy = diff_strategy)
    θ̂ = tmb_r.θ_map
    Σ_θ = tmb_r.θ_cov

    # ── Step 2: build LogDensityProblems target ──────────────────────────
    target = _hmc_target(model, y_obs, θ̂, spec, diff_strategy)
    ws = target.ws

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
        obs_lik = model.observation_model(y_obs; θ_nt...)
        # θ_nt is a concrete sample here (primal), so the workspace is always safe.
        x_post = if model.latent_prior isa NonGaussianLatentPrior
            gaussian_approximation(model.latent_prior, obs_lik; θ = θ_nt, ws = ws)
        else
            gaussian_approximation(model.latent_prior(ws; θ_nt...), obs_lik)
        end
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

    # θ samples on the chain are stored in working space; convert to
    # natural space to match the PosteriorSamples contract.
    spec = r.model.hyperparameter_spec
    n_hp = size(r.θ_samples, 2)
    θ_mat = Matrix{Float64}(undef, n, n_hp)
    n_x = size(r.x_cond_means, 2)
    x_mat = Matrix{Float64}(undef, n, n_x)
    y_mat = nothing

    # Reconstruct the Gaussian approximation on demand for each unique θ_k —
    # this gives a proper joint x-draw (capturing correlations), matching the
    # convention used in INLA's rand. Batch by unique index to amortize the
    # Laplace reconstruction cost.
    # Seed the workspace from an in-domain posterior draw, not a blanket 1.0
    # (out of domain for bounded hyperparameters such as an AR(1) ρ).
    seed_nt = convert(NamedTuple, convert(NaturalHyperparameters, WorkingHyperparameters(r.θ_samples[1, :], spec)))
    ws = make_workspace(r.model.latent_prior; seed_nt...)

    for k in unique(idxs)
        θ_wh = WorkingHyperparameters(r.θ_samples[k, :], spec)
        θ_nt = convert(NamedTuple, convert(NaturalHyperparameters, θ_wh))
        # Free hyperparameters only — the θ matrix has one column per free hp
        # (θ_nt also carries any fixed values, used below for the densities).
        θ_nat_vec = collect(convert(NaturalHyperparameters, θ_wh))
        obs_lik = r.model.observation_model(r.observations; θ_nt...)
        x_post = if r.model.latent_prior isa NonGaussianLatentPrior
            gaussian_approximation(r.model.latent_prior, obs_lik; θ = θ_nt, ws = ws)
        else
            gaussian_approximation(r.model.latent_prior(ws; θ_nt...), obs_lik)
        end

        for i in findall(==(k), idxs)
            θ_mat[i, :] = θ_nat_vec
            x_sample = rand(rng, x_post)
            x_mat[i, :] = x_sample
            if include_y
                x_for_obs = _x_for_obs_model(r.model, x_sample)
                y_dist = GaussianMarkovRandomFields.conditional_distribution(
                    r.model.observation_model, x_for_obs; θ_nt...
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
