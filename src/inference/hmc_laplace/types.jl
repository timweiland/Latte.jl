using Distributions: Normal, MixtureModel
using OrderedCollections: OrderedDict
using Statistics: mean, std

export HMCLaplaceResult

"""
    HMCLaplaceResult <: InferenceResult

Result of tmbstan-style HMC on the Laplace marginal via `hmc_laplace(model, y)`.
θ is sampled by NUTS on `L(θ) = log ∫ p(y, x | θ) dx` (Laplace-approximated);
the latent field is reconstructed per-sample from the inner Laplace at each
drawn θ.

# Fields
- `θ_samples::Matrix{Float64}` — `(n_samples × n_θ)` θ chain in working space.
- `x_cond_means::Matrix{Float64}` / `x_cond_stds::Matrix{Float64}` —
  `(n_samples × n_latent)` per-θ-sample conditional means and marginal
  standard deviations from the inner Laplace.
- `stats` — per-step AdvancedHMC statistics (tree depth, step size,
  divergences, acceptance).
- `tmb_mode::Vector{Float64}` — warm-start point from the TMB MAP.
- `n_warmup::Int` — number of warmup samples.
- `model::LatentGaussianModel`, `observations` — the fitted model and data.
- `time_elapsed::Float64` — total wall time.

Latent marginals are built at construction as `MixtureModel{Normal}` per
site. Hyperparameter marginals are a `Normal` fit per column of
`θ_samples` (a simple summary; users wanting raw chain access via
`samples(r)`). `log_marginal_likelihood(r)` returns `nothing` — HMC
doesn't natively produce one without bridge sampling.
"""
struct HMCLaplaceResult{Hp, Lm, S, M, Y} <: InferenceResult
    hyperparameter_marginals::Hp        # Vector{Normal}
    latent_marginals::Lm                # Vector{MixtureModel{...}}
    θ_samples::Matrix{Float64}
    x_cond_means::Matrix{Float64}
    x_cond_stds::Matrix{Float64}
    stats::S
    tmb_mode::Vector{Float64}
    n_warmup::Int
    model::M
    observations::Y
    time_elapsed::Float64
end

# ─── Constructor: build Tier 1 marginals from chain samples ────────────────
function _build_hmc_marginals(θ_samples::Matrix{Float64}, x_cond_means::Matrix{Float64}, x_cond_stds::Matrix{Float64})
    n_samples, n_θ = size(θ_samples)
    n_latent = size(x_cond_means, 2)

    # θ-marginals: Normal fit per coordinate (summary of empirical chain)
    θ_marginals = [
        Normal(mean(view(θ_samples, :, j)), std(view(θ_samples, :, j)))
            for j in 1:n_θ
    ]

    # Latent marginals: per-site mixture of per-θ-sample Gaussians, uniform
    # weights. p(x_i | y) ≈ (1/K) Σ_k N(μ̂_i(θ_k), σ̂_i(θ_k)).
    latent_marginals = Vector{MixtureModel}(undef, n_latent)
    for i in 1:n_latent
        components = [
            Normal(x_cond_means[k, i], x_cond_stds[k, i])
                for k in 1:n_samples
        ]
        latent_marginals[i] = MixtureModel(components)  # uniform weights by default
    end

    return θ_marginals, latent_marginals
end

# ─── Protocol implementations ──────────────────────────────────────────────
latent_marginals(r::HMCLaplaceResult) = r.latent_marginals
hyperparameter_marginals(r::HMCLaplaceResult) = r.hyperparameter_marginals

latent_groups(::HMCLaplaceResult) = OrderedDict{Symbol, UnitRange{Int}}()

function hyperparameter_groups(r::HMCLaplaceResult)
    names = collect(keys(r.model.hyperparameter_spec.free))
    groups = OrderedDict{Symbol, UnitRange{Int}}()
    for (i, name) in enumerate(names)
        groups[name] = i:i
    end
    return groups
end

function hyperparameter_mode(r::HMCLaplaceResult)
    wh = WorkingHyperparameters(r.tmb_mode, r.model.hyperparameter_spec)
    return convert(NaturalHyperparameters, wh)
end

model(r::HMCLaplaceResult) = r.model
observations(r::HMCLaplaceResult) = r.observations

# Convergence heuristic: divergent transitions are a small fraction of
# post-warmup samples (< 5%). Strict zero is too harsh — occasional
# divergences happen even on well-behaved targets. Users wanting the
# exact count should call `divergences(r)`.
function converged(r::HMCLaplaceResult)
    n = length(r.stats)
    return n == 0 ? false : divergences(r) < 0.05 * n
end

time_elapsed(r::HMCLaplaceResult) = r.time_elapsed

# log p(y) not natively available from HMC; would require bridge sampling.
log_marginal_likelihood(::HMCLaplaceResult) = nothing

# ─── MCMC diagnostics (HMC-specific; not on the abstract protocol) ─────────
export samples, divergences, mean_tree_depth, acceptance_rate, mean_step_size

"""
    samples(r::HMCLaplaceResult) -> Matrix{Float64}

Raw θ chain, `(n_samples × n_θ)`. Each row is one post-warmup draw in
working space.
"""
samples(r::HMCLaplaceResult) = r.θ_samples

"""
    divergences(r::HMCLaplaceResult) -> Int

Number of divergent transitions in the post-warmup chain. Typical quality
threshold: 0 (or at most a small number relative to `length(r.stats)`).
"""
function divergences(r::HMCLaplaceResult)
    return count(s -> getfield(s, :numerical_error), r.stats)
end

"""
    mean_tree_depth(r::HMCLaplaceResult) -> Float64

Mean NUTS tree depth across the post-warmup chain. Low values (≤ 3) with
good preconditioning indicate efficient sampling; hitting the cap (≥ 10)
suggests misspecified mass matrix or hard-to-explore geometry.
"""
mean_tree_depth(r::HMCLaplaceResult) = mean(s -> s.tree_depth, r.stats)

"""
    acceptance_rate(r::HMCLaplaceResult) -> Float64

Mean accept probability across the post-warmup chain. Typical target: 0.8.
"""
acceptance_rate(r::HMCLaplaceResult) = mean(s -> s.acceptance_rate, r.stats)

"""
    mean_step_size(r::HMCLaplaceResult) -> Float64

Mean integrator step size across the post-warmup chain. Useful for
diagnosing adaptation; extreme values often indicate preconditioner issues.
"""
mean_step_size(r::HMCLaplaceResult) = mean(s -> s.step_size, r.stats)

# ─── Pretty printing ───────────────────────────────────────────────────────
function Base.show(io::IO, r::HMCLaplaceResult)
    spec = r.model.hyperparameter_spec
    names = collect(keys(spec.free))
    n_samples = size(r.θ_samples, 1)

    println(io, "HMCLaplaceResult:")
    println(io, "  Model: ", typeof(r.model))
    println(io, "  θ samples: ", n_samples, " (+ ", r.n_warmup, " warmup)")
    println(io, "  Hyperparameters (HMC mean ± SD in working space):")
    for (i, name) in enumerate(names)
        μ = mean(view(r.θ_samples, :, i))
        σ = std(view(r.θ_samples, :, i))
        @printf(io, "    %-8s %+8.4f ± %.4f\n", String(name), μ, σ)
    end
    println(io, "  Latent dimension: ", size(r.x_cond_means, 2))
    println(io, "  Diagnostics:")
    @printf(io, "    accept rate:  %.3f\n", acceptance_rate(r))
    @printf(io, "    tree depth:   %.2f (mean)\n", mean_tree_depth(r))
    @printf(io, "    step size:    %.4f (mean)\n", mean_step_size(r))
    @printf(io, "    divergences:  %d\n", divergences(r))
    return @printf(io, "  Time: %.2f s\n", r.time_elapsed)
end
