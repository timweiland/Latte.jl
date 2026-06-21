# Laplace-approximation quality diagnostic via PSIS-k̂.
#
# Given a Laplace-based inference result (INLA, TMB, HMC-Laplace), tests
# whether the inner Gaussian approximation `q(x|θ) ≈ p(x|y,θ)` is
# trustworthy by:
#  1. Drawing M samples from q(x|θ)
#  2. Computing log-importance-weights log p(x,y|θ) - log q(x|θ)
#  3. Reporting PSIS-k̂ + relative ESS + a qualitative verdict
#
# Cost: O(M) Gaussian draws + O(M) log-joint evaluations. Typically
# sub-second even for 10⁴-dim latent fields.

using Random
using Distributions: logpdf
using Statistics: quantile

export diagnose, diagnose_chain

"""
    psis_inner_laplace(model::LatentGaussianModel, y, θ_natural::NamedTuple;
                       M = 500, rng = Random.default_rng())

Compute the PSIS-k̂ diagnostic for the inner Laplace approximation at a
given θ (natural scale, as a NamedTuple). Returns a NamedTuple:
`(; pareto_k, ess, rel_ess, interpretation, M)`.
"""
function psis_inner_laplace(
        model::LatentGaussianModel, y, θ_natural::NamedTuple;
        M::Int = 500,
        rng::AbstractRNG = Random.default_rng(),
    )
    prior = model.latent_prior(; θ_natural...)
    obs_lik = model.observation_model(y; θ_natural...)
    q_post = gaussian_approximation(prior, obs_lik)

    log_w = Vector{Float64}(undef, M)
    for i in 1:M
        x_i = rand(rng, q_post)
        log_w[i] = logpdf(prior, x_i) + loglik(x_i, obs_lik) - logpdf(q_post, x_i)
    end

    k̂_est = pareto_k(log_w)
    rel = rel_ess_is(log_w)
    return (;
        rel_ess = rel,
        ess = rel * M,
        pareto_k = k̂_est,
        interpretation = trust_verdict(rel),
        M = M,
    )
end

"""
    diagnose(r::InferenceResult; M = 500, rng = Random.default_rng())

Run the PSIS-k̂ diagnostic at the hyperparameter mode of a Laplace-based
inference result. Works uniformly over `INLAResult`, `TMBResult`, and
`HMCLaplaceResult` — all three have a well-defined inner Laplace at
their MAP (INLA uses it for grid centering; TMB reports it as the
answer; HMC-Laplace uses it per-sample).

# Returns

`NamedTuple{(:rel_ess, :ess, :pareto_k, :interpretation, :M)}`:
- `rel_ess ∈ (0, 1]`  — relative effective sample size (primary metric)
- `ess`               — absolute ESS of the importance weights
- `pareto_k`          — Zhang-Stephens GPD shape parameter
- `interpretation`    — `:excellent` / `:acceptable` / `:unreliable`
- `M`                 — number of Gaussian samples used

`:unreliable` suggests switching to a method that doesn't rely on the
Laplace approximation being exact (e.g., a future `sparsenuts(lgm, y)`
that samples the joint (θ, x) directly).
"""
function diagnose(
        r::InferenceResult;
        M::Int = 500, rng::AbstractRNG = Random.default_rng(),
    )
    θ_natural_nt = convert(NamedTuple, hyperparameter_mode(r))
    return psis_inner_laplace(
        model(r), observations(r), θ_natural_nt; M = M, rng = rng,
    )
end

"""
    diagnose_chain(r::HMCLaplaceResult; M = 500, rng = ...,
                   quantiles = (0.025, 0.5, 0.975))

HMC-Laplace only: run the PSIS diagnostic at the MAP *and* at chain
quantiles of θ. Useful when HMC explores regions where the local
quadratic fit may be poor — if `:at_map` is excellent but `:q_0_975` is
unreliable, the Laplace approximation degrades off-centre and you
should distrust tail inference.

Returns a `NamedTuple` mapping quantile names (`q_0_025`, `q_0_5`,
`q_0_975`, plus `:at_map`) to per-point diagnostic NamedTuples.
"""
function diagnose_chain(
        r::HMCLaplaceResult;
        M::Int = 500,
        rng::AbstractRNG = Random.default_rng(),
        quantiles = (0.025, 0.5, 0.975),
    )
    spec = r.model.hyperparameter_spec
    y = r.observations

    # At MAP (TMB warm-start point)
    map_wh = WorkingHyperparameters(r.tmb_mode, spec)
    map_nt = convert(NamedTuple, convert(NaturalHyperparameters, map_wh))
    at_map = psis_inner_laplace(r.model, y, map_nt; M = M, rng = rng)

    # At each requested chain quantile (elementwise)
    quantile_results = Pair{Symbol, NamedTuple}[]
    for q in quantiles
        θ_q_vec = [quantile(view(r.θ_samples, :, j), q) for j in axes(r.θ_samples, 2)]
        θ_q_wh = WorkingHyperparameters(θ_q_vec, spec)
        θ_q_nt = convert(NamedTuple, convert(NaturalHyperparameters, θ_q_wh))
        key = Symbol("q_", replace(string(q), "." => "_"))
        push!(quantile_results, key => psis_inner_laplace(r.model, y, θ_q_nt; M = M, rng = rng))
    end

    return (; at_map, quantile_results...)
end
