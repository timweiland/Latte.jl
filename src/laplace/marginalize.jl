using Distributions: Normal

export marginalize

"""
    reported_moments(method, μ_baseline, σ_baseline, marginal) -> (μ, σ)

Per-marginal first-and-second moments used to compute the diagnostic
KLD reported in `MarginalResult.kld_values` and `INLAResult.kld`.
Default extracts `(mean(marginal), std(marginal))` from the marginal
distribution; `MarginalApproximation` subtypes can override this hook
to skip moment re-derivation when construction-time facts already
imply the moments cheaply.

Used by `_compute_kld_values` to avoid (a) constructing a `Normal`
baseline per index and (b) re-deriving SkewNormal moments via the
`Distributions` `mean`/`std` machinery on hot paths. Pairs with the
scalar form of [`moment_symmetric_kld`](@ref).

For methods whose marginals moment-match the Gaussian baseline by
construction, the moment-based KLD is identically zero — those methods
should override `adaptive_upgrade_score` (used by `AdaptiveMarginal`)
with a shape-aware metric.
"""
reported_moments(::MarginalApproximation, μ_baseline::Real, σ_baseline::Real, marginal) =
    (mean(marginal), std(marginal))

# GaussianMarginal: the "marginal" is just the baseline.
reported_moments(::GaussianMarginal, μ_baseline::Real, σ_baseline::Real, ::Any) =
    (μ_baseline, σ_baseline)

# Vector form used by the marginalize wrapper.
function _compute_kld_values(method::MarginalApproximation, marginals, indices, μ_ga, σ_ga)
    kld_values = Vector{Float64}(undef, length(indices))
    @inbounds for (j, i) in enumerate(indices)
        μ_marg, σ_marg = reported_moments(method, μ_ga[i], σ_ga[i], marginals[j])
        kld_values[j] = moment_symmetric_kld(μ_ga[i], σ_ga[i], μ_marg, σ_marg)
    end
    return kld_values
end

# GaussianMarginal short-circuit: KLD is identically 0; skip the loop.
function _compute_kld_values(::GaussianMarginal, marginals, indices, μ_ga, σ_ga)
    return zeros(length(indices))
end

"""
    marginalize(ga, obs_lik, log_prior_θ, method, indices=1:length(mean(ga));
                prior_gmrf=nothing, augmentation_info=nothing)

Compute marginal approximations for specified latent variables.

# Arguments
- `ga`: Gaussian approximation (GMRF object)
- `obs_lik`: Materialized observation likelihood (contains data and hyperparameters)
- `log_prior_θ::Real`: Log-density of hyperparameter prior
- `method::MarginalApproximation`: Approximation method
- `indices::Vector{Int}`: Variable indices to marginalize (default: all)
- `prior_gmrf`: Original prior GMRF (required for Laplace methods, ignored for Gaussian)
- `augmentation_info`: Pass the LGM's `augmentation_info` here when the
  caller is fitting an `AugmentedLatentModel`. `SimplifiedLaplace` uses
  this to apply a base-coordinate equivalence correction (matches R-INLA
  compact-mode behaviour) when computing skew for base latents. Other
  strategies ignore it. `nothing` (default) means "treat the model as
  un-augmented" — appropriate for direct callers and tests that don't
  go through `inla()`.
- `mean_override`: When supplied (a length-`length(mean(ga))` vector),
  `VBCMarginal` uses it as the corrected latent mean μ* rather than
  recomputing the per-θ correction; all other methods ignore it. Used by
  the per-θ INLA hook, which computes μ* once per grid point.

# Returns
`MarginalResult` containing marginal distributions and computation time.
"""
function marginalize(
        ga, obs_lik, log_prior_θ::Real,
        method::MarginalApproximation,
        indices::AbstractVector{<:Integer} = collect(1:length(mean(ga)));
        prior_gmrf = nothing,
        augmentation_info = nothing,
        mean_override = nothing,
    )
    μ_ga = mean(ga)
    σ_ga = std(ga)

    n = length(μ_ga)
    if any(i -> i < 1 || i > n, indices)
        throw(BoundsError(1:n, indices))
    end
    if length(unique(indices)) != length(indices)
        throw(ArgumentError("Duplicate indices not allowed"))
    end

    start_time = time()
    marginals = _marginalize_impl(
        ga, obs_lik, log_prior_θ, method, indices, prior_gmrf;
        augmentation_info = augmentation_info,
        mean_override = mean_override,
    )

    kld_values = _compute_kld_values(method, marginals, indices, μ_ga, σ_ga)
    computation_time = time() - start_time

    return MarginalResult(indices, marginals, method, computation_time, kld_values)
end

# Default fallback: methods that don't care about augmentation_info / mean_override
# just discard them. Only `SimplifiedLaplace` (augmentation) and `VBCMarginal`
# (mean_override) read these kwargs.
function _marginalize_impl(
        ga, obs_lik, log_prior_θ, method, indices, prior_gmrf;
        augmentation_info = nothing,
        mean_override = nothing,
    )
    return _marginalize_impl(ga, obs_lik, log_prior_θ, method, indices, prior_gmrf)
end
