using Distributions: Normal

export marginalize

# GaussianMarginal: KLD is trivially zero (comparing Gaussian to itself)
function _compute_kld_values(::GaussianMarginal, marginals, indices, μ_ga, σ_ga)
    return zeros(length(indices))
end

# Non-Gaussian methods: compute SKLD against Gaussian baseline from GA
function _compute_kld_values(::MarginalApproximation, marginals, indices, μ_ga, σ_ga)
    kld_values = Vector{Float64}(undef, length(indices))
    for (j, i) in enumerate(indices)
        gaussian_baseline = Normal(μ_ga[i], σ_ga[i])
        kld_values[j] = symmetric_kld(gaussian_baseline, marginals[j])
    end
    return kld_values
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

# Returns
`MarginalResult` containing marginal distributions and computation time.
"""
function marginalize(
        ga, obs_lik, log_prior_θ::Real,
        method::MarginalApproximation,
        indices::AbstractVector{<:Integer} = collect(1:length(mean(ga)));
        prior_gmrf = nothing,
        augmentation_info = nothing,
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
    )

    kld_values = _compute_kld_values(method, marginals, indices, μ_ga, σ_ga)
    computation_time = time() - start_time

    return MarginalResult(indices, marginals, method, computation_time, kld_values)
end

# Default fallback: methods that don't care about augmentation_info just
# discard it. Only `SimplifiedLaplace` (and any future method that needs
# augmented-coordinate awareness) overrides this.
function _marginalize_impl(
        ga, obs_lik, log_prior_θ, method, indices, prior_gmrf;
        augmentation_info = nothing,
    )
    return _marginalize_impl(ga, obs_lik, log_prior_θ, method, indices, prior_gmrf)
end
