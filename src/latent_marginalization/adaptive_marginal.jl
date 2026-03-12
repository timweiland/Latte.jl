using Distributions: ContinuousUnivariateDistribution
using GaussianMarkovRandomFields: NormalLikelihood, LinearlyTransformedLikelihood

# When the observation model is Gaussian, the Gaussian approximation is exact —
# no SimplifiedLaplace or Laplace correction is needed.
function _marginalize_impl(
        ga, obs_lik::NormalLikelihood, log_prior_θ::Real,
        method::AdaptiveMarginal, indices::AbstractVector{<:Integer}, prior_gmrf
    )
    return _marginalize_impl(ga, obs_lik, log_prior_θ, GaussianMarginal(), indices, prior_gmrf)
end

function _marginalize_impl(
        ga, obs_lik::LinearlyTransformedLikelihood{<:NormalLikelihood}, log_prior_θ::Real,
        method::AdaptiveMarginal, indices::AbstractVector{<:Integer}, prior_gmrf
    )
    return _marginalize_impl(ga, obs_lik, log_prior_θ, GaussianMarginal(), indices, prior_gmrf)
end

"""
    _marginalize_impl(ga, obs_lik, log_prior_θ, method::AdaptiveMarginal, indices, prior_gmrf)

Adaptive marginalization: starts with SimplifiedLaplace, escalates to LaplaceMarginal
for variables where SKLD(Gaussian, SimplifiedLaplace) exceeds the threshold.

When the observation model is Gaussian (Normal family), dispatches to GaussianMarginal
since the Gaussian approximation is exact.
"""
function _marginalize_impl(
        ga, obs_lik, log_prior_θ::Real,
        method::AdaptiveMarginal, indices::AbstractVector{<:Integer}, prior_gmrf
    )
    if isempty(indices)
        return ContinuousUnivariateDistribution[]
    end

    # Collect indices to Vector{Int} for downstream _marginalize_impl methods
    indices_vec = collect(Int, indices)

    # Step 1: Run SimplifiedLaplace for all indices
    sl_marginals = _marginalize_impl(ga, obs_lik, log_prior_θ, SimplifiedLaplace(), indices_vec, prior_gmrf)

    # Step 2: Compute SKLD(Gaussian, SimplifiedLaplace) per variable
    μ_ga = mean(ga)
    σ_ga = std(ga)
    sl_kld = _compute_kld_values(SimplifiedLaplace(), sl_marginals, indices_vec, μ_ga, σ_ga)

    # Step 3: Check which variables need upgrading
    upgrade_mask = sl_kld .> method.kld_threshold

    if !any(upgrade_mask)
        return sl_marginals
    end

    # Step 4: Upgrade flagged variables to LaplaceMarginal
    upgrade_positions = findall(upgrade_mask)
    upgrade_indices = indices_vec[upgrade_positions]

    idx_str = length(upgrade_indices) <= 10 ? " at indices $upgrade_indices" : ""
    @info "AdaptiveMarginal: upgrading $(length(upgrade_indices))/$(length(indices_vec)) variables to LaplaceMarginal (SKLD > $(method.kld_threshold))$idx_str"

    la_marginals = _marginalize_impl(
        ga, obs_lik, log_prior_θ, LaplaceMarginal(true), upgrade_indices, prior_gmrf
    )

    # Step 5: Compute SKLD(SimplifiedLaplace, Laplace) for upgraded variables
    sl_la_klds = Vector{Float64}(undef, length(upgrade_positions))
    for (j, pos) in enumerate(upgrade_positions)
        sl_la_klds[j] = symmetric_kld(sl_marginals[pos], la_marginals[j])
    end

    max_sl_la = maximum(sl_la_klds)
    mean_sl_la = sum(sl_la_klds) / length(sl_la_klds)

    if max_sl_la > method.kld_threshold
        @warn "AdaptiveMarginal: SKLD(SimplifiedLaplace, Laplace) exceeds threshold for upgraded variables" *
            " (max=$(round(max_sl_la, digits = 4)), mean=$(round(mean_sl_la, digits = 4)))"
    else
        @info "AdaptiveMarginal: SKLD(SimplifiedLaplace, Laplace) for upgraded variables:" *
            " max=$(round(max_sl_la, digits = 4)), mean=$(round(mean_sl_la, digits = 4))"
    end

    # Step 6: Merge results — start with SL, overwrite upgraded positions
    result = convert(Vector{ContinuousUnivariateDistribution}, copy(sl_marginals))
    for (j, pos) in enumerate(upgrade_positions)
        result[pos] = la_marginals[j]
    end

    return result
end
