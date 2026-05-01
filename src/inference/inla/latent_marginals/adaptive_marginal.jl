using Distributions: ContinuousUnivariateDistribution, SkewNormal, skewness
using GaussianMarkovRandomFields: NormalLikelihood, LinearlyTransformedLikelihood

export adaptive_upgrade_score

"""
    adaptive_upgrade_score(adaptive::AdaptiveMarginal, candidate::MarginalApproximation,
                           gaussian_baseline::Normal, marginal) -> Float64

Per-index non-Gaussianity score that drives `AdaptiveMarginal`'s upgrade
decision: indices with `score > adaptive.kld_threshold` get re-fit with
the heavier Laplace approximation. Logically distinct from
[`diagnostic_kld`](@ref) (which is a moment-based diagnostic for
display); the upgrade gate must actually distinguish methods that
moment-match their Gaussian baseline.

Default falls back to [`diagnostic_kld`](@ref). Specialised below for
`(SimplifiedLaplace, SkewNormal)` to use `abs(skewness(marginal))`,
which is a closed-form O(1) shape detector — moment-based KLD is
identically zero for moment-matched skew-normals and would never
trigger upgrades.
"""
adaptive_upgrade_score(::AdaptiveMarginal, candidate::MarginalApproximation, baseline::Normal, marginal) =
    diagnostic_kld(candidate, baseline, marginal)

# SimplifiedLaplace builds SkewNormals whose first two moments are
# constructed to match the Gaussian baseline (`_get_skew_params`); the
# only signal of non-Gaussianity is the skew parameter itself. Use
# |skewness(·)|, which is closed-form for SkewNormal and dimensionless.
adaptive_upgrade_score(::AdaptiveMarginal, ::SimplifiedLaplace, ::Normal, marginal::SkewNormal) =
    abs(skewness(marginal))

# When the observation model is Gaussian, the Gaussian approximation is exact —
# no SimplifiedLaplace or Laplace correction is needed.
function _marginalize_impl(
        ga, obs_lik::NormalLikelihood, log_prior_θ::Real,
        method::AdaptiveMarginal, indices::AbstractVector{<:Integer}, prior_gmrf;
        augmentation_info = nothing,
    )
    return _marginalize_impl(ga, obs_lik, log_prior_θ, GaussianMarginal(), indices, prior_gmrf)
end

function _marginalize_impl(
        ga, obs_lik::LinearlyTransformedLikelihood{<:NormalLikelihood}, log_prior_θ::Real,
        method::AdaptiveMarginal, indices::AbstractVector{<:Integer}, prior_gmrf;
        augmentation_info = nothing,
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
        method::AdaptiveMarginal, indices::AbstractVector{<:Integer}, prior_gmrf;
        augmentation_info = nothing,
    )
    if isempty(indices)
        return ContinuousUnivariateDistribution[]
    end

    # Collect indices to Vector{Int} for downstream _marginalize_impl methods
    indices_vec = collect(Int, indices)

    # Step 1: Run SimplifiedLaplace for all indices, passing
    # augmentation_info so the SLA gets its base-coordinate-equivalent
    # correction when the model is augmented.
    sl_marginals = _marginalize_impl(
        ga, obs_lik, log_prior_θ, SimplifiedLaplace(), indices_vec, prior_gmrf;
        augmentation_info = augmentation_info,
    )

    # Step 2: Compute the per-index adaptive upgrade score. For
    # SimplifiedLaplace's SkewNormal output this is `|skewness|`; for
    # other candidate methods it falls back to the moment-based
    # diagnostic KLD. See `adaptive_upgrade_score` docs.
    μ_ga = mean(ga)
    σ_ga = std(ga)
    sl_scores = Vector{Float64}(undef, length(indices_vec))
    @inbounds for (j, i) in enumerate(indices_vec)
        baseline = Normal(μ_ga[i], σ_ga[i])
        sl_scores[j] = adaptive_upgrade_score(method, SimplifiedLaplace(), baseline, sl_marginals[j])
    end

    # Step 3: Check which variables need upgrading
    upgrade_mask = sl_scores .> method.kld_threshold

    if !any(upgrade_mask)
        return sl_marginals
    end

    # Step 4: Upgrade flagged variables to LaplaceMarginal
    upgrade_positions = findall(upgrade_mask)
    upgrade_indices = indices_vec[upgrade_positions]

    idx_str = length(upgrade_indices) <= 10 ? " at indices $upgrade_indices" : ""
    @info "AdaptiveMarginal: upgrading $(length(upgrade_indices))/$(length(indices_vec)) variables to LaplaceMarginal (score > $(method.kld_threshold))$idx_str"

    la_marginals = _marginalize_impl(
        ga, obs_lik, log_prior_θ, LaplaceMarginal(true), upgrade_indices, prior_gmrf
    )

    # Step 5: Quadrature-based SKLD(SimplifiedLaplace, Laplace) on the
    # upgraded subset only. We use the integration-based SKLD here (not
    # the moment-only diagnostic) because we genuinely care about shape
    # disagreement between the SLA and full Laplace on the indices that
    # already triggered an upgrade. Cost is bounded by the number of
    # upgraded indices (typically small).
    sl_la_klds = Vector{Float64}(undef, length(upgrade_positions))
    for (j, pos) in enumerate(upgrade_positions)
        sl_la_klds[j] = quadrature_symmetric_kld(sl_marginals[pos], la_marginals[j])
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
