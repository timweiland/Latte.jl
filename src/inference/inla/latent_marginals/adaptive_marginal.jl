using Distributions: ContinuousUnivariateDistribution, SkewNormal, skewness
using GaussianMarkovRandomFields: NormalLikelihood, LinearlyTransformedLikelihood
using LinearAlgebra

export adaptive_upgrade_score

function _is_adaptive_marginal_numerical_failure(e)
    return e isa DomainError ||
        e isa PosDefException ||
        e isa LinearAlgebra.ZeroPivotException ||
        e isa LinearAlgebra.SingularException
end

"""
    adaptive_upgrade_score(adaptive::AdaptiveMarginal, candidate::MarginalApproximation,
                           gaussian_baseline::Normal, marginal) -> Float64

Fallback non-Gaussianity score (used only when no diagonal exp-family fast
path exists, so `_fourth_order_scores` returns `nothing`): indices with
`score > adaptive.tol` get re-fit with the heavier Laplace approximation.
Logically distinct from [`diagnostic_kld`](@ref) (which is a moment-based
diagnostic for display); the upgrade gate must actually distinguish methods
that moment-match their Gaussian baseline.

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

# Magnitude of the leading term SimplifiedLaplace neglects, per marginalized
# index: the standardized 4th-order log-density coefficient
# `a₄_i = Σ_k h⁗(η_k)·(Σ[k,i]/σ_i)⁴`, with the i-th term dropped to match the
# SLA's γ_3 construction. This gates escalation on whether the 3rd-order
# (skew-normal) form misses curvature — not on how skewed the marginal is.
# Diagonal exp-fam fast path only; returns `nothing` (caller falls back to the
# skew-based score) when the observation Hessian isn't diagonal.
function _fourth_order_scores(ga, obs_lik, indices::AbstractVector{<:Integer})
    μ = mean(ga)
    diag_h4 = fourth_derivative_diagonal(obs_lik, μ)
    diag_h4 === nothing && return nothing
    Σ = selected_covariance(ga)
    σ = sqrt.(max.(diag(Σ), 0.0))
    scores = Vector{Float64}(undef, length(indices))
    for (j, i) in enumerate(indices)
        σ_i = σ[i]
        if σ_i < 1.0e-10
            scores[j] = 0.0
            continue
        end
        cond_col = conditional_column(ga, i)
        s = 0.0
        @inbounds for m in eachindex(diag_h4.indices)
            k = diag_h4.indices[m]
            k == i && continue
            d = cond_col[k] / σ_i
            s += diag_h4.values[m] * d^4
        end
        scores[j] = abs(s)
    end
    return scores
end

# When the observation model is Gaussian, the Gaussian approximation is exact —
# no SimplifiedLaplace or Laplace correction is needed.
function _marginalize_impl(
        ga, obs_lik::NormalLikelihood, log_prior_θ::Real,
        method::AdaptiveMarginal, indices::AbstractVector{<:Integer}, prior_gmrf;
        augmentation_info = nothing,
        mean_override = nothing,
    )
    return _marginalize_impl(ga, obs_lik, log_prior_θ, GaussianMarginal(), indices, prior_gmrf)
end

function _marginalize_impl(
        ga, obs_lik::LinearlyTransformedLikelihood{<:NormalLikelihood}, log_prior_θ::Real,
        method::AdaptiveMarginal, indices::AbstractVector{<:Integer}, prior_gmrf;
        augmentation_info = nothing,
        mean_override = nothing,
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
        mean_override = nothing,
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

    # Step 2: score each index by the magnitude of the leading term SLA
    # neglects (the 4th-order coefficient |a₄|). With no diagonal exp-fam
    # fast path, fall back to the skew-based score against the Gaussian.
    sl_scores = _fourth_order_scores(ga, obs_lik, indices_vec)
    if sl_scores === nothing
        μ_ga = mean(ga)
        σ_ga = std(ga)
        sl_scores = Vector{Float64}(undef, length(indices_vec))
        @inbounds for (j, i) in enumerate(indices_vec)
            baseline = Normal(μ_ga[i], σ_ga[i])
            sl_scores[j] = adaptive_upgrade_score(method, SimplifiedLaplace(), baseline, sl_marginals[j])
        end
    end

    # Step 3: Check which variables need upgrading
    upgrade_mask = sl_scores .> method.tol

    if !any(upgrade_mask)
        return sl_marginals
    end

    # Step 4: Upgrade flagged variables to LaplaceMarginal
    upgrade_positions = findall(upgrade_mask)
    upgrade_indices = indices_vec[upgrade_positions]

    idx_str = length(upgrade_indices) <= 10 ? " at indices $upgrade_indices" : ""
    @info "AdaptiveMarginal: upgrading $(length(upgrade_indices))/$(length(indices_vec)) variables to full Laplace (|a₄| > $(method.tol))$idx_str"

    la_marginals = try
        _marginalize_impl(
            ga, obs_lik, log_prior_θ, LaplaceMarginal(true), upgrade_indices, prior_gmrf
        )
    catch e
        _is_adaptive_marginal_numerical_failure(e) || rethrow(e)
        @warn "AdaptiveMarginal: Laplace upgrade failed numerically; keeping SimplifiedLaplace marginals for this grid point" exception_type = typeof(e) exception = sprint(showerror, e)
        return sl_marginals
    end

    # Step 5: Merge results — start with SL, overwrite upgraded positions
    result = convert(Vector{ContinuousUnivariateDistribution}, copy(sl_marginals))
    for (j, pos) in enumerate(upgrade_positions)
        result[pos] = la_marginals[j]
    end

    return result
end
