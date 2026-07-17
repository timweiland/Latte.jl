"""
CCDInterpolantMarginal: CCD interpolant + profiling -> SplineMarginalDistribution.

Builds a parametric CCD interpolant (skewness-corrected Gaussian in z-space)
from the exploration data plus 1+2d extra evaluations. Then profiles along each
θ dimension to get 1D marginals, which are fit with cubic splines.
"""
function _marginalize_impl(
        method::CCDInterpolantMarginal,
        exploration::AbstractHyperparameterExploration,
        model::LatentGaussianModel,
        y,
        progress_callback
    )
    if progress_callback === nothing
        progress_callback = (; kwargs...) -> nothing
    end

    transform = exploration.transform
    θ_star = transform.θ_star
    spec = θ_star.spec
    n_dim = length(θ_star)
    # One marginal per flat coordinate; vector blocks expand to `name[i]`.
    param_names = _expanded_hp_names(spec)

    # One workspace reused across the skewness-correction evaluations.
    θ_star_nt = convert(NamedTuple, convert(NaturalHyperparameters, θ_star))
    ws = make_workspace(model.latent_prior; θ_star_nt...)

    # Step 1: Build CCD interpolant with skewness corrections
    progress_callback(status = "Building CCD interpolant", dimensions = n_dim)
    interp = _build_ccd_interpolant(exploration, model, y, ws)

    # Step 2: Profile each dimension and build spline marginals
    bounds = exploration.integration_bounds

    marginals = map(1:n_dim) do d
        progress_callback(
            status = "Profiling dimension", dimension = d, total = n_dim,
            progress = (d - 1) / n_dim,
        )
        θ_grid, log_profile = profile_marginal(interp, d, method.n_grid, bounds)
        _build_spline_marginal(θ_grid, log_profile, spec, d)
    end

    progress_callback(status = "CCD interpolant marginals complete")

    return NamedTuple(param_names[i] => marginals[i] for i in 1:n_dim)
end

"""
    _build_ccd_interpolant(exploration::CCDExploration, model, y)

Build a CCDInterpolant by reusing stored axial data from CCD exploration.
Zero additional `hyperparameter_logpdf` evaluations needed.
"""
function _build_ccd_interpolant(
        exploration::CCDExploration,
        model::LatentGaussianModel,
        y, ws,
    )
    transform = exploration.transform
    n_dim = length(transform.θ_star)

    mode_logp = exploration.mode_raw_logp

    # CCD axial points are at z = ±f₀√d along each axis.
    # Expected drop for N(0,I): 0.5 * (f₀√d)² = 0.5 * f₀² * d
    expected_drop = 0.5 * exploration.f0^2 * n_dim

    sigma_corr_plus = Vector{Float64}(undef, n_dim)
    sigma_corr_minus = Vector{Float64}(undef, n_dim)

    for i in 1:n_dim
        axial_plus = exploration.axial_raw_logp_plus[i]
        axial_minus = exploration.axial_raw_logp_minus[i]

        if isfinite(axial_plus) && isfinite(mode_logp)
            drop_plus = mode_logp - axial_plus
            sigma_corr_plus[i] = sqrt(expected_drop / max(drop_plus, 1.0e-6))
        else
            # Fallback: fresh evaluation at z=+√2 for this dimension
            sigma_corr_plus[i] = _fresh_sigma_corr(transform, model, y, mode_logp, i, +1, n_dim, ws)
        end

        if isfinite(axial_minus) && isfinite(mode_logp)
            drop_minus = mode_logp - axial_minus
            sigma_corr_minus[i] = sqrt(expected_drop / max(drop_minus, 1.0e-6))
        else
            sigma_corr_minus[i] = _fresh_sigma_corr(transform, model, y, mode_logp, i, -1, n_dim, ws)
        end
    end

    inv_hessian = inv(transform.H)
    return CCDInterpolant(mode_logp, sigma_corr_plus, sigma_corr_minus, transform, inv_hessian)
end

"""
    _build_ccd_interpolant(exploration::AbstractHyperparameterExploration, model, y)

Fallback: Build a CCDInterpolant from fresh evaluations at z=±√2.
Used when exploration is grid-based (no stored CCD axial data).

Performs 1 + 2d evaluations of `hyperparameter_logpdf`.
"""
function _build_ccd_interpolant(
        exploration::AbstractHyperparameterExploration,
        model::LatentGaussianModel,
        y, ws,
    )
    transform = exploration.transform
    θ_star = transform.θ_star
    n_dim = length(θ_star)

    mode_logp = hyperparameter_logpdf(model, θ_star, y; ws = ws)

    sigma_corr_plus = Vector{Float64}(undef, n_dim)
    sigma_corr_minus = Vector{Float64}(undef, n_dim)

    for i in 1:n_dim
        sigma_corr_plus[i] = _fresh_sigma_corr(transform, model, y, mode_logp, i, +1, n_dim, ws)
        sigma_corr_minus[i] = _fresh_sigma_corr(transform, model, y, mode_logp, i, -1, n_dim, ws)
    end

    inv_hessian = inv(transform.H)
    return CCDInterpolant(mode_logp, sigma_corr_plus, sigma_corr_minus, transform, inv_hessian)
end

"""Compute sigma_corr for one dimension/direction via fresh evaluation at z=±√2."""
function _fresh_sigma_corr(transform, model, y, mode_logp, dim, sign, n_dim, ws)
    δ = sqrt(2.0)
    z = zeros(n_dim)
    z[dim] = sign * δ
    θ = transform(z)
    logp = hyperparameter_logpdf(model, θ, y; ws = ws)
    drop = mode_logp - logp
    return 1.0 / sqrt(max(drop, 1.0e-6))
end
