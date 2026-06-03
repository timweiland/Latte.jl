"""
Build a `SplineMarginalDistribution` from working-space grid values and
unnormalized log marginal density.
"""
function _build_spline_marginal(
        η_grid::Vector{Float64},
        log_marginal_unnorm::Vector{Float64},
        spec,
        dim::Int
    )
    T = Float64
    n = length(η_grid)

    hp = spec.free[dim]
    tfm = hp.transform              # natural → working
    inv_tfm = Bijectors.inverse(tfm)  # working → natural

    # The pdf is a continuous cubic spline of the log-density. Normalize it and
    # compute the CDF + moments by integrating that spline on a fine quadrature
    # grid (build-time spline evaluations only — no extra model evaluations), so
    # the marginal is self-consistent (∫pdf = 1) and its accuracy is decoupled
    # from the exploration-grid density.
    logpdf_unnorm = CubicSpline(log_marginal_unnorm, η_grid; extrapolation = ExtrapolationType.Linear)

    n_fine = max(1024, 16 * n)
    ηf = collect(range(η_grid[1], η_grid[end], length = n_fine))
    Δf = ηf[2] - ηf[1]
    logd_f = [logpdf_unnorm(η) for η in ηf]
    trapz(vals) = (sum(vals) - (vals[1] + vals[end]) / 2) * Δf

    # log Z = log ∫ exp(logpdf) dη via the trapezoidal rule (max-shifted for stability)
    mshift = maximum(logd_f)
    log_Z = log(trapz(exp.(logd_f .- mshift))) + mshift

    # Normalized log-density spline on the original grid (a constant shift of the
    # unnormalized spline keeps logpdf queries interpolating the actual points).
    logpdf_spline = CubicSpline(log_marginal_unnorm .- log_Z, η_grid; extrapolation = ExtrapolationType.Linear)

    # Fine normalized working-space density + natural-space points.
    densf = exp.(logd_f .- log_Z)
    x_f = [inv_tfm(η) for η in ηf]

    # CDF via cumulative trapezoid on the fine grid.
    cdf_fine = zeros(T, n_fine)
    for i in 2:n_fine
        cdf_fine[i] = cdf_fine[i - 1] + (densf[i] + densf[i - 1]) / 2 * Δf
    end
    cdf_fine ./= cdf_fine[end]
    cdf_fine[end] = one(T)
    cdf_spline = CubicSpline(cdf_fine, ηf; extrapolation = ExtrapolationType.Linear)

    # Transform direction + natural-space bounds.
    x_lo = inv_tfm(η_grid[1])
    x_hi = inv_tfm(η_grid[end])
    transform_increasing = x_lo < x_hi
    bounds = (T(min(x_lo, x_hi)), T(max(x_lo, x_hi)))

    # Moments via fine trapezoid in natural space: E[Xᵏ] = ∫ g(η)ᵏ p_η(η) dη.
    mean_val = trapz(x_f .* densf)
    second_moment = trapz((x_f .^ 2) .* densf)
    var_val = max(zero(T), second_moment - mean_val^2)

    # Mode: argmax of the natural-space log-density on the fine grid.
    log_density_natural = [
        (logd_f[i] - log_Z) + Bijectors.logabsdetjac(tfm, x_f[i])
            for i in 1:n_fine
    ]
    mode_val = T(x_f[argmax(log_density_natural)])

    return SplineMarginalDistribution{T, typeof(logpdf_spline), typeof(cdf_spline)}(
        η_grid, logpdf_spline, cdf_spline,
        tfm, inv_tfm, transform_increasing,
        bounds,
        T(mean_val), T(var_val), mode_val
    )
end

# ==================== summary_df support ====================

function summary_df(marginals::NamedTuple{<:Any, <:Tuple{Vararg{<:SplineMarginalDistribution}}})
    parameters = Symbol[]
    modes = Float64[]
    medians = Float64[]
    q025s = Float64[]
    q975s = Float64[]
    means = Float64[]
    stds = Float64[]

    for (param_name, marginal) in pairs(marginals)
        push!(parameters, param_name)
        push!(modes, mode(marginal))
        push!(medians, quantile(marginal, 0.5))
        push!(q025s, quantile(marginal, 0.025))
        push!(q975s, quantile(marginal, 0.975))
        push!(means, mean(marginal))
        push!(stds, std(marginal))
    end

    return DataFrame(
        parameter = parameters,
        mode = modes,
        median = medians,
        q2_5 = q025s,
        q97_5 = q975s,
        mean = means,
        std = stds
    )
end
