using StatsFuns: logsumexp

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

    # Normalize: compute log Z via logsumexp + log(step)
    Δη = η_grid[2] - η_grid[1]
    log_Z = logsumexp(log_marginal_unnorm) + log(Δη)
    log_density_norm = log_marginal_unnorm .- log_Z

    # Build logpdf spline in working space
    logpdf_spline = CubicSpline(log_density_norm, η_grid; extrapolation = ExtrapolationType.Linear)

    # Compute CDF via cumulative trapezoid rule on normalized density
    density_norm = exp.(log_density_norm)
    cdf_values = zeros(T, n)
    for i in 2:n
        cdf_values[i] = cdf_values[i - 1] + (density_norm[i] + density_norm[i - 1]) / 2 * Δη
    end
    # Force endpoints
    cdf_values[1] = zero(T)
    cdf_values ./= cdf_values[end]
    cdf_values[end] = one(T)

    cdf_spline = CubicSpline(cdf_values, η_grid; extrapolation = ExtrapolationType.Linear)

    # Determine transform direction
    x_lo = inv_tfm(η_grid[1])
    x_hi = inv_tfm(η_grid[end])
    transform_increasing = x_lo < x_hi

    bounds = (T(min(x_lo, x_hi)), T(max(x_lo, x_hi)))

    # Convert grid to natural space for moment computation
    x_nat = [inv_tfm(η) for η in η_grid]

    # Compute moments in natural space via trapezoid rule
    # E[X] = ∫ x * p_η(η) dη  where x = g(η)
    mean_val = sum(
        (x_nat[i] * density_norm[i] + x_nat[i - 1] * density_norm[i - 1]) / 2 * Δη
            for i in 2:n
    )

    # E[X²]
    second_moment = sum(
        (x_nat[i]^2 * density_norm[i] + x_nat[i - 1]^2 * density_norm[i - 1]) / 2 * Δη
            for i in 2:n
    )

    var_val = max(zero(T), second_moment - mean_val^2)

    # Mode: maximize logpdf in natural space
    # log p_X(x) = log p_η(f(x)) + logabsdetjac(f, x)
    log_density_natural = [
        log_density_norm[i] + Bijectors.logabsdetjac(tfm, x_nat[i])
            for i in 1:n
    ]
    mode_idx = argmax(log_density_natural)
    mode_val = T(x_nat[mode_idx])

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
