export posterior_predictive, ppc_stat, bayesian_pvalue

"""
    posterior_predictive(result::InferenceResult, n::Int; rng=Random.default_rng())

Draw `n` posterior-predictive datasets from an `inla()` / `tmb()` /
`hmc_laplace()` result.

Returns an `n × n_obs` matrix `y_rep` where row `k` is one simulated
dataset under the posterior. Each row is drawn by sampling
`(θ_k, x_k) ∼ p(θ, x | y)` and then `y_rep_k ∼ p(y | x_k, θ_k)`.

This is just a convenience shim over `rand(result, n; include_y=true)`
that pulls out the `y` matrix.

# Example
```julia
result = inla(lgm, y)
y_rep = posterior_predictive(result, 1000)   # 1000 × n_obs
```
"""
function posterior_predictive(
        result::InferenceResult, n::Int;
        rng::AbstractRNG = Random.default_rng(),
    )
    samples = rand(rng, result, n; include_y = true)
    samples.y === nothing && error(
        "posterior_predictive: underlying rand(...; include_y=true) returned no y. " *
            "This method probably doesn't support predictive draws yet."
    )
    return samples.y
end

"""
    ppc_stat(T, y, y_rep)

Evaluate a test statistic `T(::AbstractVector) -> Real` on the observed
data `y` and each posterior-predictive row of `y_rep`. Returns
`(T_obs, T_rep)` where `T_obs::Real` and `T_rep::Vector` has one value
per predictive dataset.

Common choices: `mean`, `std`, `maximum`, `x -> quantile(x, 0.95)`, or a
discrepancy statistic appropriate to the likelihood (e.g. the number of
zeros for count data).
"""
function ppc_stat(T, y::AbstractVector, y_rep::AbstractMatrix)
    T_obs = T(y)
    n = size(y_rep, 1)
    T_rep = [T(view(y_rep, k, :)) for k in 1:n]
    return T_obs, T_rep
end

"""
    bayesian_pvalue(T, y, y_rep) -> Float64

Two-sided posterior-predictive p-value for the test statistic `T`:

    p = mean(T(y_rep_k) ≥ T(y))

Values near 0 or 1 indicate that the observed statistic is in the tail
of the predictive distribution — a sign of mis-fit in that aspect. A
p-value near 0.5 means the model reproduces that statistic well.

Note this is the "tail" convention (Gelman, Meng, Stern 1996); it is
*not* a frequentist p-value and is not calibrated uniformly under the
prior.
"""
function bayesian_pvalue(T, y::AbstractVector, y_rep::AbstractMatrix)
    T_obs, T_rep = ppc_stat(T, y, y_rep)
    return mean(T_rep .>= T_obs)
end
