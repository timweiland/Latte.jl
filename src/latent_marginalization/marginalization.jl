export marginalize

"""
    marginalize(ga, obs_model, θ, y, log_prior_θ, method, indices=1:length(mean(ga)); prior_gmrf=nothing)

Compute marginal approximations for specified latent variables.

# Arguments
- `ga`: Gaussian approximation (GMRF object)
- `obs_model`: Observation model for likelihood computations
- `θ`: Hyperparameters
- `y`: Observed data
- `log_prior_θ::Real`: Log-density of hyperparameter prior
- `method::MarginalApproximation`: Approximation method
- `indices::Vector{Int}`: Variable indices to marginalize (default: all)
- `prior_gmrf`: Original prior GMRF (required for Laplace methods, ignored for Gaussian)

# Returns
`MarginalResult` containing marginal distributions and computation time.
"""
function marginalize(
        ga, obs_model, θ, y, log_prior_θ::Real,
        method::MarginalApproximation,
        indices::AbstractVector{<:Integer} = collect(1:length(mean(ga)));
        prior_gmrf = nothing
    )

    # Validate indices
    n = length(mean(ga))
    if any(i -> i < 1 || i > n, indices)
        throw(BoundsError(1:n, indices))
    end
    if length(unique(indices)) != length(indices)
        throw(ArgumentError("Duplicate indices not allowed"))
    end

    # Measure computation time
    start_time = time()
    marginals = _marginalize_impl(ga, obs_model, θ, y, log_prior_θ, method, indices, prior_gmrf)
    computation_time = time() - start_time

    return MarginalResult(indices, marginals, method, computation_time)
end
