export DICAccumulator, DICStrategy, DICPointSummary

"""
    DICAccumulator()

Compute Deviance Information Criterion (DIC) and effective parameters.

DIC is a Bayesian model comparison metric that balances fit and complexity:
- DIC = D̄ + p_D = 2D̄ - D(θ*)
- D̄ = E_θ[-2 log p(y|x*(θ),θ)] (mean deviance)
- p_D = D̄ - D(θ*) (effective number of parameters)
- D(θ*) = deviance at posterior mode

Lower DIC indicates better predictive model. Similar to AIC but accounts for
posterior uncertainty in hyperparameters.

# Fields (after finalize!)
- `DIC::Float64`: Deviance Information Criterion
- `p_D::Float64`: Effective number of parameters
- `D_bar::Float64`: Mean deviance across posterior
- `D_mode::Float64`: Deviance at mode

# References
Spiegelhalter et al. (2002). "Bayesian measures of model complexity and fit."
"""
mutable struct DICAccumulator <: PosteriorAccumulator
    # Accumulated data (unweighted)
    deviances::Vector{Float64}
    mode_deviance::Float64

    # Results (computed in finalize!)
    D_bar::Float64
    p_D::Float64
    DIC::Float64

    DICAccumulator() = new(Float64[], 0.0, 0.0, 0.0, 0.0)
end

"""
    DICStrategy()

Immutable config requesting DIC computation during `inla()`. Materialises into
a fresh `DICAccumulator` per run, so reusing a strategy tuple across calls is
safe.
"""
struct DICStrategy <: PosteriorStrategy end

materialize(::DICStrategy) = DICAccumulator()

"""Pre-computed summary data for one grid point (DIC)."""
struct DICPointSummary
    deviance::Float64
end

function compute_point_summary(acc::DICAccumulator; total_loglikelihood, kwargs...)
    return DICPointSummary(-2 * total_loglikelihood)
end

function accumulate!(acc::DICAccumulator, summary::DICPointSummary; is_mode::Bool = false, kwargs...)
    push!(acc.deviances, summary.deviance)
    if is_mode
        acc.mode_deviance = summary.deviance
    end
    return nothing
end

function accumulate!(
        acc::DICAccumulator;
        total_loglikelihood::Float64,
        is_mode::Bool = false,
        kwargs...
    )
    D = -2 * total_loglikelihood
    push!(acc.deviances, D)

    if is_mode
        acc.mode_deviance = D
    end

    return nothing
end

function finalize!(acc::DICAccumulator, exploration::AbstractHyperparameterExploration)
    weights = get_integration_weights(exploration)

    # Weighted mean deviance
    acc.D_bar = sum(weights .* acc.deviances)

    # Effective parameters
    acc.p_D = acc.D_bar - acc.mode_deviance

    # DIC
    acc.DIC = acc.D_bar + acc.p_D

    return nothing
end

# Pretty printing (Julia-native style)
function Base.show(io::IO, ::MIME"text/plain", acc::DICAccumulator)
    println(io, "Deviance Information Criterion (DIC):")
    println(io, "  DIC: ", round(acc.DIC, digits = 2))
    println(io, "  Effective parameters (p_D): ", round(acc.p_D, digits = 2))
    println(io, "  Mean deviance (D̄): ", round(acc.D_bar, digits = 2))
    return println(io, "  Deviance at mode: ", round(acc.mode_deviance, digits = 2))
end
