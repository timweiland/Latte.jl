using GaussianMarkovRandomFields: ExponentialFamily, LinkFunction, LinearlyTransformedObservationModel

export observation_marginals

# Peel Latte-side observation-model wrappers that carry per-site data or a design
# matrix but leave the link function on the inner ExponentialFamily.
_unwrap_to_exponential_family(m::ExponentialFamily) = m
_unwrap_to_exponential_family(m::BinomialTrialsObservationModel) = _unwrap_to_exponential_family(m.base)
_unwrap_to_exponential_family(m::LinearlyTransformedObservationModel) = _unwrap_to_exponential_family(m.base_model)
_unwrap_to_exponential_family(m) = m

"""
    linear_predictor_marginals(result::INLAResult)

Marginal distributions of the linear predictor η = A·x (one per observation),
uniform across modes. Augmented models carry η as the stored η-block of the
latent marginals. Compact models do not materialize η, so they derive it from
the latent posterior via the design map, using the VBC-corrected mean and the
constraint-correct selected-inverse variance (the same moments
`observation_marginals` consumes).
"""
function linear_predictor_marginals(result::INLAResult)
    lpm = getfield(result, :linear_predictor_marginals)
    lpm === nothing || return lpm
    result.model.observation_model isa LinearlyTransformedObservationModel || error(
        "linear_predictor_marginals requires a LinearlyTransformedObservationModel " *
            "(η = A·x). Got observation model of type $(typeof(result.model.observation_model)).",
    )
    return _predictor_marginals_compact(result)
end

"""
    observation_marginals(result::INLAResult; rtol::Real = 1.0e-3, atol::Real = 1.0e-6)

Compute marginal distributions for observations (fitted values) by transforming
linear predictor marginals through the inverse link function.

# Mathematical Background

For an observation model with link function g:
- Linear predictor: η (in ℝ, typically Gaussian-like)
- Expected observation: μ = g⁻¹(η) (in observation space)

This function transforms the marginal distributions for η to obtain marginal
distributions for μ = g⁻¹(η), which represent the fitted values or expected
observations under the model.

# Requirements

This function requires:
1. **Augmented latent model**: The model must have been created with automatic
   augmentation (default for LinearlyTransformedObservationModel).
2. **ExponentialFamily observation model**: Currently only supported for
   ExponentialFamily models with extractable link functions.

# Arguments
- `result::INLAResult`: INLA inference results with linear predictor marginals
- `rtol::Real = 1.0e-3`: Relative tolerance for numerical integration in moment calculations
- `atol::Real = 1.0e-6`: Absolute tolerance for numerical integration in moment calculations

# Returns
A vector of `TransformedWeightedMixture` distributions, one for each observation,
representing the marginal distribution of the expected observation μᵢ = g⁻¹(ηᵢ).

Each distribution supports the full `Distributions.jl` interface:
- `mean(obs_marginal)`: Expected value of the observation
- `var(obs_marginal)`: Variance of the observation
- `quantile(obs_marginal, p)`: Quantiles for credible intervals
- `pdf(obs_marginal, y)`: Density evaluation
- `rand(obs_marginal)`: Sampling

# Examples
```julia
# After running INLA with augmented model
result = inla(model, y)

# Get observation marginals (fitted values)
obs_marginals = observation_marginals(result)

# Access statistics for each observation
for i in 1:length(obs_marginals)
    μ_mean = mean(obs_marginals[i])
    μ_std = std(obs_marginals[i])
    μ_ci = (quantile(obs_marginals[i], 0.025), quantile(obs_marginals[i], 0.975))
    println("Observation \$i: μ = \$μ_mean ± \$μ_std, 95% CI: \$μ_ci")
end

# For Poisson regression with log link, these represent the rate parameter λ
# For logistic regression, these represent the probability p
```

# Common Link Functions and Their Interpretations
- **LogLink (Poisson, Gamma, etc.)**: μ = exp(η) represents the rate/scale parameter
- **LogitLink (Binomial, Bernoulli)**: μ = logistic(η) represents the success probability
- **IdentityLink (Gaussian)**: μ = η is the mean directly

# Implementation Notes
- Uses `TransformedWeightedMixture` which applies change of variables to the linear
  predictor marginals.
- Moments (mean, variance) computed via numerical integration (1D quadrature).
- The bijector stored internally is the link function g; inverse is taken when needed.

# Error Handling
- Throws an error if `result.linear_predictor_marginals` is `nothing` (augmentation not used).
- Throws an error if the observation model is not an `ExponentialFamily`.
- Throws an error if the link function is not supported by `get_bijector`.
"""
function observation_marginals(
        result::INLAResult;
        rtol::Real = 1.0e-3,
        atol::Real = 1.0e-6
    )
    # Linear-predictor marginals, uniform across compact/augmented modes.
    lpm = linear_predictor_marginals(result)

    # Extract observation model, peeling Latte-side wrappers
    # (BinomialTrialsObservationModel) that carry per-site data but leave the
    # link function on the inner ExpFam.
    obs_model = _unwrap_to_exponential_family(result.model.observation_model)

    if !(obs_model isa ExponentialFamily)
        error(
            "observation_marginals currently only supports ExponentialFamily observation models " *
                "(possibly wrapped in Offset/BinomialTrials). " *
                "Got observation model of type $(typeof(result.model.observation_model))."
        )
    end

    # Extract link function
    link = obs_model.link

    # Map link function to bijector. An offset that lives inside the linear
    # predictor (η = A·x + b) is absorbed into the augmented prior mean, so the
    # stored linear-predictor marginals already hold the full ηᵢ (including the
    # offset). The fitted value is then μᵢ = g⁻¹(ηᵢ) directly: the offset is part
    # of the linear predictor, so the fitted values include it with no extra shift.
    bijector = get_bijector(link)

    # Transform each linear predictor marginal to observation space
    n_obs = length(lpm)
    obs_marginals = Vector{TransformedWeightedMixture}(undef, n_obs)

    for i in 1:n_obs
        η_marginal = lpm[i]
        obs_marginals[i] = TransformedWeightedMixture(
            η_marginal, bijector;
            rtol = rtol, atol = atol
        )
    end

    return obs_marginals
end

# Compact-mode linear-predictor marginals, computed on demand (the compact latent
# never materializes η). One WeightedMixture per observation: per integration
# point, η ~ Normal(A·μ* , √vη) with the corrected mean (μ* = μ0 under non-VBC
# methods) carrying the LTM offset via μ_η, and the constraint-correct selected-
# inverse variance vη from `linear_predictor_marginals` (NOT lincomb_variance).
function _predictor_marginals_compact(result::INLAResult)
    model = result.model
    y_obs = _get_y_obs(result)
    method = result.options.latent_marginalization_method
    exploration = result.exploration

    weights = _integration_weights(exploration)
    points = exploration.grid_points[exploration.integration_indices]
    θ_ref_nt = convert(NamedTuple, convert(NaturalHyperparameters, points[1].θ))
    ws = make_workspace(model.latent_prior; θ_ref_nt...)

    n_pts = length(points)
    A = model.observation_model.design_matrix
    n_obs = size(A, 1)
    components = [Vector{Normal{Float64}}(undef, n_pts) for _ in 1:n_obs]

    for (j, point) in enumerate(points)
        ga, prior_gmrf, obs_lik, _ = _reconstruct_ga(model, y_obs, point.θ, ws)
        μ_star = _corrected_latent_mean(method, ga, obs_lik, prior_gmrf, model)
        μ_η, v_η, _ = GaussianMarkovRandomFields.linear_predictor_marginals(ga, obs_lik)
        # μ_η = A·μ0 + offset; add A·(μ* − μ0) to get the offset-aware A·μ* + offset.
        η_mean = μ_η .+ A * (μ_star .- collect(mean(ga)))
        for i in 1:n_obs
            components[i][j] = Normal(η_mean[i], sqrt(max(v_η[i], 0.0)))
        end
    end

    return [WeightedMixture(components[i], weights) for i in 1:n_obs]
end
