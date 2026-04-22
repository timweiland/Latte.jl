using GaussianMarkovRandomFields: ExponentialFamily, LinkFunction

export observation_marginals

# Peel Latte-side observation-model wrappers that carry per-site data but
# leave the link function on the inner ExponentialFamily.
_unwrap_to_exponential_family(m::ExponentialFamily) = m
_unwrap_to_exponential_family(m::OffsetObservationModel) = _unwrap_to_exponential_family(m.base)
_unwrap_to_exponential_family(m::BinomialTrialsObservationModel) = _unwrap_to_exponential_family(m.base)
_unwrap_to_exponential_family(m) = m

# Peel wrappers to look for an OffsetObservationModel and return its per-site
# offset vector (or `nothing` if no offset wrapper is present).
# R-INLA convention: any offset that lives inside the linear predictor is
# included in the fitted-values (i.e. μᵢ = g⁻¹(ηᵢ + offsetᵢ)). This mirrors
# that behaviour for models built by `latte_from_dppl` when the detected
# linear predictor had a non-zero constant term.
_obs_model_offset(m::OffsetObservationModel) = m.offset
_obs_model_offset(m::BinomialTrialsObservationModel) = _obs_model_offset(m.base)
_obs_model_offset(_) = nothing

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
    # Check that augmentation was used
    if result.linear_predictor_marginals === nothing
        error(
            "observation_marginals requires an augmented latent model with linear predictor marginals. " *
                "Ensure the model was created with LinearlyTransformedObservationModel and augmentation " *
                "was not disabled (augment_latent=false)."
        )
    end

    # Extract observation model, peeling Latte-side wrappers
    # (OffsetObservationModel, BinomialTrialsObservationModel) that carry
    # per-site data but leave the link function on the inner ExpFam.
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

    # Map link function to bijector. The stored linear-predictor marginals
    # hold η = A·x (no offset). Matching R-INLA's "fitted values include
    # offset" convention, we shift each site's η by its offset before
    # applying the inverse link — i.e. the effective forward bijector at
    # site i is `y ↦ g(y) - offsetᵢ`.
    bijector = get_bijector(link)
    offset_vec = _obs_model_offset(result.model.observation_model)

    # Transform each linear predictor marginal to observation space
    n_obs = length(result.linear_predictor_marginals)
    obs_marginals = Vector{TransformedWeightedMixture}(undef, n_obs)

    for i in 1:n_obs
        η_marginal = result.linear_predictor_marginals[i]
        bij_i = offset_vec === nothing ? bijector :
            Bijectors.Shift(-offset_vec[i]) ∘ bijector
        obs_marginals[i] = TransformedWeightedMixture(
            η_marginal, bij_i;
            rtol = rtol, atol = atol
        )
    end

    return obs_marginals
end
