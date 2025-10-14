using StatsModels
using DataFrames
using Distributions
using GaussianMarkovRandomFields

export inla  # Export formula-based inla method

"""
    inla(formula::FormulaTerm, data::DataFrame; family=Normal, kwargs...)

Formula-based INLA inference interface.

This method provides R-INLA style formula syntax for specifying INLA models in Julia.
It automatically constructs the design matrix, precision structure, and observation model
from the formula specification.

# Arguments
- `formula::FormulaTerm`: Model formula (e.g., `@formula(y ~ x + RandomWalk(1, time))`)
- `data::DataFrame`: Data containing all variables referenced in the formula, including response

# Keyword Arguments
- `family`: Distribution family for observations (default: `Normal`)
- `hyperparameter_prior::Union{Nothing, HyperparameterPrior} = nothing`: Custom hyperparameter prior
- All other kwargs are passed to the main `inla()` method

# Returns
- `INLAResult`: Complete INLA inference results

# Example
```julia
# Create data
df = DataFrame(
    y = randn(100),
    x = randn(100),
    time = repeat(1:25, 4),
    group = repeat(1:5, 20)
)

# Run INLA with formula syntax
result = inla(@formula(y ~ x + RandomWalk(1, time)), df)

# Access results
θ_mode = result.hyperparameter_mode
marginals = result.latent_marginals
```
"""
function inla(
        formula::FormulaTerm,
        data::DataFrame;
        family = Normal,
        trials = :n,
        hyperparameter_prior::Union{Nothing, HyperparameterPrior} = nothing,
        kwargs...
    )

    # 1. Construct design matrix with arrowhead optimization
    A, terms, response = construct_design_matrix(formula, data)
    random_terms, fixed_terms = terms

    # 2. Handle Binomial family with BinomialObservations
    if family == Binomial
        # Look for trials column
        if hasproperty(data, trials)
            y = BinomialObservations(response, data[!, trials])
        else
            error("Binomial family requires column '$trials' in data. Available columns: $(names(data))")
        end
    else
        y = response
    end

    # 3. Create base observation model
    base_model = ExponentialFamily(family)

    # 4. Create linearly transformed observation model
    obs_model = LinearlyTransformedObservationModel(base_model, A)

    # 5. Create latent prior function
    function latent_prior(θ_named)
        Q = construct_gmrf_precision(random_terms, fixed_terms, data, θ_named)
        μ = zeros(size(Q, 1))
        return GMRF(μ, Q, CholeskySolverBlueprint())
    end

    # 6. Set up hyperparameter prior
    if hyperparameter_prior === nothing
        # Create default hyperparameter prior
        required_params = collect_hyperparameters(random_terms, fixed_terms)
        hp_prior = create_default_hyperparameter_prior(required_params, family)
    else
        hp_prior = hyperparameter_prior
        # Validate that all required parameters are provided
        required_params = collect_hyperparameters(random_terms, fixed_terms)
        validate_hyperparameters_in_prior(hp_prior, required_params)
    end

    # 7. Create INLAModel and run inference
    model = INLAModel(hp_prior, latent_prior, obs_model)

    # 8. Call main INLA inference
    return inla(model, y; kwargs...)
end

"""
    create_default_hyperparameter_prior(required_params, family)

Create default hyperparameter prior for formula-based INLA.

Uses R-INLA compatible defaults:
- Random walk precision: LogNormal(-1, 1) (median ≈ 0.37)
- Observation precision: Gamma(1, 0.00005) for Normal family
"""
function create_default_hyperparameter_prior(required_params, family)
    prior_specs = NamedTuple()

    # Add defaults for each required parameter
    for param in required_params
        if param == :τ_rw
            # Random walk precision: LogNormal prior (R-INLA default)
            prior_specs = merge(prior_specs, (τ_rw = LogNormal(-1.0, 1.0),))
        elseif param == :τ_iid
            # Independent effects precision: LogNormal prior
            prior_specs = merge(prior_specs, (τ_iid = LogNormal(-1.0, 1.0),))
        end
    end

    # Add observation model parameters if Normal family
    if family == Normal
        # Observation precision: Gamma prior (R-INLA default)
        prior_specs = merge(prior_specs, (τ_obs = Gamma(1.0, 0.00005),))
    end

    return HyperparameterPrior(prior_specs)
end

"""
    validate_hyperparameters_in_prior(hp_prior, required_params)

Validate that hyperparameter prior contains all required parameters.
"""
function validate_hyperparameters_in_prior(hp_prior, required_params)
    # Get all parameter names from the prior (both free and fixed)
    all_param_names = Set(keys(hp_prior.all_parameters))

    missing_params = setdiff(Set(required_params), all_param_names)

    if !isempty(missing_params)
        error("Hyperparameter prior missing required parameters: $(collect(missing_params))")
    end

    return true
end
