using Latte
using GaussianMarkovRandomFields
using Distributions
using LinearAlgebra
using StatsFuns
using Plots
using SparseArrays
using Zygote

# AR-1 precision matrix function (from the example)
function ar_precision(ρ, k)
    return spdiagm(-1 => -ρ * ones(k - 1), 0 => ones(k) .+ ρ^2, 1 => -ρ * ones(k - 1))
end

# Model parameters
k = 1000   # number of time points (smaller for speed)

# Using new @hyperparams API
spec = @hyperparams begin
    (σ_gmrf ~ Gamma(2, 3), transform = log, space = natural)
    (ρ ~ Uniform(0, 0.5), transform = logit, space = natural)
    σ = 1.0e-6  # Fixed parameter
end

# Function to create latent GMRF given hyperparameters
function latent_gmrf(; σ_gmrf, ρ, kwargs...)
    # Create AR-1 precision matrix (from the example)
    Q = ar_precision(ρ, k) ./ σ_gmrf^2

    # Zero mean (simpler than the example which used μ*ones(k))
    μ = zeros(k)

    return (μ, Q)
end
obs_model = ExponentialFamily(Poisson)
inla_model = LatentGaussianModel(spec, latent_gmrf, obs_model)
θ_true, x_true, y_true = rand(inla_model)
σ_gmrf_true, ρ_true = θ_true.σ_gmrf, θ_true.ρ

θ_test = [1.8, 0.4]
θ_test_working = to_named_tuple(θ_test, inla_model.hyperparameter_spec)
θ_test_named = merge(to_natural(θ_test_working, inla_model.hyperparameter_spec), inla_model.hyperparameter_spec.fixed)
x_test = inla_model.latent_prior(; θ_test_named...)
obs_lik_test = inla_model.observation_model(y_true; θ_test_named...)

gaussian_approximation(x_test, obs_lik_test)
