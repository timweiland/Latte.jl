using Latte
using GaussianMarkovRandomFields
using Distributions
using LinearAlgebra
using StatsFuns
using Plots
using SparseArrays
using Random

function ar_precision(ρ, k)
    return spdiagm(-1 => -ρ * ones(k - 1), 0 => ones(k) .+ ρ^2, 1 => -ρ * ones(k - 1))
end

# Model parameters
k = 300   # number of time points

# Note: Using the new HyperparameterSpec API with @hyperparams macro
# Prior specification below

function latent_gmrf(; τ_gmrf, ρ)
    #σ = θ.σ_gmrf
    #ρ = θ.ρ

    # Create AR-1 precision matrix (from the example)
    #Q = ar_precision(ρ, k) ./ σ^2
    Q = ar_precision(ρ, k) .* τ_gmrf

    # Zero mean (simpler than the example which used μ*ones(k))

    μ₀ = log(1000.0)
    μ = μ₀ .* [ρ^i for i in 1:k]

    #@show (τ, ρ)

    return (μ, Q)
end
spec = @hyperparams begin
    (τ_gmrf ~ Exponential(50.0), transform = log, space = natural)
    (ρ ~ Normal(2.9444, 1.0), transform = logit, space = working)
end

obs_model = ExponentialFamily(Poisson)
#model = LatentGaussianModel(θ_prior, latent_gmrf, obs_model)
model = LatentGaussianModel(spec, latent_gmrf, obs_model)

Random.seed!(83498)
τ_gmrf_true = 100.0

ρ_true = 0.98   # autocorrelation coefficient
θ_true_named = (τ_gmrf = τ_gmrf_true, ρ = ρ_true)

x_gt = rand(GMRF(latent_gmrf(; θ_true_named...)...))
y_gt = rand(conditional_distribution(obs_model, x_gt))

res = inla(model, y_gt)

# Convert test parameters from vector to working/natural space using new API
θ_test_working = to_named_tuple([4.0, 2.0], model.hyperparameter_spec)
θ_test_natural = to_natural(θ_test_working, model.hyperparameter_spec)
θ_test = merge(θ_test_natural, model.hyperparameter_spec.fixed)

x_test = latent_gmrf(; θ_test...)
obs_lik = obs_model(y_gt; θ_test...)

ga = gaussian_approximation(x_test, obs_lik)

function la(θ)
    # Convert vector parameters to working/natural space
    θ_working = to_named_tuple(θ, model.hyperparameter_spec)
    θ_natural = to_natural(θ_working, model.hyperparameter_spec)
    θ_named = merge(θ_natural, model.hyperparameter_spec.fixed)

    latent_gmrf = model.latent_prior(; θ_named...)
    obs_lik = model.observation_model(y_gt; θ_named...)
    ga = gaussian_approximation(latent_gmrf, obs_lik)
    return logpdf(ga, mean(ga))
end

function latent_gmrf_ad(θ)
    σ = θ.σ_gmrf
    ρ = θ.ρ

    # Create AR-1 precision matrix (from the example)
    Q = ar_precision(ρ, k) ./ σ^2

    # Zero mean (simpler than the example which used μ*ones(k))

    μ₀ = log(1000.0)
    μ = μ₀ .* [ρ^i for i in 1:k]

    return (μ, Q)
end
@model function my_inla(y)
    σ ~ Gamma(2, 3)
    ρ ~ Uniform(0.9, 0.99)
    x ~ latent_gmrf_ad((σ_gmrf = σ, ρ = ρ))
    y ~ likelihood(obs_model, x, (σ_gmrf = σ, ρ = ρ))
end
chain = sample(my_inla(y_gt), NUTS(), 1000, progress = true)
