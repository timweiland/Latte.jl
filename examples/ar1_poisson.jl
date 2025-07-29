using IntegratedNestedLaplace
using GaussianMarkovRandomFields
using Distributions
using LinearAlgebra
using StatsFuns
using Plots
using SparseArrays

function ar_precision(ρ, k)
    return spdiagm(-1 => -ρ * ones(k - 1), 0 => ones(k) .+ ρ^2, 1 => -ρ * ones(k - 1))
end

# Model parameters
k = 300   # number of time points

#θ_prior = HyperparameterPrior((σ_gmrf = Gamma(2, 3), ρ = Uniform(0.90, 0.99)))
# \rho --- atanh ---> \eta
# \eta --- tanh ---> \rho
desired_std_dev = 0.5 * (atanh(0.98) - atanh(0.95))
#θ_prior = HyperparameterPrior((σ_gmrf = Gamma(2, 3), η = Normal(atanh(0.95), desired_std_dev)))
θ_prior = HyperparameterPrior((τ_gmrf_log = Normal(0, 1), η = Normal(atanh(0.95), desired_std_dev)))

function latent_gmrf(θ)
    τ = exp(θ.τ_gmrf_log)
    #σ = θ.σ_gmrf
    #ρ = θ.ρ
    ρ = tanh(θ.η)

    # Create AR-1 precision matrix (from the example)
    #Q = ar_precision(ρ, k) ./ σ^2
    Q = ar_precision(ρ, k) .* τ

    # Zero mean (simpler than the example which used μ*ones(k))

    μ₀ = log(1000.0)
    μ = μ₀ .* [ρ^i for i in 1:k]

    #@show (τ, ρ)

    return GMRF(μ, Q, CholeskySolverBlueprint())
end

obs_model = ExponentialFamily(Poisson)
model = INLAModel(θ_prior, latent_gmrf, obs_model)

Random.seed!(83498)
σ_gmrf_true = 0.1   # marginal standard deviation
τ_gmrf_log_true = log(1 / σ_gmrf_true^2)
ρ_true = 0.98   # autocorrelation coefficient
η_true = atanh(ρ_true)
θ_named = (τ_gmrf_log = τ_gmrf_log_true, η = η_true)
x_gt = rand(latent_gmrf(θ_named))
y_gt = rand(data_distribution(obs_model, x_gt, θ_named))

res = inla(model, y_gt)

θ_test = to_named([4.0, 2.0], model.hyperparameter_prior)
x_test = latent_gmrf(θ_test)
obs_lik = obs_model(y_gt; θ_test...)

ga = gaussian_approximation(x_test, obs_lik)

function la(θ)
    θ_named = to_named(θ, model.hyperparameter_prior)
    latent_gmrf = model.latent_prior(θ_named)
    obs_lik = model.observation_model(y_gt; θ_named...)
    ga = gaussian_approximation_new(latent_gmrf, obs_lik)
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

    return GMRF(μ, Q, CholeskySolverBlueprint{:autodiffable}())
end

@model function my_inla(y)
    σ ~ Gamma(2, 3)
    ρ ~ Uniform(0.9, 0.99)
    x ~ latent_gmrf_ad((σ_gmrf = σ, ρ = ρ))
    y ~ likelihood(obs_model, x, (σ_gmrf = σ, ρ = ρ))
end
chain = sample(my_inla(y_gt), NUTS(), 1000, progress = true)
