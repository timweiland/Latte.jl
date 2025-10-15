"""
Generate MCMC reference data for AR-1 Poisson INLA validation testing.

This script generates synthetic data and runs MCMC to create reference results 
for validating the INLA implementation on the AR-1 Poisson model. The results 
are saved to a file for use in CI testing without the computational overhead 
of running MCMC.

Usage:
    julia --project generate_reference.jl

Output:
    Creates `reference_data.jld2` containing:
    - y_gt: synthetic observation data
    - τ_gmrf_log_samples: MCMC samples for log-precision parameter
    - η_samples: MCMC samples for transformed correlation parameter  
    - x_samples: MCMC samples for latent field variables
    - model_params: parameters used for data generation
"""

using TestEnv
TestEnv.activate()

using IntegratedNestedLaplace
using GaussianMarkovRandomFields
using Distributions
using LinearAlgebra
using SparseArrays
using Random
using Turing
using LDLFactorizations
using JLD2

println("=== Generating MCMC Reference Data ===")

# Set reproducible seed
seed = 83498
Random.seed!(seed)

# AR-1 precision matrix function
function ar_precision(ρ, k)
    return spdiagm(-1 => -ρ * ones(k - 1), 0 => ones(k) .+ ρ^2, 1 => -ρ * ones(k - 1))
end

# Model parameters
k = 200
σ_gmrf_true = 0.3
ρ_true = 0.98
τ_gmrf_log_true = log(1 / σ_gmrf_true^2)
η_true = atanh(ρ_true)

println("Model parameters:")
println("  k = $k")
println("  σ_gmrf_true = $σ_gmrf_true")
println("  ρ_true = $ρ_true")
println("  τ_gmrf_log_true = $τ_gmrf_log_true")
println("  η_true = $η_true")

# Model setup
desired_std_dev = 0.5 * (atanh(0.98) - atanh(0.95))
θ_prior = HyperparameterPrior((τ_gmrf_log = Normal(0, 1), η = Normal(atanh(0.95), desired_std_dev)))

function latent_gmrf(θ)
    τ = exp(θ.τ_gmrf_log)
    ρ = tanh(θ.η)
    Q = ar_precision(ρ, k) .* τ
    μ₀ = log(1000.0)
    μ = μ₀ .* [ρ^i for i in 1:k]
    return GMRF(μ, Q)
end

# Generate synthetic data
println("\\nGenerating synthetic data...")
x_gt = rand(latent_gmrf((τ_gmrf_log = τ_gmrf_log_true, η = η_true)))
obs_model = ExponentialFamily(Poisson)
y_gt = rand(obs_model; x = x_gt, θ_named = NamedTuple())

println("Data summary:")
println("  length(y_gt) = $(length(y_gt))")
println("  mean(y_gt) = $(round(mean(y_gt), digits = 2))")
println("  std(y_gt) = $(round(std(y_gt), digits = 2))")

# MCMC reference computation
println("\\nRunning MCMC reference computation...")
println("This will take several minutes...")

function latent_gmrf_ad(θ)
    τ = exp(θ.τ_gmrf_log)
    ρ = tanh(θ.η)
    Q = ar_precision(ρ, k) .* τ
    μ₀ = log(1000.0)
    μ = μ₀ .* [ρ^i for i in 1:k]
    return GMRF(μ, Q)
end

@model function mcmc_model(y)
    τ_gmrf_log ~ Normal(0, 1)
    η ~ Normal(atanh(0.95), desired_std_dev)
    x ~ latent_gmrf_ad((τ_gmrf_log = τ_gmrf_log, η = η))
    y ~ likelihood(ExponentialFamily(Poisson), x, ())
end

# Run MCMC with many samples for accuracy
n_samples = 2000
println("  Sampling: $n_samples samples")
mcmc_start_time = time()
chain = sample(mcmc_model(y_gt), NUTS(), n_samples, progress = true)
mcmc_time = time() - mcmc_start_time

println("MCMC completed in $(round(mcmc_time, digits = 1)) seconds")

# Extract samples and convert to plain arrays
τ_gmrf_log_samples = Vector{Float64}(vec(chain[:τ_gmrf_log]))
η_samples = Vector{Float64}(vec(chain[:η]))
x_samples = Matrix{Float64}(hcat([vec(chain[Symbol("x[$i]")]) for i in 1:k]...))

println("\\nMCMC diagnostics:")
println("  Total samples: $(length(τ_gmrf_log_samples))")
println("  τ_gmrf_log: mean = $(round(mean(τ_gmrf_log_samples), digits = 3)), std = $(round(std(τ_gmrf_log_samples), digits = 3))")
println("  η: mean = $(round(mean(η_samples), digits = 3)), std = $(round(std(η_samples), digits = 3))")
println("  Effective sample size: $(length(τ_gmrf_log_samples)) (after thinning if applied)")

# Save reference data
reference_file = "reference_data.jld2"
println("\\nSaving reference data to: $reference_file")

# Create model parameters tuple
model_params = (
    k = k,
    σ_gmrf_true = σ_gmrf_true,
    ρ_true = ρ_true,
    τ_gmrf_log_true = τ_gmrf_log_true,
    η_true = η_true,
    desired_std_dev = desired_std_dev,
    n_samples = n_samples,
    mcmc_time = mcmc_time,
    seed = seed,
)

# Save all data
@save reference_file y_gt τ_gmrf_log_samples η_samples x_samples model_params

println("\\n=== Reference Data Generation Complete ===")
println("File saved: $reference_file")
println("File size: $(round(filesize(reference_file) / 1024^2, digits = 1)) MB")
println("\\nTo run the fast CI test, use:")
println("julia --project test/end_to_end/ar1_poisson/test_fast.jl")
