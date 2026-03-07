using Turing
using Distributions
using Random
using JLD2
using LinearAlgebra

# IID Normal prior + Bernoulli(logit(x)) model
# n=20, small enough for fast MCMC

seed = 12345
Random.seed!(seed)

n = 20

# Generate ground truth
x_true = randn(n)
p_true = 1 ./ (1 .+ exp.(-x_true))
y = [rand(Bernoulli(p)) for p in p_true]

println("Generated data: $(sum(y))/$(n) successes")

# MCMC reference via Turing
@model function bernoulli_iid_model(y, n, τ)
    x ~ MvNormal(zeros(n), (1 / τ) * I)
    for i in 1:n
        y[i] ~ BernoulliLogit(x[i])
    end
end

# Use τ = 1.0 (matching our INLA model's prior mode)
τ_fixed = 1.0

println("Running MCMC (5000 samples)...")
mcmc_start = time()
chain = sample(bernoulli_iid_model(y, n, τ_fixed), NUTS(), 5000; progress = true)
mcmc_time = time() - mcmc_start
println("MCMC completed in $(round(mcmc_time, digits = 1))s")

# Extract latent field samples
x_samples = hcat([vec(chain[Symbol("x[$i]")]) for i in 1:n]...)

model_params = (
    n = n,
    τ_fixed = τ_fixed,
    seed = seed,
    mcmc_time = mcmc_time,
)

reference_file = joinpath(@__DIR__, "reference_data.jld2")
@save reference_file y x_samples model_params

println("Saved reference data to $reference_file")
println("  x_samples size: $(size(x_samples))")
println("  MCMC means: $(round.(mean(x_samples, dims = 1)[1:5], digits = 3))...")
