using Latte
using DataFrames, Distributions
using StatsModels

# Deterministic latent effect (simulating a random walk)
f = [0.0, 0.5, 1.0]  # RW1: f[t] = f[t-1] + 0.5

# Fixed intercept
β0 = 1.0

# Compute linear predictor and rate
η = β0 .+ f
λ = exp.(η)

# Simulate Poisson response from known λ
y = [1, 4, 8]

# Time index (for random walk effect)
time = 1:3

# Construct DataFrame
df = DataFrame(
    time = time,
    y = y
)

# Print for manual inspection
println("f(t): ", f)
println("η: ", η)
println("λ: ", λ)
println("y: ", y)

df


res = inla(
    @formula(y ~ 1 + RandomWalk(1, time)),
    df,
    family = Poisson
)
