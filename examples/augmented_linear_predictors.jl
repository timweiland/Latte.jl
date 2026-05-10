# Augmented Latent Field for Linear Predictor Marginals
#
# This example demonstrates automatic latent field augmentation when using
# LinearlyTransformedObservationModel. The augmentation enables computation
# of marginals for linear predictors η using full INLA approximations.
#
# Model:
# - Base latent field: x_base ~ IID(τ)  (independent components)
# - Design matrix: A (maps base latent to linear predictors)
# - Linear predictors: η = A * x_base (augmented into latent field)
# - Observations: y_i ~ Poisson(exp(η_i))
#
# Key Feature: The augmented model treats [η; x_base] as a joint latent field,
# allowing us to get marginals for η directly rather than approximating them.

using Latte
using GaussianMarkovRandomFields
using Distributions
using LinearAlgebra
using Random

println("="^70)
println("Augmented Latent Field Example: Linear Predictor Marginals")
println("="^70)
println()

# Set random seed for reproducibility
Random.seed!(12345)

# Model dimensions
n_base = 20    # Number of base latent components
n_obs = 50     # Number of observations (and linear predictors)

println("Model dimensions:")
println("  Base latent components: $n_base")
println("  Observations: $n_obs")
println("  Total augmented dimension: $(n_obs + n_base)")
println()

# Create design matrix (each observation depends on multiple base components)
# This creates a structured relationship where linear predictors combine base effects
A = randn(n_obs, n_base) / sqrt(n_base)
println("Design matrix A: $(size(A))")
println("  Sparsity: $(round(100 * count(abs.(A) .< 0.1) / length(A), digits = 1))% of entries near zero")
println()

# Base latent model (simple IID for this example)
base_latent_model = IIDModel(n_base)

# Base observation model (Poisson with log link)
base_obs_model = ExponentialFamily(Poisson)

# Wrap with LinearlyTransformedObservationModel
# This says: observations depend on η = A * x_base, not directly on x_base
obs_model = LinearlyTransformedObservationModel(base_obs_model, A)

# Hyperparameter specification
hp_spec = @hyperparams begin
    (τ ~ Exponential(1.0), transform = log, space = natural)
end

println("Creating LatentGaussianModel with automatic augmentation...")

# Create INLA model - augmentation happens automatically!
# The constructor detects LinearlyTransformedObservationModel and:
# 1. Wraps base_latent_model in AugmentedLatentModel
# 2. Creates joint latent field [η; x_base]
# 3. Maintains relationship η ≈ A * x_base via joint precision matrix
# 4. Unwraps observation model to base_obs_model
model = LatentGaussianModel(hp_spec, base_latent_model, obs_model)

println("✓ Model created with augmentation")
println("  Augmentation info: ", model.augmentation_info)
println("  Linear predictor indices: ", model.augmentation_info.linear_predictor_indices)
println("  Base latent indices: ", model.augmentation_info.base_latent_indices)
println()

# Generate synthetic data
println("Generating synthetic data...")
τ_true = 5.0
θ_true = (τ = τ_true,)

# Generate base latent field
base_gmrf = base_latent_model(τ = τ_true)
x_base_true = rand(base_gmrf)

# Compute true linear predictors
η_true = A * x_base_true

# Generate observations
λ_true = exp.(η_true)  # Poisson rate parameters
y = rand.(Poisson.(λ_true))

println("✓ Data generated")
println("  True τ: $τ_true")
println("  Linear predictor range: [$(round(minimum(η_true), digits = 2)), $(round(maximum(η_true), digits = 2))]")
println("  Observation range: [$(minimum(y)), $(maximum(y))]")
println()

# Run INLA inference
println("Running INLA inference...")
println("  (This may take a minute...)")
println()

result = inla(model, y; progress = false)

println("✓ INLA inference complete!")
println()

# Access different types of marginals
println("="^70)
println("Results: Accessing Different Marginal Types")
println("="^70)
println()

# 1. All latent marginals (includes both η and x_base)
println("1. All latent marginals:")
println("   Length: ", length(result.latent_marginals))
println("   These are marginals for the full augmented field [η₁...η₅₀; x_base₁...x_base₂₀]")
println()

# 2. Linear predictor marginals (η components)
println("2. Linear predictor marginals:")
if result.linear_predictor_marginals !== nothing
    println("   Length: ", length(result.linear_predictor_marginals))
    println("   These are marginals for η = A * x_base")
    println("   Example - η[1] marginal:")
    η1_marginal = result.linear_predictor_marginals[1]
    println("     Mean: $(round(mean(η1_marginal), digits = 3))")
    println("     Std:  $(round(std(η1_marginal), digits = 3))")
    println("     True: $(round(η_true[1], digits = 3))")
else
    println("   Not available (augmentation was not used)")
end
println()

# 3. Base latent marginals (x_base components)
println("3. Base latent marginals:")
if result.base_latent_marginals !== nothing
    println("   Length: ", length(result.base_latent_marginals))
    println("   These are marginals for the base latent components x_base")
    println("   Example - x_base[1] marginal:")
    x1_marginal = result.base_latent_marginals[1]
    println("     Mean: $(round(mean(x1_marginal), digits = 3))")
    println("     Std:  $(round(std(x1_marginal), digits = 3))")
    println("     True: $(round(x_base_true[1], digits = 3))")
else
    println("   Not available (augmentation was not used)")
end
println()

# 4. Hyperparameter marginals
println("4. Hyperparameter marginals:")
τ_marginal = result.hyperparameter_marginals.τ
println("   τ marginal:")
println("     Mean: $(round(mean(τ_marginal), digits = 3))")
println("     Std:  $(round(std(τ_marginal), digits = 3))")
println("     True: $(τ_true)")
println()

# Demonstrate accessing via indices directly
println("="^70)
println("Direct Access Examples")
println("="^70)
println()

println("Access η[5] (5th linear predictor):")
idx_η5 = 5
η5_marginal = result.latent_marginals[idx_η5]
println("  Via latent_marginals[$idx_η5]: mean = $(round(mean(η5_marginal), digits = 3))")
println("  Via linear_predictor_marginals[5]: mean = $(round(mean(result.linear_predictor_marginals[5]), digits = 3))")
println("  True value: $(round(η_true[5], digits = 3))")
println()

println("Access x_base[3] (3rd base component):")
idx_x3 = n_obs + 3  # Base components start after linear predictors
x3_marginal = result.latent_marginals[idx_x3]
println("  Via latent_marginals[$idx_x3]: mean = $(round(mean(x3_marginal), digits = 3))")
println("  Via base_latent_marginals[3]: mean = $(round(mean(result.base_latent_marginals[3]), digits = 3))")
println("  True value: $(round(x_base_true[3], digits = 3))")
println()

println("="^70)
println("Key Takeaways")
println("="^70)
println()
println("✓ Automatic augmentation enables linear predictor marginals")
println("✓ No manual GMRF construction needed - just use LinearlyTransformedObservationModel")
println("✓ Get full INLA approximations for η (not just Gaussian approximations)")
println("✓ Access both η marginals and x_base marginals separately")
println("✓ The joint structure maintains the relationship η = A * x_base")
println()
println("To opt-out of augmentation, pass augment_latent=false to LatentGaussianModel constructor")
