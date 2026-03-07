# Hospital Surgical Mortality Analysis
# Based on R-INLA Example 4.4 - Surg dataset

using IntegratedNestedLaplace
using GaussianMarkovRandomFields
using DataFrames
using Distributions
using StatsModels
using SparseArrays

# Create Surg dataset
surg_data = DataFrame(
    hospital = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"],
    n = [47, 148, 119, 810, 211, 196, 148, 215, 207, 97, 256, 360],
    r = [0, 18, 8, 46, 8, 13, 9, 31, 14, 8, 29, 24]
)

# Model: r_i ~ Binomial(n_i, p_i), logit(p_i) = α + u_i, u_i ~ N(0, σ²)
f = @formula(r ~ 1 + IID(hospital))

# Test formula interface components
A, y, obs_model, combined_model = build_formula_components(f, surg_data; family = Binomial, trials = :n)

new_iid = IIDModel(combined_model.components[1].n, constraint = :sumtozero)
new_combined = CombinedModel(new_iid, combined_model.components[2])

Q = precision_matrix(combined_model, τ_iid = 2.0)

# Results summary
println("Dataset: $(nrow(surg_data)) hospitals")
println("Design matrix: $(size(A)) ($(round(design_matrix_sparsity(A).sparsity_percent, digits = 1))% sparse)")
println("Precision matrix: $(size(Q)) ($(round((nnz(Q) / length(Q)) * 100, digits = 1))% non-zero)")

#hp_prior = HyperparameterPrior((σ = LogNormal(0.0, 1.0), τ_iid = LogNormal(0.0, 1.0)))
hp_spec = @hyperparams begin
    (τ_iid ~ PCPrior.Precision(1.0, α = 0.01), transform = log, space = natural)
end

A_constr = zeros(1, 13)
A_constr[1:12] .= 1.0
e = [0.0]

model = INLAModel(hp_spec, new_combined, obs_model)

# Test formula-based INLA with BinomialObservations
result = inla(model, y)
#result = inla(f, surg_data; family = Binomial, trials = :n)
println("INLA inference complete!")
