using Test

# AR-1 Poisson model test case
include("ar1_poisson/test_fast.jl")

# IID Bernoulli model test case
include("iid_bernoulli/test_fast.jl")

# Vector-valued hyperparameters (issue #41)
include("vector_hyperparameters/test_fast.jl")
