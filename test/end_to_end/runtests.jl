using Test

# AR-1 Poisson model test case
include("ar1_poisson/test_fast.jl")

# IID Bernoulli model test case
include("iid_bernoulli/test_fast.jl")

# SAM (non-Gaussian state-space) vs NUTS gold standard
include("sam/test_sam.jl")
