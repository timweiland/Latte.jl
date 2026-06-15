# Shared fixtures for accumulator test files. Kept at module scope (the
# `struct` definitions hoist out of `@testset` anyway) so name collisions
# between WAIC and CPO tests don't crop up.

using GaussianMarkovRandomFields
using Distributions

"""
    LikWithoutLPM(y, σ)

Stub Normal-like observation likelihood that intentionally lacks a
`linear_predictor_marginals` method, used to force the sample-based
fallback path in WAIC and CPO accumulator tests.
"""
struct LikWithoutLPM
    y::Vector{Float64}
    σ::Float64
end

GaussianMarkovRandomFields.pointwise_loglik(x::AbstractVector, lik::LikWithoutLPM) = [
    logpdf(Normal(x[i], lik.σ; check_args = false), lik.y[i])
        for i in eachindex(lik.y)
]
