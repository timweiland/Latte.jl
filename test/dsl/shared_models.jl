# Shared canonical models for the dsl test files. Each distinct `@model`/`@latte` function is a
# fresh DynamicPPL + AD compile specialization — the dominant cost of this suite — so files that
# need a structurally identical model reuse these definitions instead of redefining them.
# Include via:  isdefined(@__MODULE__, :shared_hier_poisson) || include("shared_models.jl")
#
# Only structurally *identical* models were merged here. Variants that pin a distinct
# recognition surface (vectorized vs indexed predictor, matrix-indexed observations,
# body-local constants, …) stay in their own files.

using Distributions
using LinearAlgebra
using DynamicPPL
using DynamicPPL: @model
using GaussianMarkovRandomFields: IIDModel
using Latte

# Hierarchical Poisson GLMM (indexed linear predictor, explicit group count).
@model function shared_hier_poisson(y, X, group, G)
    τ_u ~ Gamma(2.0, 1.0)
    β ~ MvNormal(zeros(size(X, 2)), 100.0 * I(size(X, 2)))
    u ~ MvNormal(zeros(G), (1 / τ_u) * I(G))
    for i in eachindex(y)
        y[i] ~ Poisson(exp(X[i, :] ⋅ β + u[group[i]]); check_args = false)
    end
end

# The same GLMM with a vectorized linear predictor and the group count recovered from the
# data — a distinct adapter input, kept as its own model type deliberately.
@model function shared_hier_poisson_vec(y, X, group)
    n = length(y)
    p = size(X, 2)
    G = maximum(group)
    τ_u ~ Gamma(2, 1)
    β ~ MvNormal(zeros(p), 100.0 * I(p))
    u ~ MvNormal(zeros(G), (1 / τ_u) * I(G))
    η = X * β .+ u[group]
    for i in 1:n
        y[i] ~ Poisson(exp(η[i]); check_args = false)
    end
end

# Sum-to-zero-constrained IID random effect with Poisson observations: the canonical
# constrained-prior adapter input.
@model function shared_iid_sumtozero_poisson(y, n_iid)
    τ ~ Gamma(2, 1)
    u ~ IIDModel(n_iid, constraint = :sumtozero)(τ = τ)
    for i in eachindex(y)
        y[i] ~ Poisson(exp(u[i]); check_args = false)
    end
end

# Age-structured SAM: two coupled nonlinear latent fields (logN, logF) with a survival
# recursion and a Baranov catch observation, matrix indexing throughout. The canonical
# non-Gaussian-latent model; its first build pays the expensive nested-AD compile, so the
# factor-extraction and structured-guard tests share this one model type.
@latte function shared_sam(logC, nA, nY)
    log_σN ~ Normal(-2.0, 0.5)
    log_σF ~ Normal(-2.0, 0.5)
    log_σc ~ Normal(-2.0, 0.5)
    σN = exp(log_σN); σF = exp(log_σF); σc = exp(log_σc)
    logN = Matrix{Real}(undef, nA, nY)
    logF = Matrix{Real}(undef, nA, nY)
    for a in 1:nA
        logN[a, 1] ~ Normal(8.0, 0.5)
        logF[a, 1] ~ Normal(-1.5, 0.5)
    end
    for y in 2:nY
        for a in 1:nA
            logF[a, y] ~ Normal(logF[a, y - 1], σF)
        end
        logN[1, y] ~ Normal(logN[1, y - 1], σN)
        for a in 2:nA
            logN[a, y] ~ Normal(logN[a - 1, y - 1] - exp(logF[a - 1, y - 1]) - 0.2, σN)
        end
    end
    for y in 1:nY, a in 1:nA
        Z = exp(logF[a, y]) + 0.2
        logC[(y - 1) * nA + a] ~ Normal(logN[a, y] + logF[a, y] - log(Z) + log1p(-exp(-Z)), σc)
    end
end
